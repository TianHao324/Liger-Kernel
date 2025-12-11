import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // sl
    cos_row_idx = pid % sl

    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    q_base = q_ptr + pid * q_row_stride
    k_base = k_ptr + pid * k_row_stride

    # Process in chunks to prevent UB overflow
    for qh_block in range(0, n_qh, BLOCK_Q):
        qh_idx = tl.arange(0, BLOCK_Q) + qh_block

        for d_block in range(0, hd // 2, BLOCK_D):
            d_idx = tl.arange(0, BLOCK_D) + d_block

            qh_mask = qh_idx < n_qh
            d_mask = d_idx < (hd // 2)
            block_mask = qh_mask[:, None] & d_mask[None, :]

            offsets = qh_idx[:, None] * hd + d_idx[None, :]

            cos_vals = tl.load(cos + d_idx, mask=d_mask, other=0)
            sin_vals = tl.load(sin + d_idx, mask=d_mask, other=0)


            q_left = tl.load(q_base + offsets, mask=block_mask, other=0)
            q_right = tl.load(q_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = q_left * cos_vals - q_right * sin_vals
                new_right = q_right * cos_vals + q_left * sin_vals
            else:
                new_left = q_left * cos_vals + q_right * sin_vals
                new_right = q_right * cos_vals - q_left * sin_vals

            tl.store(q_base + offsets, new_left, mask=block_mask)
            tl.store(q_base + offsets + (hd // 2), new_right, mask=block_mask)

    for kh_block in range(0, n_kh, BLOCK_K):
        kh_idx = tl.arange(0, BLOCK_K) + kh_block

        for d_block in range(0, hd // 2, BLOCK_D):
            d_idx = tl.arange(0, BLOCK_D) + d_block

            kh_mask = kh_idx < n_kh
            d_mask = d_idx < (hd // 2)
            block_mask = kh_mask[:, None] & d_mask[None, :]

            offsets = kh_idx[:, None] * hd + d_idx[None, :]

            cos_vals = tl.load(cos + d_idx, mask=d_mask, other=0)
            sin_vals = tl.load(sin + d_idx, mask=d_mask, other=0)

            k_left = tl.load(k_base + offsets, mask=block_mask, other=0)
            k_right = tl.load(k_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = k_left * cos_vals - k_right * sin_vals
                new_right = k_right * cos_vals + k_left * sin_vals
            else:
                new_left = k_left * cos_vals + k_right * sin_vals
                new_right = k_right * cos_vals - k_left * sin_vals

            tl.store(k_base + offsets, new_left, mask=block_mask)
            tl.store(k_base + offsets + (hd // 2), new_right, mask=block_mask)

def rope_forward(q, k, cos, sin):
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]

    BLOCK_Q = min(32, triton.next_power_of_2(pad_n_q_head))
    BLOCK_K = min(32, triton.next_power_of_2(pad_n_kv_head))
    BLOCK_D = min(64, triton.next_power_of_2(pad_hd // 2))
    _triton_rope_npu[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
        BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    cos_batch_size = cos.shape[0]
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    n_row = batch_size * seq_len

    # ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    # backward is similar to forward except swapping few ops

    BLOCK_Q = min(32, triton.next_power_of_2(pad_n_q_head))
    BLOCK_K = min(32, triton.next_power_of_2(pad_n_kv_head))
    BLOCK_D = min(64, triton.next_power_of_2(pad_hd // 2))
    _triton_rope_npu[(n_row,)](
        dq,
        dq.stride(1),
        dk,
        dk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
        BACKWARD_PASS=True,
    )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerRopeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """

        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
