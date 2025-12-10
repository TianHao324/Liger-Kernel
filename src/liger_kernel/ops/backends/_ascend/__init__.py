from liger_kernel.ops.backends.registry import VendorInfo, register_vendor

# Register Ascend vendor for NPU device
register_vendor(
    VendorInfo(
        vendor="ascend",
        device="npu",
        module_path="liger_kernel.ops.backends._ascend.ops",
    )
)

