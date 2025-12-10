from liger_kernel.ops.backends.registry import (
    VendorInfo,
    VENDOR_REGISTRY,
    register_vendor,
    get_vendor_for_device,
)

# Import vendor packages to trigger registration
# Each vendor's __init__.py calls register_vendor() when imported
from liger_kernel.ops.backends import _ascend  # noqa: F401

