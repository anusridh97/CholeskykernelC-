"""CUDA FP32 inverse-gauge updates; Triton required."""
from .inplace import InplaceConfig, InplaceWorkspace, update_inplace
