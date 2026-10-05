"""Build CPU and optional CUDA extensions against the installed PyTorch."""
import os
import platform
import re
import subprocess
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

root = Path("src/doublesided_cholesky/csrc")
flags = ["-O3", "-ffast-math"]
if platform.machine().lower() in ("arm64", "aarch64"):
    flags.append("-mcpu=native")
elif platform.machine().lower() in ("x86_64", "amd64"):
    flags.append("-march=native")
extensions = [CppExtension("doublesided_cholesky._C_cpu", [str(root / "cpu/fast_update.cpp")], extra_compile_args=flags)]
if torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1":
    if not CUDA_HOME:
        raise RuntimeError("CUDA build requested but CUDA toolkit/nvcc is unavailable")
    version = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version)
    modern = match is not None and tuple(map(int, match.groups())) >= (12, 8)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0;10.0+PTX" if modern else "8.0;9.0+PTX")
    extensions.append(CUDAExtension("doublesided_cholesky._C", [str(root / "cuda" / name) for name in ("bindings.cpp", "symmetric_descriptor.cu", "asymmetric_descriptor.cu", "p_tile.cu")], extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]}))
setup(ext_modules=extensions, cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)})
