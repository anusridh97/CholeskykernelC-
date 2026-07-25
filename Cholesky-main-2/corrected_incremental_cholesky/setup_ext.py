from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="fast_cholesky",
    version="0.2.0",
    ext_modules=[
        CppExtension(
            name="fast_cholesky",
            sources=["fast_update.cpp"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17", "-march=native"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
