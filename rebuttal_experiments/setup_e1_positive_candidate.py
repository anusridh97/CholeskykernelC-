from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="e1_positive_candidate",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="e1_positive_candidate",
            sources=["e1_positive_candidate_ext.cpp"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-mavx512f",
                    "-mavx512dq",
                    "-mavx512bw",
                    "-mavx512vl",
                    "-mprefer-vector-width=512",
                    "-fopenmp",
                ]
            },
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
)
