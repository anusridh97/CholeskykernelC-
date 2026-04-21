from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="fast_cholesky",
    ext_modules=[
        CppExtension(
            "fast_cholesky",
            ["fast_update.cpp"],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
