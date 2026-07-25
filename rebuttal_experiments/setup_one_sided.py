from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="one_sided_cholesky",
    ext_modules=[
        CppExtension(
            "one_sided_cholesky",
            ["one_sided_ext.cpp"],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
