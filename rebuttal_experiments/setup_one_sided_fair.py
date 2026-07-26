from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="one_sided_fair",
    ext_modules=[
        CppExtension(
            "one_sided_fair",
            ["one_sided_fair_ext.cpp"],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-ffast-math",
                "-fno-math-errno",
                "-DNDEBUG",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
