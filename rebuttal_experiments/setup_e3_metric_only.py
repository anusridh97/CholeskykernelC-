from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="e3_metric_only",
    ext_modules=[
        CppExtension(
            "e3_metric_only",
            ["e3_metric_only_ext.cpp"],
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
