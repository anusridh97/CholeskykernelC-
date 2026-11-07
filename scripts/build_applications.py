"""Build the preserved signed CPU application backend; run from repository root."""
from pathlib import Path
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).resolve().parents[1]
flags = ["-O3", "-std=c++20"]
if platform.machine().lower() in ("arm64", "aarch64"):
    flags.append("-mcpu=native")
elif platform.machine().lower() in ("x86_64", "amd64"):
    flags.append("-march=native")

setup(
    name="doublesided-cholesky-applications",
    packages=[],
    package_dir={"": "."},
    ext_modules=[CppExtension(
        "benchmarks.applications._corrected._C",
        [str(root / "benchmarks/applications/_corrected/fast_update.cpp")],
        extra_compile_args=flags,
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
