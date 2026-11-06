#!/usr/bin/env bash
# Use an activated virtual environment or a preconfigured CUDA container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${1:-cpu}"
py="${PYTHON:-python}"
case "$mode" in
  cpu)
    "$py" -m pip install 'torch>=2.8' numpy 'setuptools>=69' wheel pytest
    "$py" -m pip install --no-build-isolation --no-deps .
    ;;
  b200)
    # Keep the container's matched Torch/CUDA/Triton installation.
    "$py" -c 'import torch,triton; assert torch.cuda.is_available(); print(torch.__version__, triton.__version__, torch.cuda.get_device_name())'
    "$py" -m pip install 'setuptools>=69' wheel pytest numpy
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}" \
      "$py" -m pip install --no-build-isolation --no-deps -e .
    ;;
  applications)
    "$py" -m pip install 'torch>=2.8' numpy scipy 'setuptools>=69' wheel pytest
    "$py" scripts/build_applications.py build_ext --inplace
    "$py" -c 'from benchmarks.applications._corrected.core import extension_available; assert extension_available(), "Native application backend failed to import"'
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$py" -m pytest tests/test_application_core.py tests/test_application_models.py -q
    ;;
  well)
    # Run this in a separate CPU environment; package compilation is unnecessary.
    "$py" -m pip install numpy==2.2.6 scipy==1.15.3 h5py==3.14.0 huggingface_hub
    ;;
  *)
    echo "Usage: bash scripts/setup.sh {cpu|b200|applications|well}" >&2
    exit 2
    ;;
esac
