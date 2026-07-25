#!/bin/bash
#SBATCH --job-name=blackhole-prepare
#SBATCH --account=marlowe-m000151
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=rebuttal_logs/blackhole-prepare-%j.out
#SBATCH --error=rebuttal_logs/blackhole-prepare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# The repository-local dependency directory contains Python-3.10 wheels for
# NumPy, SciPy, and h5py.  Use /usr/bin/python3 explicitly: the cluster's
# default module currently places Python 3.9 first, which is ABI-incompatible
# with those wheels.
export PYTHONPATH="${PWD}/rebuttal_deps:${PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

raw_path="${BLACKHOLE_RAW_PATH:-${PWD}/data/the_well/post_neutron_star_merger/data/train/post_neutron_star_merger_scenario_4.hdf5}"
trace_path="${BLACKHOLE_TRACE_PATH:-${PWD}/data/the_well/processed/post_neutron_star_merger_scenario_4_dct65536.npy}"
manifest_path="${BLACKHOLE_MANIFEST_PATH:-${PWD}/rebuttal_results/blackhole/blackhole-s4-v1/feature_manifest.json}"

force_args=()
if [[ "${BLACKHOLE_FORCE:-0}" == "1" ]]; then
  force_args+=(--force)
fi

/usr/bin/python3 - <<'PY'
import h5py
import numpy
import scipy

print(
    "black-hole preparation dependencies:",
    f"h5py={h5py.__version__}",
    f"numpy={numpy.__version__}",
    f"scipy={scipy.__version__}",
)
PY

/usr/bin/python3 -m rebuttal_experiments.blackhole_features \
  --raw "${raw_path}" \
  --trace "${trace_path}" \
  --manifest "${manifest_path}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "${force_args[@]}"
