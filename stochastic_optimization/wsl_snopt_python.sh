#!/bin/bash
# Run the WSL `project` Python with the installed SNOPT helper library and
# licence (solvers/snopt is local and not versioned). Arguments are passed to
# Python from the repository root, e.g.
#   JAX_PLATFORMS=cuda bash stochastic_optimization/wsl_snopt_python.sh stochastic_traj_opt_jax.py --case psyche
set -eu
cd "$(dirname "$0")/.."
export LD_PRELOAD="$PWD/solvers/snopt/libsnopt_helpers_units.so"
export SNOPT_LICENSE="$PWD/solvers/snopt/snopt7.lic"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
exec "${PSYCHE_PYTHON:-/home/lukas_messina/anaconda3/envs/project/bin/python}" "$@"
