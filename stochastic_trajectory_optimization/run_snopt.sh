#!/usr/bin/env bash
# Run the double-integrator example with the locally installed licensed SNOPT build.
set -eu
cd "$(dirname "$0")/.."
export LD_PRELOAD="${SNOPT_HELPER_LIBRARY:-$PWD/solvers/snopt/libsnopt_helpers_units.so}${LD_PRELOAD:+:$LD_PRELOAD}"
export SNOPT_LICENSE="${SNOPT_LICENSE:-$PWD/solvers/snopt/snopt7.lic}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
exec "${SNOPT_PYTHON:-python3}" "$@"
