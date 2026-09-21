#!/bin/bash
#SBATCH --job-name=sbci-tests
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

# Run the test suite as a batch job. The login node is fine for this suite
# today (it runs in well under a second), but the same script will carry the
# full-grid and tutorial-subject tests once the data release exists, and those
# do not belong on a login node.
set -euo pipefail

ONYEN="$(whoami)"
WORK="/work/users/${ONYEN:0:1}/${ONYEN:1:1}/${ONYEN}"
VENV="${SBCI_VENV:-$WORK/sbci-venv}"

# `module` is a shell function Lmod defines in login shells. It does not always
# survive into a script, so initialize it if it is missing rather than failing
# with "module: command not found".
if ! command -v module >/dev/null 2>&1; then
    if [ -n "${LMOD_PKG:-}" ] && [ -f "$LMOD_PKG/init/bash" ]; then
        # shellcheck disable=SC1091
        source "$LMOD_PKG/init/bash"
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    else
        echo "error: Lmod not found; run this on a Longleaf login or compute node" >&2
        exit 1
    fi
fi

module load "${PYTHON_MODULE:-python/3.12.4}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$HOME/sbci}"
pytest -v --cov=sbci --cov-report=term-missing
