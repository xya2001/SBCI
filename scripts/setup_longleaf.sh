#!/bin/bash
# Create (or recreate) the development environment for sbci on Longleaf.
#
# The virtualenv lives on /work rather than in $HOME because home quotas are
# 50 GB and a scientific-Python environment is several hundred megabytes. /work
# is scratch and is purged, so this script is the way the environment comes
# back: it is disposable by design and nothing unreproducible belongs in it.
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.4}"
ONYEN="$(whoami)"
WORK="/work/users/${ONYEN:0:1}/${ONYEN:1:1}/${ONYEN}"
VENV="${SBCI_VENV:-$WORK/sbci-venv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "repo:   $REPO"
echo "venv:   $VENV"
echo "module: $PYTHON_MODULE"

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

module load "$PYTHON_MODULE"
python3 --version

if [ ! -d "$VENV" ]; then
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e "$REPO[dev]"

echo
echo "Ready. Activate it with:"
echo "    module load $PYTHON_MODULE && source $VENV/bin/activate"
