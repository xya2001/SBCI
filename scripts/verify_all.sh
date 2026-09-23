#!/usr/bin/env bash
# Run every verification tier available in this environment and summarise.
#
#   bash scripts/verify_all.sh            # tiers 1 and 2
#   bash scripts/verify_all.sh --matlab   # also regenerate the MATLAB references
#
# Each stage prints PASS or FAIL and the script exits non-zero if any failed.
# Stages that need data which is not present are reported as SKIP, never as a
# pass -- a verifier should be able to tell the difference at a glance.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

WITH_MATLAB=0
[ "${1:-}" = "--matlab" ] && WITH_MATLAB=1

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS

record() {   # record <status> <name> <detail>
    RESULTS+=("$1|$2|$3")
    case "$1" in
        PASS) PASS=$((PASS+1)) ;;
        FAIL) FAIL=$((FAIL+1)) ;;
        SKIP) SKIP=$((SKIP+1)) ;;
    esac
    printf "  %-4s  %-44s %s\n" "$1" "$2" "$3"
}

echo "=============================================================="
echo " sbci verification"
echo " $(date -u '+%Y-%m-%d %H:%M UTC')   $ROOT"
echo "=============================================================="

# --- environment ----------------------------------------------------------
echo
echo "--- environment ---"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "  no virtual environment active."
    echo "  On Longleaf:  module load python/3.12.4"
    echo "                source /work/users/x/y/xya/sbci-venv/bin/activate"
    echo "  Elsewhere:    python -m venv .venv && . .venv/bin/activate"
    echo "                pip install -e '.[dev]'"
    exit 1
fi
record PASS "python $(python -V 2>&1 | cut -d' ' -f2)" "$VIRTUAL_ENV"

# --- tier 1 ---------------------------------------------------------------
echo
echo "--- tier 1: the package alone ---"

if OUT=$(python -c "import sbci; print(sbci.__version__ if hasattr(sbci,'__version__') else 'ok')" 2>&1); then
    record PASS "sbci imports" "$OUT"
else
    record FAIL "sbci imports" "$OUT"
fi

if OUT=$(python -m pytest -q 2>&1 | tail -1); then
    record PASS "unit tests" "$OUT"
else
    record FAIL "unit tests" "$OUT"
fi

if OUT=$(ruff check . --output-format=concise 2>&1 | tail -1); then
    record PASS "ruff check" "$OUT"
else
    record FAIL "ruff check" "$OUT"
fi

if OUT=$(ruff format --check src tests 2>&1 | tail -1); then
    record PASS "ruff format" "$OUT"
else
    record FAIL "ruff format" "$OUT"
fi

TMPW=$(mktemp -d)
if OUT=$(python -m build --wheel -o "$TMPW" 2>&1 | tail -1); then
    record PASS "wheel builds" "$(basename "$(ls "$TMPW"/*.whl 2>/dev/null | head -1)")"
else
    record FAIL "wheel builds" "$OUT"
fi
rm -rf "$TMPW"

# --- tier 2 ---------------------------------------------------------------
echo
echo "--- tier 2: the documented API on real data ---"
DERIV=${SBCI_DERIVATIVES:-/work/users/x/y/xya/sbci-derivatives}
if [ -z "${SLURM_JOB_ID:-}" ]; then
    # The audit smooths, reduces and aligns on the full grid: hours of CPU
    # and gigabytes of memory, which belong on a compute node, never the login node.
    record SKIP "API audit (every README row)" "heavy compute: run inside sbatch/srun"
elif [ -f "$DERIV/sub-example_sc.h5" ]; then
    if OUT=$(python scripts/audit_api.py 2>&1 | tail -1); then
        record PASS "API audit (every README row)" "$OUT"
    else
        record FAIL "API audit (every README row)" "$OUT"
    fi
else
    record SKIP "API audit (every README row)" "no subject at $DERIV"
fi

# --- tier 4 ---------------------------------------------------------------
echo
echo "--- tier 4: against MATLAB ---"
if [ "$WITH_MATLAB" = "1" ]; then
    if command -v matlab >/dev/null 2>&1; then
        pushd tests/reference >/dev/null || exit 1
        for script in reference_run sfc_reference parcellate_reference \
                      seed_reference encore_reference; do
            if matlab -batch "run('${script}.m')" >/tmp/${script}.log 2>&1; then
                record PASS "regenerate ${script}.m" "$(tail -1 /tmp/${script}.log | cut -c1-40)"
            else
                record FAIL "regenerate ${script}.m" "see /tmp/${script}.log"
            fi
        done
        python fpca_make_inputs.py >/tmp/fpca_inputs.log 2>&1 &&
            matlab -batch "run('fpca_reference.m')" >/tmp/fpca_ref.log 2>&1 &&
            record PASS "regenerate fpca_reference.m" "ok" ||
            record FAIL "regenerate fpca_reference.m" "see /tmp/fpca_ref.log"
        popd >/dev/null || exit 1
    else
        record SKIP "regenerate MATLAB references" "matlab not on PATH"
    fi
fi

COUNT=$(python -m pytest tests/test_matlab_reference.py --collect-only -q 2>/dev/null | grep -c '::')
if OUT=$(python -m pytest tests/test_matlab_reference.py -q 2>&1 | tail -1); then
    case "$OUT" in
        *skipped*) record SKIP "MATLAB acceptance tests" "$OUT -- references absent" ;;
        *)         record PASS "MATLAB acceptance tests" "$OUT of $COUNT" ;;
    esac
else
    record FAIL "MATLAB acceptance tests" "$OUT"
fi

# --- summary --------------------------------------------------------------
echo
echo "=============================================================="
printf " %d passed, %d failed, %d skipped\n" "$PASS" "$FAIL" "$SKIP"
if [ "$SKIP" -gt 0 ]; then
    echo
    echo " Skipped stages are NOT passes. What they need:"
    for r in "${RESULTS[@]}"; do
        case "$r" in SKIP*) echo "   - ${r#SKIP|}" | sed 's/|/: /' ;; esac
    done
fi
echo "=============================================================="
[ "$FAIL" -eq 0 ]
