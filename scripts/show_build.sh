#!/bin/bash
# Walk the whole build, printing what each stage actually produces.
set -uo pipefail
if ! command -v module >/dev/null 2>&1; then source /usr/share/lmod/lmod/init/bash; fi
module load python/3.12.4 >/dev/null 2>&1
source /work/users/x/y/xya/sbci-venv/bin/activate
cd "$HOME/sbci"

line() { printf '\n=== %s ===\n' "$1"; }

line "1. SOURCE: what the repository holds"
printf '  source modules : %s .py files under src/sbci\n' "$(find src -name '*.py' | wc -l | tr -d ' ')"
printf '  bundled data   : %s .npz files (atlases + surfaces)\n' "$(find src -name '*.npz' | wc -l | tr -d ' ')"
printf '  tests          : %s test files\n' "$(find tests -name 'test_*.py' | wc -l | tr -d ' ')"
printf '  total tracked  : %s files, %s\n' "$(git ls-files | wc -l | tr -d ' ')" "$(du -sh --exclude=.git . | cut -f1)"

line "2. DECLARATION: what pyproject.toml tells the build system"
python - <<'PY'
import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
p, b = cfg["project"], cfg["build-system"]
print(f"  name            : {p['name']}")
print(f"  version         : {p['version']}")
print(f"  requires-python : {p['requires-python']}")
print(f"  build backend   : {b['build-backend']}  (from {', '.join(b['requires'])})")
print(f"  hard deps       : {', '.join(p['dependencies'])}")
print(f"  optional extras : {', '.join(p.get('optional-dependencies', {}))}")
print(f"  console script  : {list(p.get('scripts', {}).items())}")
print(f"  wheel contents  : {cfg['tool']['hatch']['build']['targets']['wheel']['packages']}")
PY

line "3. DEV INSTALL: pip install -e '.[dev]'  (already done here)"
python -c "
import sbci, pathlib
print('  import resolves to :', pathlib.Path(sbci.__file__).parent)
print('  version            :', sbci.__version__)
print('  editable install means this points at src/, so edits take effect')
print('  without reinstalling.')
"

line "4. CHECK: lint and tests must pass before building"
ruff check src tests tools >/dev/null 2>&1 && echo "  ruff check        : clean" || echo "  ruff check        : FAILED"
pytest -q 2>&1 | tail -1 | sed 's/^/  pytest            : /'

line "5. BUILD: python -m build  (pyproject -> sdist + wheel)"
rm -rf dist
python -m build 2>&1 | grep -E "Successfully built|Building" | sed 's/^/  /'
ls -la dist/ | tail -n +2 | awk '{printf "  %-46s %8.1f KB\n", $9, $5/1024}'

line "6. WHAT IS INSIDE THE WHEEL"
python - <<'PY'
import glob, zipfile, collections
wheel = glob.glob("dist/*.whl")[0]
names = zipfile.ZipFile(wheel).namelist()
kinds = collections.Counter()
for n in names:
    kinds[n.rsplit(".", 1)[-1] if "." in n else "other"] += 1
print(f"  {len(names)} entries:")
for ext, count in kinds.most_common():
    print(f"    .{ext:<12s} {count:4d}")
print("  key metadata files:")
for n in sorted(names):
    if "dist-info" in n:
        print(f"    {n}")
PY

line "7. INSTALL INTO A CLEAN ENVIRONMENT (the 'stranger' test)"
CLEAN=$(mktemp -d)/venv
python3 -m venv "$CLEAN"
"$CLEAN/bin/pip" install -q dist/*.whl
printf '  installed packages: %s\n' "$("$CLEAN/bin/pip" list --format=freeze | wc -l | tr -d ' ')"
"$CLEAN/bin/pip" list --format=freeze | sed 's/^/    /'

line "8. VERIFY IT WORKS FROM SOMEWHERE ELSE ENTIRELY"
cd /tmp
"$CLEAN/bin/python" - <<'PY'
from sbci import ContinuousConnectome, load_atlas, list_atlases
print("  import from /tmp   : ok")
print("  atlases available  :", len(list_atlases()))
print("  Schaefer200        :", load_atlas("Schaefer200").n_regions, "regions")
PY
printf '  console script     : %s\n' "$("$CLEAN/bin/sbci" --version)"
rm -rf "$(dirname "$CLEAN")"

line "9. NOT DONE: publishing"
echo "  twine upload dist/*   would push to PyPI, after which"
echo "  'pip install sbci' works for anyone. The name is free; nothing"
echo "  has been uploaded."
