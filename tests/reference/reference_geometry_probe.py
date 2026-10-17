"""``reference_geometry.py`` against the removed ``reference=True``, on the axis and beside it.

    git worktree add ../sbci-b25f6f2 b25f6f2
    PYTHONPATH=../sbci-b25f6f2/src python tests/reference/reference_geometry_probe.py save OUT.npz
    PYTHONPATH=src python tests/reference/reference_geometry_probe.py compare OUT.npz

b25f6f2 is the last commit whose ``SphericalGrid`` takes ``reference=True``. ``save`` builds that
grid's basis and divergence on the level-2 icosphere of ``tests/test_alignment.py``, which has a
vertex on each pole, at orders 3, 6 and 15, with the poles turned off the axis by 0 to 2e-5
radians. ``compare`` builds the package's grid on the same vertices, puts the reference's errors
back with ``as_the_reference_forms_them``, and prints the largest difference of each, relative
to its largest entry. PORTING.md item 23 quotes the output.
"""

import ast
import sys
from pathlib import Path

import numpy as np

from sbci.alignment import SphericalGrid

HERE = Path(__file__).resolve().parent
ORDERS = (3, 6, 15)
OFFSETS = (0.0, 1e-7, 1e-6, 5e-6, 9e-6, 2e-5)


def icosphere():
    tree = ast.parse((HERE.parent / "test_alignment.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "icosphere")
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), "icosphere", "exec"), namespace)
    return namespace["icosphere"](2)


def turned(vertices, offset):
    c, s = np.cos(offset), np.sin(offset)
    return vertices @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]).T


def relative(ours, theirs):
    return np.abs(ours - theirs).max() / np.abs(theirs).max()


def main(mode, out):
    vertices, faces = icosphere()
    if mode == "save":
        saved = {}
        for order in ORDERS:
            for k, offset in enumerate(OFFSETS):
                grid = SphericalGrid(turned(vertices, offset), faces, order, reference=True)
                saved[f"basis_{order}_{k}"] = grid.basis
                saved[f"laplacian_{order}_{k}"] = grid.laplacian
        np.savez(out, **saved)
        print(f"saved {out}")
        return
    sys.path.insert(0, str(HERE))
    from reference_geometry import as_the_reference_forms_them

    saved = np.load(out)
    for order in ORDERS:
        for k, offset in enumerate(OFFSETS):
            grid = SphericalGrid(turned(vertices, offset), faces, order)
            basis, laplacian = as_the_reference_forms_them(grid)
            print(
                f"order {order:2d}, poles {offset:7.1e} rad off the axis: "
                f"basis {relative(basis, saved[f'basis_{order}_{k}']):.1e}, "
                f"divergence {relative(laplacian, saved[f'laplacian_{order}_{k}']):.1e}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("save", "compare"):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
