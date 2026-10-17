"""Where MeshQuery's narrowed search cannot settle a point, and whether its face moves.

    python tests/reference/mesh_query_probe.py

``MeshQuery`` keeps the best of the twelve faces whose centroids lie nearest a point only when
no face outside them can be as near, and otherwise tests every face near enough to matter
(PORTING.md item 27). For each sphere the package searches -- the bundled ico4 hemispheres as
ENCORE turns them and as ConSEAL keeps them, the fs_LR 32k spheres as the HCP gives them and
carried to fsaverage (``templates.fsaverage_to_fslr``), and the grid on fsaverage
(``templates.grid_to_fsaverage``) -- this prints how many of 100,000 random points and of the
mesh's vertices go past the twelve, and how many of those end in another face than the best of
the twelve, the face the search kept before item 27. CPU only; a minute or two.
"""

import numpy as np

from sbci.alignment import MeshQuery, _closest_on_triangles, _hemisphere_grids, normalize_rows
from sbci.templates import _fslr_spheres, fsaverage_to_fslr, grid_to_fsaverage


def best_of_the_candidates(query, points):
    """The face the narrowed search alone keeps: the best of the twelve, the lowest index first."""
    _, near = query._tree.query(points, k=query.candidates)
    near = np.sort(near.reshape(points.shape[0], query.candidates), axis=1)
    closest = _closest_on_triangles(
        points[:, None, :], query._a[near], query._b[near], query._c[near]
    )
    distance = ((closest - points[:, None, :]) ** 2).sum(-1)
    return near[np.arange(points.shape[0]), np.argmin(distance, axis=1)]


def main():
    points = normalize_rows(np.random.default_rng(0).normal(size=(100_000, 3)))
    meshes = {}
    for name, rotate in (("ENCORE's", True), ("ConSEAL's", False)):
        for side, grid in zip(("left", "right"), _hemisphere_grids(1, rotate=rotate), strict=True):
            meshes[f"ico4 {side}, {name} grid"] = (grid.vertices, grid.faces)
    spheres = _fslr_spheres()
    for side in "LR":
        meshes[f"fs_LR 32k {side}, standard"] = (
            spheres[f"{side}_standard"],
            spheres[f"{side}_faces"],
        )
    for side, carried in zip("LR", fsaverage_to_fslr(), strict=True):
        meshes[f"fs_LR 32k {side}, carried to fsaverage"] = (carried.source, carried.faces)
    left, _ = grid_to_fsaverage()
    meshes["the grid on fsaverage, left"] = (left.target, left.faces)

    for name, (vertices, faces) in meshes.items():
        vertices = normalize_rows(np.asarray(vertices, dtype=np.float64))
        query = MeshQuery(vertices, np.asarray(faces))
        for label, located in (("random points", points), ("vertices", vertices)):
            unsettled = []
            narrowed = query._nearest_faces

            def counted(chosen, gap, narrowed=narrowed, unsettled=unsettled):
                unsettled.append(chosen)
                return narrowed(chosen, gap)

            query._nearest_faces = counted
            _, _, face = query.query_faces(located)
            del query._nearest_faces
            past = np.vstack(unsettled) if unsettled else np.empty((0, 3))
            moved = 0
            if past.shape[0]:
                _, _, settled = query.query_faces(past)
                moved = int(np.count_nonzero(settled != best_of_the_candidates(query, past)))
            print(
                f"{name}, {len(located):,} {label}: {past.shape[0]:,} past the twelve "
                f"({100 * past.shape[0] / len(located):.2f}%), {moved} of them in another face",
                flush=True,
            )


if __name__ == "__main__":
    main()
