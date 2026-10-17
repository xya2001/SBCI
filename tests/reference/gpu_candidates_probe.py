"""How many faces around a point's nearest centroid hold the twelve its k-d tree finds.

    python tests/reference/gpu_candidates_probe.py

On the CPU, ``MeshQuery`` narrows each point to the twelve faces whose centroids a k-d tree
finds nearest it. On the GPU it narrows each point to the faces whose centroids lie nearest
the centroid nearest the point, a table made once a mesh with three times as many, 36
(PORTING.md item 26). For each mesh this prints how many of those the table needs to hold
every one of the tree's twelve, for 400,000 random points of the sphere and for the
vertices themselves: the bundled ico4 hemispheres, as stored and as each alignment turns
them, and icospheres of one to three subdivisions. CPU only; a few minutes.
"""

import numpy as np

from sbci.alignment import DEFAULT_ORDER, MeshQuery, _hemisphere_grids, normalize_rows
from sbci.conseal import default_grids, icosphere

KEPT = 80  # the faces around each centroid that are ranked; more than any mesh here needs


def needed(vertices, faces, points):
    """The fewest faces around the nearest centroid that hold the tree's twelve at every point."""
    query = MeshQuery(vertices, faces)
    centroids = (query._a + query._b + query._c) / 3.0
    kept = min(KEPT, faces.shape[0])
    _, around = query._tree.query(centroids, k=kept)
    around = around.reshape(centroids.shape[0], kept)
    # as the GPU finds it: |p - c|^2 / 2 less the point's own |p|^2 / 2
    half_squared = 0.5 * (centroids**2).sum(axis=1)
    worst = 0
    for start in range(0, points.shape[0], 20000):
        chunk = points[start : start + 20000]
        _, twelve = query._tree.query(chunk, k=query.candidates)
        twelve = twelve.reshape(chunk.shape[0], query.candidates)
        nearest = np.argmin(half_squared[None, :] - chunk @ centroids.T, axis=1)
        ranked = around[nearest]
        # where each of the twelve stands in its nearest centroid's ranking (1 = first)
        place = np.full(twelve.shape, kept + 1)
        for rank in range(kept):
            hit = ranked[:, rank, None] == twelve
            place[hit] = np.minimum(place[hit], rank + 1)
        worst = max(worst, int(place.max()))
    return worst


def main():
    points = normalize_rows(np.random.default_rng(0).normal(size=(400_000, 3)))
    conseal_lh, conseal_rh = default_grids()
    encore_lh, encore_rh = _hemisphere_grids(DEFAULT_ORDER)
    meshes = {
        "ico4 left, ConSEAL's grid (as stored)": (conseal_lh.vertices, conseal_lh.faces),
        "ico4 right, ConSEAL's grid": (conseal_rh.vertices, conseal_rh.faces),
        "ico4 left, ENCORE's grid (turned off the poles)": (encore_lh.vertices, encore_lh.faces),
        "ico4 right, ENCORE's grid": (encore_rh.vertices, encore_rh.faces),
        "icosphere(1)": icosphere(1),
        "icosphere(2)": icosphere(2),
        "icosphere(3)": icosphere(3),
    }
    for name, (mesh_vertices, mesh_faces) in meshes.items():
        print(
            f"{name}: {mesh_faces.shape[0]} faces; the tree's twelve within the nearest "
            f"centroid's first {needed(mesh_vertices, mesh_faces, points)} at 400,000 random "
            f"points, {needed(mesh_vertices, mesh_faces, mesh_vertices)} at the vertices",
            flush=True,
        )


if __name__ == "__main__":
    main()
