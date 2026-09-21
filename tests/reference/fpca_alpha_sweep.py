"""Where does the roughness penalty stop smoothing and start inverting?

The reference picks the component with `eigs(M, 1)`, whose default is the
eigenvalue of largest *magnitude*. The penalty enters as `- alpha * R`, so once
`alpha * R` dominates the data term the largest-magnitude eigenvalue is the most
negative one -- the roughest direction, not the smoothest.
"""

import numpy as np

from sbci.reduction import fit_basis


def main():
    rng = np.random.default_rng(4)
    n, n_subjects, true_rank = 24, 12, 3
    truth = rng.standard_normal((n, true_rank))
    truth /= np.linalg.norm(truth, axis=0, keepdims=True)
    weights = rng.standard_normal((n_subjects, true_rank)) * np.array([4.0, 2.0, 1.0])
    matrices = np.stack(
        [
            sum(weights[i, k] * np.outer(truth[:, k], truth[:, k]) for k in range(true_rank))
            for i in range(n_subjects)
        ]
    )
    noise = rng.standard_normal(matrices.shape) * 0.05
    matrices = matrices + (noise + np.transpose(noise, (0, 2, 1))) / 2
    matrices -= matrices.mean(axis=0, keepdims=True)

    ring = np.zeros((n, n))
    for i in range(n):
        ring[i, i] = 2.0
        ring[i, (i + 1) % n] = ring[(i + 1) % n, i] = -1.0

    print(
        f"data term scale: |M| ~ {np.abs(matrices).max():.3f};  "
        f"roughness eigenvalues up to {np.linalg.eigvalsh(ring).max():.3f}"
    )
    print(f"\n{'alpha':>10} {'roughness of the component':>28} {'explained':>11}")
    for alpha in (0.0, 1e-10, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        result = fit_basis(matrices, np.eye(n), ring, rank=1, alpha=alpha, seed=0)
        vector = result.basis[:, 0]
        print(f"{alpha:10g} {vector @ ring @ vector:28.6f} {result.explained[-1]:11.4f}")


if __name__ == "__main__":
    main()
