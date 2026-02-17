import numpy as np
import numpy.typing as npt
from sklearn.utils.extmath import randomized_svd


# TODO add an option to use pytorch?
# maybe just CuPy or Jax?
def choose_k(
    A_norm: npt.ArrayLike, K: int = 100, thresh: int = 6, noise_start: int = 80, q: int = 2, **kwargs
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Really need a description of this, because 2 years later I don't remember exactly what it does

    Parameters
    ----------
    A_norm : npt.ArrayLike
        Log-transformed expression matrix
    K : int, default=100
        Number of singular values to compute. Must be < A_norm.shape[1]
    thresh : int, default=6
        Number of standard deviations away from noise must the singular values be?
    noise_start : int, default=80
        Index for which all smaller singular values are considered noise
    q : int, default=2
        Number of additional power iterations
    kwargs

    Returns
    -------
    A tuple containing:
        * the chosen k : int
        * P values of all k's examined : ArrayLike
        * Singluar values of A_norm : ArrayLike
    """
    if K > np.min(A_norm.shape):  # ty:ignore[unresolved-attribute] <- sorry, but A_norm definitely has a shape
        msg = "For an m by n matrix, K must be smaller than the min(m,n)."
        raise ValueError(msg)
    if noise_start > (K - 5):
        msg = "There need to be at least 5 singular values considered noise."
        raise ValueError(msg)

    noise_svals = list(range(noise_start - 1, K))
    seed = kwargs.get("seed", None)
    u, d, v = randomized_svd(M=A_norm, n_components=K, n_iter=q, random_state=seed)

    diffs = d[:-1] - d[1:]

    mu = diffs[np.subtract(noise_svals, 1)].mean()
    sigma = diffs[np.subtract(noise_svals, 1)].std()
    num_of_sds = (diffs - mu) / sigma

    k = np.where(num_of_sds > thresh)[0].max() + 1

    return k, num_of_sds, u[:, : k + 1], d[: k + 1], v[: k + 1, :]
