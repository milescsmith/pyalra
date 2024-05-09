from statistics import mean, stdev

from jax import jit
import jax.numpy as jnp
import jax.typing as jt
from sklearn.utils.extmath import randomized_svd

@jit
def choose_k(
    A_norm: jt.ArrayLike,
    K: int = 100,
    thresh: int = 6,
    noise_start: int = 80,
    q: int = 2,
    **kwargs,
) -> tuple[int, jt.ArrayLike, jt.ArrayLike]:
    if K > jnp.min(A_norm.shape):
        msg = "For an m by n matrix, K must be smaller than the min(m,n)."
        raise ValueError(msg)
    if noise_start > (K - 5):
        msg = "There need to be at least 5 singular values considered noise."
        raise ValueError(msg)

    noise_svals = jnp.array(range(noise_start - 1, K))
    seed = kwargs.get("seed", None)
    _, d, _ = randomized_svd(M=A_norm, n_components=K, n_iter=q, random_state=seed)

    diffs = d[:-1] - d[1:]

    mu = mean(diffs[jnp.subtract(noise_svals, 1)])
    sigma = stdev(diffs[jnp.subtract(noise_svals, 1)])
    num_of_sds = (diffs - mu) / sigma

    k = jnp.max(jnp.where(num_of_sds > thresh)[0]) + 1

    return int(k), num_of_sds, d
