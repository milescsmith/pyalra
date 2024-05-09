from jax import jit
import jax.numpy as jnp
import jax.typing as jt
import scipy as sp
from loguru import logger
from sklearn.utils.extmath import randomized_svd

from pyalra.choose_k import choose_k
from pyalra.logging import init_logger

@jit
def alra(
    a_norm: jt.ArrayLike | sp.sparse.spmatrix,
    k: int = 0,
    q: int = 10,
    quantile_prob: float = 0.001,
    seed: float | None = None,
    debug: bool = False,
    *,
    save_log: bool = False,
    log_level: int = 1,
) -> dict[str, jt.ArrayLike]:
    if debug:
        init_logger(log_level, save_log)

    logger.info(f"Read matrix with {a_norm.shape[0]} cells and {a_norm.shape[1]} genes")

    # if isinstance(a_norm, np.matrix):
    #     a_norm = a_norm.A
    # elif isinstance(a_norm, sp.sparse.spmatrix):
    #     a_norm = a_norm.todense().A

    if k == 0:
        k, *_ = choose_k(a_norm)
        logger.info(f"Chose k = {k}")

    logger.info("Getting nonzeros\n")
    originally_nonzero = a_norm > 0

    logger.debug("Running rsvd")
    u, d, v = randomized_svd(M=a_norm, n_components=k, n_iter=q, random_state=seed)

    a_norm_rank_k = jnp.matmul(jnp.matmul(u[:, :k], jnp.diag(d[:k])), v[:k, :])

    logger.info(f"Find the {quantile_prob} quantile for each gene")
    a_norm_rank_k_mins = abs(jnp.quantile(a_norm_rank_k, axis=0, q=0.001))

    logger.info("Sweep")
    a_norm_rank_k_cor = a_norm_rank_k.copy()
    a_norm_rank_k_cor[a_norm_rank_k <= jnp.tile(a_norm_rank_k_mins, (len(a_norm_rank_k), 1))] = 0

    sigma_1 = sp.stats.tstd(a_norm_rank_k_cor, axis=0)
    sigma_2 = sp.stats.tstd(a_norm, axis=0)
    mu_1 = jnp.divide(jnp.sum(a_norm_rank_k_cor, axis=0), jnp.sum(a_norm_rank_k_cor > 0, axis=0))
    mu_2 = jnp.divide(jnp.sum(a_norm, axis=0), jnp.sum(a_norm > 0, axis=0))

    toscale = jnp.logical_and(
        jnp.logical_and(~jnp.isnan(sigma_1), ~jnp.isnan(sigma_2)),
        jnp.logical_and(~jnp.logical_and((sigma_1 == 0), (sigma_2 == 0)), ~(sigma_1 == 0)),
    )

    logger.info(f"Scaling all except for {sum(~toscale)} columns")

    sigma_1_2 = jnp.divide(sigma_2, sigma_1)

    toadd = jnp.add(-jnp.divide(jnp.multiply(mu_1, sigma_2), sigma_1), mu_2)

    a_norm_rank_k_temp = a_norm_rank_k_cor[:, toscale].copy()
    a_norm_rank_k_temp = jnp.multiply(a_norm_rank_k_temp, sigma_1_2[toscale])
    a_norm_rank_k_temp = jnp.add(a_norm_rank_k_temp, toadd[toscale])

    a_norm_rank_k_cor_sc = a_norm_rank_k_cor.copy()
    a_norm_rank_k_cor_sc[:, toscale] = a_norm_rank_k_temp
    a_norm_rank_k_cor_sc[a_norm_rank_k_cor == 0] = 0

    lt0 = a_norm_rank_k_cor_sc < 0
    a_norm_rank_k_cor_sc[lt0] = 0

    a_norm_size = a_norm.shape[0] * a_norm.shape[1]
    logger.info(
        f"{100*jnp.sum(lt0)/a_norm_size:.2f}% of the values became negative in the scaling process and were set to zero"
    )

    nonzero_mask = jnp.logical_and(originally_nonzero, (a_norm_rank_k_cor_sc == 0))
    a_norm_rank_k_cor_sc[nonzero_mask] = a_norm[nonzero_mask]

    original_nz = jnp.sum(a_norm > 0) / a_norm_size
    completed_nz = jnp.sum(a_norm_rank_k_cor_sc > 0) / a_norm_size
    logger.info(f"The matrix went from {100*original_nz:.2f}% nonzero to {100*completed_nz:.2f}% nonzero")

    return {
        "A_norm_rank_k": a_norm_rank_k,
        "A_norm_rank_k_cor": a_norm_rank_k_cor,
        "A_norm_rank_k_cor_sc": a_norm_rank_k_cor_sc,
    }
