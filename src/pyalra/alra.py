import numpy as np
import numpy.typing as npt
import scipy as sp
from loguru import logger
from sklearn.utils.extmath import randomized_svd

from pyalra.choose_k import choose_k
from pyalra.logging import init_logger


def alra(
    a_norm: npt.ArrayLike | sp.sparse.spmatrix,
    k: int = 0,
    q: int = 10,
    quantile_prob: float = 0.001,
    seed: float | None = None,
    debug: bool = False,
    *,
    save_log: bool = False,
    log_level: int = 1,
) -> dict[str, np.ndarray]:
    """Computes the k-rank approximation to A_norm and adjusts it according to the error distribution learned from
    the negative values.

    Parameters
    ----------
    A_norm : ArrayLike
        The log-transformed expression matrix of cells (rows) vs. genes (columns)
    k : int, default=0
        the rank of the rank-k approximation. Set to 0 for automated choice of k.
    q : int, default=10
        the number of additional power iterations in randomized SVD
    quantile_prob : float, default=0.001
        Quantile to calculate for each gene
    seed : float, optional
        Seed used to initiate the randomized SVD

    Returns
    -------
    A dict with the following items:
        A_norm_rank_k : ArrayLike
            The rank k approximation of A_norm.
        A_norm_rank_k_cor : ArrayLike
            The rank k approximation of A_norm, adaptively thresholded
        A_norm_rank_k_cor_sc : ArrayLike
            The rank k approximation of A_norm, adaptively thresholded and with the first two moments of the non-zero
            values matched to the first two moments of the non-zeros of A_norm. This is the completed
            matrix most people will want to work with
    """
    if debug:
        init_logger(log_level, save_log)

    logger.info(f"Read matrix with {a_norm.shape[0]} cells and {a_norm.shape[1]} genes")

    if k == 0:
        k, _, u, d, v = choose_k(a_norm)
        logger.info(f"Chose k = {k}")
    else:
        logger.debug("Running rsvd")
        u, d, v = randomized_svd(M=a_norm, n_components=k, n_iter=q, random_state=seed)

    logger.info("Getting nonzeros\n")
    originally_nonzero = a_norm > 0

    a_norm_rank_k = u @ np.diag(d) @ v

    logger.info(f"Find the {quantile_prob} quantile for each gene")
    # should this be `np.nanquantile`?
    a_norm_rank_k_mins = np.absolute(np.quantile(a_norm_rank_k, axis=0, q=quantile_prob))

    logger.info("Sweep")
    a_norm_rank_k_cor = np.zeros(shape=a_norm_rank_k.shape)
    np.copyto(
        dst=a_norm_rank_k_cor,
        src=a_norm_rank_k,
        where=(a_norm_rank_k > np.tile(a_norm_rank_k_mins, (len(a_norm_rank_k), 1))),
    )

    sigma_1 = np.nanstd(a_norm_rank_k_cor, axis=0)
    if sp.sparse.issparse(a_norm):
        sigma_2 = np.nanstd(a_norm.toarray(), axis=0)
    else:
        sigma_2 = np.nanstd(a_norm, axis=0)

    mu_1 = np.divide(a_norm_rank_k_cor.sum(axis=0), a_norm_rank_k_cor.sum(axis=0))
    mu_2 = np.divide(a_norm.sum(axis=0), (a_norm > 0).sum(axis=0))

    toscale = np.logical_and(
        np.logical_and(~np.isnan(sigma_1), ~np.isnan(sigma_2)),
        np.logical_and(~np.logical_and((sigma_1 == 0), (sigma_2 == 0)), ~(sigma_1 == 0)),
    )

    logger.info(f"Scaling all except for {sum(~toscale)} columns")

    sigma_1_2 = np.divide(sigma_2, sigma_1)

    toadd = np.ravel(np.add(-np.divide(np.multiply(mu_1, sigma_2), sigma_1), mu_2))

    a_norm_rank_k_temp = a_norm_rank_k_cor[:, toscale].copy()
    a_norm_rank_k_temp = np.multiply(a_norm_rank_k_temp, sigma_1_2[toscale])
    a_norm_rank_k_temp = np.add(a_norm_rank_k_temp, toadd[toscale])

    a_norm_rank_k_cor_sc = a_norm_rank_k_cor.copy()
    np.copyto(dst=a_norm_rank_k_cor_sc[:, toscale], src=a_norm_rank_k_temp)
    a_norm_rank_k_cor_sc_zeros = np.zeros_like(a_norm_rank_k_cor_sc)
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm_rank_k_cor_sc_zeros, where=(a_norm_rank_k_cor == 0))

    lt0 = a_norm_rank_k_cor_sc < 0
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm_rank_k_cor_sc_zeros, where=lt0)

    a_norm_size = a_norm.shape[0] * a_norm.shape[1]
    logger.info(
        f"{100 * np.sum(lt0) / a_norm_size:.2f}% of the values became negative in the scaling process and were set to zero"
    )

    if sp.sparse.issparse(originally_nonzero):
        nonzero_mask = np.logical_and(originally_nonzero.toarray(), (a_norm_rank_k_cor_sc == 0))
    else:
        nonzero_mask = np.logical_and(originally_nonzero, (a_norm_rank_k_cor_sc == 0))
    np.copyto(dst=a_norm_rank_k_cor_sc, src=a_norm, where=nonzero_mask)

    original_nz = np.sum(a_norm > 0) / a_norm_size
    completed_nz = np.sum(a_norm_rank_k_cor_sc > 0) / a_norm_size
    logger.info(f"The matrix went from {100 * original_nz:.2f}% nonzero to {100 * completed_nz:.2f}% nonzero")

    return {
        "A_norm_rank_k": a_norm_rank_k,
        "A_norm_rank_k_cor": a_norm_rank_k_cor,
        "A_norm_rank_k_cor_sc": a_norm_rank_k_cor_sc,
    }
