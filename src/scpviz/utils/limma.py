"""Moderated (limma-style) two-group t-test used by ``de(method='limma')``.

Not star-imported by ``scpviz.utils``; ``pdata.de`` and ``de_adata`` call it internally.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import t as t_dist

from scpviz.utils.formatting import format_log_prefix


def _trigamma_inverse(x: float) -> float:
    """Solve ``trigamma(y) = x`` for ``y > 0``.

    Newton iteration on ``1 / trigamma(y)``, which is convex and nearly linear
    (Smyth 2004, Appendix). The asymptotes ``y ~ 1/sqrt(x)`` (large ``x``) and
    ``y ~ 1/x`` (small ``x``) cover the extremes.
    """
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = polygamma(1, y)
        dif = tri * (1.0 - tri / x) / polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)

def _fit_fdist_legacy(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Moment estimate of the prior ``(scale, df)`` for ``s2 ~ scale * F(df, d0)``.

    Method of moments on ``e = log(s2) + log(df/2) - digamma(df/2)`` (Smyth 2004):
    ``var(e) - mean(trigamma(df/2)) = trigamma(d0/2)``. Used when all positive
    residual df are equal. Returns ``d0 = inf`` and the mean variance when the
    excess dispersion is not positive.
    """
    ok = np.isfinite(s2) & np.isfinite(df) & (df > 1e-15) & (s2 > -1e-15)
    x, d = np.maximum(s2[ok], 0.0), df[ok]
    n = x.size
    if n == 0:
        raise ValueError(f"{format_log_prefix('error')} limma: no informative residual variances to estimate the prior.")
    if n == 1:
        return float(x[0]), 0.0
    m = np.median(x)
    x = np.maximum(x, 1e-5 * (m if m > 0 else 1.0))
    e = np.log(x) + np.log(d / 2) - digamma(d / 2)
    emean = e.mean()
    evar = ((e - emean) ** 2).sum() / (n - 1) - polygamma(1, d / 2).mean()
    if evar > 0:
        d0 = 2.0 * _trigamma_inverse(evar)
        return float(np.exp(emean - (np.log(d0 / 2) - digamma(d0 / 2)))), d0
    return float(x.mean()), np.inf

def _fit_fdist_unequal_df(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Profile-likelihood estimate of the prior ``(scale, df)`` when residual df differ.

    The scale is profiled out through the df-weighted mean of the adjusted log
    variances (weights ``1 / trigamma(df/2)``), then the scaled-F log-likelihood
    is maximised over the prior half-df ``b`` through ``q = b / (1 + b)`` on
    ``[0.5, 0.9998]`` (so ``2 <= d0 <= 9998``). Features with ``df < 0.01`` or a
    missing variance carry no weight.
    """
    x = np.where(np.isfinite(s2), s2, 0.0)
    w = np.isfinite(s2).astype(float)
    d = np.asarray(df, dtype=float).copy()
    small = d < 0.01
    w[small], d[small] = 0.0, 1.0
    pos = (x > 0) & (w > 0)
    if pos.sum() < 2:
        raise ValueError(f"{format_log_prefix('error')} limma: fewer than 2 informative residual variances; cannot estimate the prior.")
    xpos = np.maximum(x, 1e-12 * np.median(x[pos]))
    d1 = d / 2
    e = np.log(xpos) + np.log(d1) - digamma(d1)
    wt = w / polygamma(1, d1)
    emean = (wt * e).sum() / wt.sum()
    d1x = d1 * xpos

    def minus_twice_loglik(q: float) -> float:
        d2 = q / (1.0 - q)
        d2s20 = d2 * np.exp(emean - (np.log(d2) - digamma(d2)))
        return -2.0 * np.sum(w * (-(d1 + d2) * np.log1p(d1x / d2s20) - d1 * np.log(d2s20)
                                  + gammaln(d1 + d2) - gammaln(d2)))

    q = minimize_scalar(minus_twice_loglik, bounds=(0.5, 0.9998), method="bounded",
                        options={"xatol": 1.220703125e-4}).x
    d2 = q / (1.0 - q)
    return float(np.exp(emean - (np.log(d2) - digamma(d2)))), 2.0 * d2

def squeeze_variance(
    s2: np.ndarray,
    df: np.ndarray,
    legacy: bool | None = None,
) -> tuple[np.ndarray, float, float, str, np.ndarray]:
    """
    Empirical Bayes shrinkage of per-feature residual variances (limma ``squeezeVar``).

    Args:
        s2 (numpy.ndarray): Residual variances per feature; NaN where ``df == 0``.
        df (numpy.ndarray): Residual degrees of freedom per feature.
        legacy (bool, optional): Prior estimator. ``None`` (default) follows limma:
            moment estimator when all positive residual df are equal, otherwise the
            unequal-df profile-likelihood estimator.

    Returns:
        tuple: ``(s2_post, s2_prior, df_prior, estimator, df_total)``. ``estimator`` is
        ``"legacy"``, ``"unequal_df"`` or ``"none"`` (fewer than 3 features: variances unchanged).
        ``df_total = min(df + df_prior, sum(df))``.

    Note:
        Features with ``df == 0`` carry no weight in the prior fit but still get a
        posterior variance (the prior).
    """
    if df.size < 3:  # no benefit from moderation (matches limma)
        s2_post, s0, d0, estimator = s2.copy(), np.nan, 0.0, "none"
    else:
        var = np.where(df == 0, 0.0, s2)
        dfp = df[df > 0]
        if legacy is None:
            legacy = bool(dfp.min() == dfp.max())
        s0, d0 = (_fit_fdist_legacy if legacy else _fit_fdist_unequal_df)(var, df)
        estimator = "legacy" if legacy else "unequal_df"
        if np.isfinite(d0):
            s2_post = (df * var + d0 * s0) / (df + d0)
        else:
            s2_post = np.full(df.shape, s0)
    return s2_post, s0, d0, estimator, np.minimum(df + d0, df.sum())

def moderated_two_group_test(
    data1: np.ndarray,
    data2: np.ndarray,
    legacy: bool | None = None,
) -> dict[str, Any]:
    """
    Two-group moderated t-test (empirical Bayes variance shrinkage, limma-style).

    Fits ``y = intercept + beta * I(group1)`` per feature by ordinary least squares
    on the finite values only, shrinks the residual variances toward a common prior,
    and tests ``beta = mean(group1) - mean(group2)`` with the moderated t-statistic
    (Smyth 2004; Phipson et al. 2016). Validated numerically against R limma outputs
    (``tests/data/limma``).

    Args:
        data1 (numpy.ndarray): Log-scale abundances for group 1, shape
            ``(n_samples1, n_features)``. Missing values must be NaN.
        data2 (numpy.ndarray): Same for group 2.
        legacy (bool, optional): Prior estimator. ``None`` (default) follows limma:
            the moment estimator when all positive residual df are equal, otherwise
            the unequal-df profile-likelihood estimator. ``True`` / ``False`` force one.

    Returns:
        dict: Per-feature arrays ``beta``, ``stdev_unscaled``, ``s2``, ``df_residual``,
        ``s2_post``, ``t``, ``df_total``, ``p_value``; and scalars ``s2_prior``,
        ``df_prior``, ``estimator`` (``"legacy"``, ``"unequal_df"`` or ``"none"``).

    Raises:
        ValueError: If there are no residual degrees of freedom, or too little
            information to estimate the prior.

    Note:
        Differences from running R limma (``lmFit`` + ``eBayes``):

        - Scope is a single two-group comparison on log-scale data. Not implemented:
          ``trend`` / ``robust`` options, B-statistics (log-odds), arbitrary designs,
          contrasts and observation weights.
        - Missing values (NaN) are handled per protein, so residual df can differ. As in
          limma, the unequal-df prior estimator is then used by default. Proteins with
          one group entirely missing have ``beta = NaN`` but still contribute their
          variance to the prior and to the pooled df. With fewer than 3 proteins no
          moderation is possible (variances unchanged).
        - Tested against limma 3.66.0 outputs. Exactly-zero residual variances can shift
          the unequal-df prior by ~4e-4 (relative): R's QR fit leaves ~1e-30 noise there
          that counts as "positive" when the variance floor is set, whereas the sums of
          squares here are exactly 0.
        - Sparse proteins are not filtered (``min_valid_fraction`` is ignored); filter
          upstream so the prior is estimated from proteins you want to test.

        References: Smyth GK (2004), Stat Appl Genet Mol Biol 3:Article 3; Phipson B
        et al. (2016), Ann Appl Stat 10:946-963.

    Example:
        Moderated test between two groups of log2 abundances:
            ```python
            res = moderated_two_group_test(log2_group1, log2_group2)
            res["p_value"], res["df_prior"]
            ```
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n1, n2 = (~np.isnan(data1)).sum(axis=0), (~np.isnan(data2)).sum(axis=0)
        mean1, mean2 = np.nanmean(data1, axis=0), np.nanmean(data2, axis=0)
        sse = np.nansum((data1 - mean1) ** 2, axis=0) + np.nansum((data2 - mean2) ** 2, axis=0)
        both = (n1 > 0) & (n2 > 0)
        n = n1 + n2
        df = np.where(both, n - 2, np.maximum(n - 1, 0)).astype(float)
        s2 = np.where(df > 0, sse / np.where(df > 0, df, 1.0), np.nan)
        beta = np.where(both, mean1 - mean2, np.nan)
        sd_unscaled = np.where(both, np.sqrt(1.0 / np.maximum(n1, 1) + 1.0 / np.maximum(n2, 1)), np.nan)

    if df.max() == 0 or not np.isfinite(s2).any():
        raise ValueError(
            f"{format_log_prefix('error')} limma: no residual degrees of freedom "
            "(need at least 2 values in a group for some feature)."
        )

    s2_post, s0, d0, estimator, df_total = squeeze_variance(s2, df, legacy=legacy)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / sd_unscaled / np.sqrt(s2_post)
    p_value = 2.0 * t_dist.sf(np.abs(t), df_total)
    return {"beta": beta, "stdev_unscaled": sd_unscaled, "s2": s2, "df_residual": df,
            "s2_post": s2_post, "t": t, "df_total": df_total, "p_value": p_value,
            "s2_prior": s0, "df_prior": d0, "estimator": estimator}
