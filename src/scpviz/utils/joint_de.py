"""Joint group-means linear model DE behind ``pdata.joint_de``.

One ordinary-least-squares fit of ``y ~ 0 + group`` per protein across **all** groups,
then contrasts (pairs or weight vectors) tested with the pooled residual variance
(``method='lm'``) or the limma empirical-Bayes moderated variance (``method='limma'``).

The one-hot design makes ``X'X`` diagonal, so everything is closed form: group means,
group counts, residual variance and contrast variance ``sum(c_k**2 / n_k)``. Not star-imported
by ``scpviz.utils``; ``pdata.joint_de`` calls it internally.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from scpviz.utils.data import get_adata_layer, infer_layer_is_log
from scpviz.utils.de_reporting import (
    format_de_comparison_label,
    print_de_result_summary,
    print_de_run_header,
    print_limma_summary,
)
from scpviz.utils.formatting import format_log_prefix
from scpviz.utils.limma import squeeze_variance
from scpviz.utils.mixed_de import resolve_obs_meta
from scpviz.utils.stats import bh_adjust_pvalues, get_log_info, mask_zeros_as_nan

_SIGNIFICANCE_CATEGORIES = ["upregulated", "downregulated", "not significant", "not comparable"]


def fit_group_means(data: np.ndarray, codes: np.ndarray, n_groups: int) -> dict[str, np.ndarray]:
    """
    Fit ``y ~ 0 + group`` for every protein by closed-form group means.

    Args:
        data (numpy.ndarray): Log-scale abundances, shape ``(n_samples, n_proteins)``;
            non-finite values are missing.
        codes (numpy.ndarray): Integer group code per sample, ``0 .. n_groups - 1``.
        n_groups (int): Number of groups ``K``.

    Returns:
        dict: ``mean`` and ``n`` (shape ``(n_proteins, K)``; ``mean`` is NaN where ``n == 0``),
        ``df`` (``n_observed - n_occupied_groups``) and ``s2`` (residual variance, NaN where
        ``df <= 0``), both shape ``(n_proteins,)``.
    """
    observed = np.isfinite(data)
    onehot = (codes[:, None] == np.arange(n_groups)[None, :]).astype(float)
    n = observed.T.astype(float) @ onehot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(n > 0, np.where(observed, data, 0.0).T @ onehot / np.where(n > 0, n, 1.0), np.nan)
        sse = np.zeros(data.shape[1])
        for k in range(n_groups):
            sse += np.nansum((data[codes == k] - mean[:, k]) ** 2, axis=0)
    df = n.sum(axis=1) - (n > 0).sum(axis=1)
    s2 = np.where(df > 0, sse / np.where(df > 0, df, 1.0), np.nan)
    return {"mean": mean, "n": n, "df": df, "s2": s2}


def fit_contrasts(fit: dict[str, np.ndarray], weights: np.ndarray, method: str = "limma") -> dict[str, Any]:
    """
    Test contrasts of group means from :func:`fit_group_means`.

    Args:
        fit (dict): Output of :func:`fit_group_means`.
        weights (numpy.ndarray): Contrast weights, shape ``(K, n_contrasts)``.
        method (str): ``"lm"`` (pooled residual variance) or ``"limma"`` (one empirical Bayes
            prior fitted on all proteins with residual df, then moderated variances).

    Returns:
        dict: Arrays of shape ``(n_proteins, n_contrasts)``: ``effect``, ``se``, ``t``, ``p_value``
        (NaN where a group with non-zero weight is unobserved). Arrays of shape ``(n_proteins,)``:
        ``df_residual``, ``s2``, ``s2_post``, ``df_total`` (equal to ``s2`` and ``df_residual``
        for ``lm``). Scalars ``s2_prior``, ``df_prior`` and ``estimator`` (``"lm"`` for ``lm``).

    Raises:
        ValueError: If ``method`` is unknown or no protein has residual degrees of freedom.
    """
    if method not in ("lm", "limma"):
        raise ValueError(f"{format_log_prefix('error')} Unsupported method '{method}'. Choose 'lm' or 'limma'.")
    n, df, s2 = fit["n"], fit["df"], fit["s2"]
    if df.max() <= 0 or not np.isfinite(s2).any():
        raise ValueError(
            f"{format_log_prefix('error')} joint_de: no residual degrees of freedom "
            "(need at least 2 samples in some group for some protein)."
        )
    estimable = ((n == 0).astype(float) @ (weights != 0).astype(float)) == 0
    effect = np.where(estimable, np.nan_to_num(fit["mean"]) @ weights, np.nan)
    v = np.where(estimable, (1.0 / np.maximum(n, 1.0)) @ weights**2, np.nan)

    if method == "limma":
        s2_used, s0, d0, estimator, df_used = squeeze_variance(s2, df)
    else:
        s2_used, s0, d0, estimator, df_used = s2, np.nan, 0.0, "lm", df
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(s2_used[:, None] * v)
        t = effect / se
    p_value = 2.0 * t_dist.sf(np.abs(t), df_used[:, None])
    return {"effect": effect, "se": se, "t": t, "p_value": p_value, "df_residual": df, "s2": s2,
            "s2_post": s2_used, "df_total": df_used, "s2_prior": s0, "df_prior": d0, "estimator": estimator}


def build_contrasts(
    levels: list[str],
    level_dicts: dict[str, dict[str, str]],
    reference: str | None,
    comparisons: str | list[str] | None,
    contrasts: dict[str, Any] | None,
) -> tuple[np.ndarray, list[str], list[dict[str, float]]]:
    """
    Turn ``reference``/``comparisons`` or ``contrasts`` into a weight matrix.

    Args:
        levels (list of str): Group labels in fit order.
        level_dicts (dict): ``{level: {column: value}}``, used for ``vs`` labels.
        reference (str, optional): Shared control group.
        comparisons (str or list of str, optional): Groups tested against ``reference``.
        contrasts (dict, optional): ``{name: (test, ref)}`` or ``{name: {group: weight}}``.

    Returns:
        tuple: ``(weights, labels, specs)`` with ``weights`` of shape ``(K, n_contrasts)`` and
        ``specs`` the ``{group: weight}`` dict of each contrast. ``reference``/``comparisons`` labels
        are ``"[{col: test}] vs [{col: ref}]"``; ``contrasts`` labels are the dict keys.

    Raises:
        ValueError: On missing/conflicting arguments, unknown groups, non-finite or zero weights,
            or weights that do not sum to zero.
    """
    if contrasts is not None and (reference is not None or comparisons is not None):
        raise ValueError(f"{format_log_prefix('error')} Pass either `contrasts` or `reference` + `comparisons`, not both.")
    if contrasts is None:
        if reference is None or comparisons is None:
            raise ValueError(f"{format_log_prefix('error')} Provide `reference` + `comparisons`, or `contrasts`.")
        comps = [comparisons] if isinstance(comparisons, str) else list(comparisons)
        unknown = [g for g in [reference, *comps] if g not in level_dicts]
        if unknown:
            raise ValueError(f"{format_log_prefix('error')} Unknown group(s) {unknown}. Groups: {levels}.")
        contrasts = {format_de_comparison_label(level_dicts[c], level_dicts[reference]): (c, reference) for c in comps}
    if not contrasts:
        raise ValueError(f"{format_log_prefix('error')} No contrasts requested.")

    labels, specs = [], []
    weights = np.zeros((len(levels), len(contrasts)))
    for j, (name, spec) in enumerate(contrasts.items()):
        if isinstance(spec, dict):
            spec = {str(k): float(w) for k, w in spec.items()}
        else:
            if len(spec) != 2:
                raise ValueError(f"{format_log_prefix('error')} Contrast {name!r} must be (test, ref) or a {{group: weight}} dict.")
            if str(spec[0]) == str(spec[1]):
                raise ValueError(f"{format_log_prefix('error')} Contrast {name!r} compares a group with itself.")
            spec = {str(spec[0]): 1.0, str(spec[1]): -1.0}
        unknown = [g for g in spec if g not in levels]
        if unknown:
            raise ValueError(f"{format_log_prefix('error')} Contrast {name!r}: unknown group(s) {unknown}. Groups: {levels}.")
        w = np.array(list(spec.values()))
        if not np.isfinite(w).all() or (w == 0).any():
            raise ValueError(f"{format_log_prefix('error')} Contrast {name!r}: weights must be finite and non-zero (omit unused groups).")
        if abs(w.sum()) > 1e-9:
            raise ValueError(f"{format_log_prefix('error')} Contrast {name!r}: weights must sum to 0 (got {w.sum():g}).")
        for g, wt in spec.items():
            weights[levels.index(g), j] = wt
        labels.append(str(name))
        specs.append(spec)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{format_log_prefix('error')} Contrast labels must be unique: {labels}.")
    return weights, labels, specs


def _contrast_table(
    res: dict[str, Any],
    j: int,
    label: str,
    spec: dict[str, float],
    levels: list[str],
    n: np.ndarray,
    var: pd.DataFrame,
    names: pd.Index,
    scale: float,
) -> pd.DataFrame:
    """Per-contrast table with ``de()`` / ``mixed_de()`` column conventions (no FDR yet)."""
    df = pd.DataFrame(index=names)
    df["Genes"] = var["Genes"] if "Genes" in var.columns else var.index
    df["log2fc"] = res["effect"][:, j] / scale
    df["se"] = res["se"][:, j] / scale
    df["p_value"] = res["p_value"][:, j]
    df["test_statistic"] = res["t"][:, j]
    for g in spec:
        df[f"n_{g}"] = n[:, levels.index(g)].astype(int)
    df["contrast"] = label
    df["-log10(p_value)"] = -np.log10(df["p_value"].replace(0, np.nan).astype(float))
    df["significance_score"] = df["-log10(p_value)"] * df["log2fc"]
    return df


def _label_significance(df: pd.DataFrame, p_col: str, threshold: float, log2fc: float) -> pd.DataFrame:
    """Add the ``significance`` categorical and sort by it (stable)."""
    sig = pd.Series("not significant", index=df.index)
    sig[df["log2fc"].isna() | df["p_value"].isna()] = "not comparable"
    sig[(df[p_col] < threshold) & (df["log2fc"] > log2fc)] = "upregulated"
    sig[(df[p_col] < threshold) & (df["log2fc"] < -log2fc)] = "downregulated"
    df["significance"] = pd.Categorical(sig, categories=_SIGNIFICANCE_CATEGORIES, ordered=True)
    return df.sort_values(by="significance", kind="stable")


def run_joint_de(
    adata: Any,
    summary: pd.DataFrame | None,
    *,
    groupby: str | list[str],
    reference: str | None,
    comparisons: str | list[str] | None,
    contrasts: dict[str, Any] | None,
    method: str,
    layer: str,
    correct_fdr: bool,
    fdr_scope: str,
    threshold: float,
    log2fc: float,
) -> dict[str, Any]:
    """
    Run the joint model on protein ``adata`` and assemble the result collection.

    Returns:
        dict: ``{"contrasts": {label: DataFrame}, "meta": dict, "diagnostics": DataFrame}``.
    """
    if method not in ("lm", "limma"):
        raise ValueError(f"{format_log_prefix('error')} Unsupported method '{method}'. Choose 'lm' or 'limma'.")
    if fdr_scope not in ("per_contrast", "global", "both"):
        raise ValueError(f"{format_log_prefix('error')} fdr_scope must be 'per_contrast', 'global' or 'both'.")

    cols = [groupby] if isinstance(groupby, str) else list(groupby)
    meta = resolve_obs_meta(adata, summary, cols)[cols]
    valid = meta.notna().all(axis=1) & (meta.astype(str) != "").all(axis=1)
    n_dropped = int((~valid).sum())
    meta = meta[valid].astype(str)
    if meta.empty:
        raise ValueError(f"{format_log_prefix('error')} No samples have a value for {cols}.")
    group_labels = meta.apply("_".join, axis=1)
    levels = sorted(group_labels.unique())
    level_dicts = {lv: meta.loc[group_labels == lv, cols].iloc[0].to_dict() for lv in levels}
    if len(levels) < 2:
        raise ValueError(f"{format_log_prefix('error')} joint_de needs at least 2 groups; found {levels}.")
    weights, labels, specs = build_contrasts(levels, level_dicts, reference, comparisons, contrasts)

    layer_is_log = infer_layer_is_log(layer, adata)
    if not layer_is_log:
        raise ValueError(f"{format_log_prefix('error')} joint_de needs a log-transformed layer; run log_transform() first.")
    base_label, log_base, pseudocount = get_log_info(adata, layer, layer_is_log)
    rows = adata.obs_names.get_indexer(meta.index)
    data = np.asarray(get_adata_layer(adata, layer), dtype=float)[rows]
    n_before = int(np.isnan(data).sum())
    data = mask_zeros_as_nan(data, layer_is_log, log_base, pseudocount)
    n_zero = int(np.isnan(data).sum()) - n_before
    scale = float(np.log(2) / np.log(log_base))  # divide log_base differences to get log2

    codes = np.array([levels.index(g) for g in group_labels])
    fit = fit_group_means(data, codes, len(levels))
    res = fit_contrasts(fit, weights, method)
    n_samples = {lv: int((codes == k).sum()) for k, lv in enumerate(levels)}

    print_de_run_header(
        assay="protein",
        comparing=", ".join(labels),
        group_sizes=", ".join(f"{lv}={c}" for lv, c in n_samples.items()),
        layer_line=f"{layer} (log-transformed)",
        method_line=f"joint {method} | groupby={cols} | {len(labels)} contrast(s)",
        correct_fdr=correct_fdr,
        threshold=threshold,
        log2fc_thresh=log2fc,
    )
    if n_dropped:
        print(f"{format_log_prefix('info', 2)} {n_dropped} sample(s) without a value for {cols} were left out of the fit.")
    if n_zero:
        print(f"{format_log_prefix('info', 2)} {n_zero} zero value(s) in the selected samples treated as missing (NaN).")
    if method == "limma":
        print_limma_summary({"s2": fit["s2"], "estimator": res["estimator"],
                             "df_prior": res["df_prior"], "s2_prior": res["s2_prior"]})

    names, var = adata.var_names, adata.var
    tables = {
        lb: _contrast_table(res, j, lb, sp, levels, fit["n"], var, names, scale)
        for j, (lb, sp) in enumerate(zip(labels, specs))
    }
    if correct_fdr:
        pooled = np.concatenate([t["p_value"].to_numpy() for t in tables.values()])
        pooled_adj = np.split(bh_adjust_pvalues(pooled), len(tables)) if fdr_scope != "per_contrast" else None
        for k, t in enumerate(tables.values()):
            per = bh_adjust_pvalues(t["p_value"].to_numpy())
            if fdr_scope == "global":
                t["adj_p_value"] = pooled_adj[k]
            else:
                t["adj_p_value"] = per
                if fdr_scope == "both":
                    t["adj_p_value_global"] = pooled_adj[k]
            t["-log10(adj_p_value)"] = -np.log10(t["adj_p_value"].replace(0, np.nan).astype(float))
    p_col = "adj_p_value" if correct_fdr else "p_value"
    run_meta = {
        "method": method, "layer": layer, "groupby": cols, "levels": levels, "contrasts": dict(zip(labels, specs)),
        "fdr_scope": fdr_scope, "n_samples": n_samples, "n_samples_dropped": n_dropped,
        "estimator": res["estimator"], "s2_prior": res["s2_prior"], "df_prior": res["df_prior"],
        "log_base": base_label,
    }
    for lb, t in list(tables.items()):
        t = _label_significance(t, p_col, threshold, log2fc)
        t.attrs["joint_de"] = {**run_meta, "contrast_label": lb}
        if method == "limma":
            t.attrs["limma"] = {k: res[k] for k in ("s2_prior", "df_prior", "estimator")}
        tables[lb] = t

    diagnostics = pd.DataFrame({f"n_{lv}": fit["n"][:, k].astype(int) for k, lv in enumerate(levels)}, index=names)
    diagnostics["df_residual"] = res["df_residual"]
    diagnostics["s2"] = res["s2"]
    if method == "limma":
        diagnostics["s2_post"] = res["s2_post"]
        diagnostics["df_total"] = res["df_total"]
    return {"contrasts": tables, "meta": run_meta, "diagnostics": diagnostics}


def print_joint_de_summary(collection: dict[str, Any], *, stats_location: str, correct_fdr: bool) -> None:
    """Print one result block per contrast and the storage location once."""
    for label, table in collection["contrasts"].items():
        print_de_result_summary(
            table, stats_location=stats_location, correct_fdr=correct_fdr,
            contrast_label=label, include_storage_guide=False,
        )
    print(f"{format_log_prefix('result_only', indent=2)} Results stored in: {stats_location}")
    print(f"       • Collection: ['contrasts'][<label>] tables (log2fc, p_value, adj_p_value, significance, etc.), 'meta', 'diagnostics'")


def build_stats_key(groupby: str | list[str], reference: str | None, contrasts: dict[str, Any] | None, method: str) -> str:
    """``pdata.stats`` key for a joint_de run."""
    by = groupby if isinstance(groupby, str) else "+".join(groupby)
    what = f"ref={reference}" if contrasts is None else f"contrasts={list(contrasts)}"
    return f"joint: groupby={by} | {what} | method={method}"
