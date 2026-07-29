"""Mixed-model and pseudobulk differential expression helpers."""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, ttest_rel

from scpviz.utils.data import get_adata_layer, infer_layer_is_log, resolve_input_layer
from scpviz.utils.de_reporting import (
    format_de_comparison_label,
    format_de_column_hint,
    print_de_result_summary,
    print_de_threshold_line,
)
from scpviz.utils.formatting import format_log_prefix
from scpviz.utils.stats import bh_adjust_pvalues

ContrastMode = Literal["specified", "pairwise", "one_vs_rest"]
ObservationLevel = Literal["auto", "cells", "pseudobulk", "subsample"]
RandomEffects = Literal["intercept", "intercept_slope"]
FdrScope = Literal["per_contrast", "global", "both"]
DeMethod = Literal["mixedlm", "pseudobulk", "auto"]

MIN_DONORS_PAIRED_PSEUDOBULK = 3
MIN_DONORS_MIXEDLM_WARN = 3

DE_METHOD_LABELS: dict[str, str] = {
    "mixedlm": "mixed model",
    "pseudobulk_paired": "pseudobulk (paired)",
    "pseudobulk_unpaired": "pseudobulk (unpaired)",
    "pseudobulk": "failed",
}

DE_FAILURE_LABELS: dict[str, str] = {
    "insufficient_donors": "fewer than 2 donors",
    "constant_expression": "constant expression",
    "not_converged": "mixed model did not converge",
    "non_finite_pvalue": "mixed model returned invalid p-value",
    "fit_error": "mixed model fit error",
    "missing_contrast_coefficient": "contrast coefficient missing (check reference level)",
    "no_observations": "no non-missing observations",
    "missing_contrast_levels": "contrast level missing",
    "insufficient_donors_per_arm": "fewer than 2 donors per group",
}

def validate_mixed_de_paths(
    *,
    group_col: str | None,
    formula: str | None,
    contrast_term: str | None,
    fixed_covariates: list[str] | None,
    values: list[dict[str, Any]] | None,
) -> Literal["simple", "advanced"]:
    """Enforce simple vs advanced mutual exclusion."""
    simple_set = group_col is not None or values is not None
    advanced_set = formula is not None or contrast_term is not None

    if simple_set and advanced_set:
        raise ValueError(
            "Use either the simple path (`group_col` + `contrast`) or the advanced path "
            "(`formula` + `contrast_term`), not both."
        )
    if formula is not None and fixed_covariates:
        raise ValueError(
            "`fixed_covariates` cannot be used with `formula`. "
            "Include covariates as terms in `formula` instead."
        )
    if values is not None and formula is not None:
        raise ValueError("`values` cannot be used with `formula`.")

    if formula is not None:
        if contrast_term is None:
            raise ValueError("`contrast_term` is required when `formula` is provided.")
        return "advanced"
    if group_col is not None or values is not None:
        return "simple"
    raise ValueError(
        "Provide either `group_col` + `contrast` (simple path) or "
        "`formula` + `contrast_term` + `contrast` (advanced path)."
    )

def resolve_values_sugar(
    values: list[dict[str, Any]],
    *,
    group_col: str | None,
    contrast: tuple[str, str] | None,
    subset: dict[str, Any] | None,
) -> tuple[str, tuple[str, str], dict[str, Any] | None]:
    """Infer group_col, contrast, and subset extensions from two-group values dicts.

    When ``contrast`` is omitted, ordering matches :meth:`de` ``values`` sugar:
    label ``{group1} vs {group2}`` and log2fc = group1 - group2 on log scale.
    """
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("`values` must be a list of exactly two group dictionaries.")
    if not all(isinstance(v, dict) for v in values):
        raise ValueError("Each entry in `values` must be a dictionary.")

    v0, v1 = values
    if v0 == v1:
        raise ValueError("Both groups in `values` refer to the same condition.")

    all_keys = set(v0) | set(v1)
    differing = {k for k in all_keys if v0.get(k) != v1.get(k)}
    if not differing:
        raise ValueError("`values` entries must differ on at least one key.")

    if group_col is None:
        if len(differing) != 1:
            raise ValueError(
                "When `group_col` is not provided, `values` may differ on exactly one key "
                f"(the contrast factor). Found differing keys: {sorted(differing)}."
            )
        group_col = next(iter(differing))

    if group_col not in differing:
        raise ValueError(
            f"`group_col`={group_col!r} is not one of the keys that differ between `values`."
        )

    if contrast is None:
        # Match de() values order: [group1, group2] → contrast=(test, ref)=(g1, g2).
        contrast = (str(v0[group_col]), str(v1[group_col]))
    else:
        expected = (str(v0[group_col]), str(v1[group_col]))
        got = (str(contrast[0]), str(contrast[1]))
        if got != expected:
            print(
                f"{format_log_prefix('warn')} `values` imply contrast={expected!r} "
                f"(log2fc = {v0[group_col]!r} − {v1[group_col]!r}), but explicit "
                f"`contrast`={got!r} was also passed. Using the explicit `contrast`; "
                "label/sign may not match `values` order. Prefer one or the other."
            )

    merged_subset = dict(subset or {})
    shared_keys = all_keys - {group_col}
    for key in shared_keys:
        if key in merged_subset:
            continue
        val0, val1 = v0.get(key), v1.get(key)
        if val0 == val1 and val0 is not None:
            merged_subset[key] = val0
    return group_col, contrast, merged_subset or None


def formula_has_interaction_on_term(formula: str, contrast_term: str) -> bool:
    """Return True if formula contains * involving contrast_term."""
    rhs = formula.split("~", 1)[-1].strip()
    if "*" not in rhs:
        return False
    parts = re.split(r"\s*\*\s*", rhs)
    return any(contrast_term in p.replace("expr", "").strip() for p in parts if contrast_term in p)

def apply_reference_levels(meta: pd.DataFrame, reference_levels: dict[str, str] | None) -> pd.DataFrame:
    if not reference_levels:
        return meta
    out = meta.copy()
    for col, ref in reference_levels.items():
        if col not in out.columns:
            continue
        cat = out[col].astype("string").astype("category")
        levels = [str(lv) for lv in cat.cat.categories]
        ref_s = str(ref)
        if ref_s in levels:
            cat = cat.cat.reorder_categories([ref_s] + [lv for lv in levels if lv != ref_s])
        out[col] = cat
    return out


def merge_contrast_reference_levels(
    reference_levels: dict[str, str] | None,
    *,
    contrast_term: str,
    contrast: tuple[str, str] | None,
    contrast_mode: ContrastMode,
) -> dict[str, str] | None:
    """Ensure the contrast factor uses ``contrast`` ref as the model baseline.

    Patsy/statsmodels default to alphabetical category order. When the requested
    test level sorts first (e.g. Cortex vs SNpc), the test coefficient is
    absent and mixed-model extraction fails. Auto-set the baseline to ``ref``
    (second element of ``contrast=(test, ref)``) unless the user already
    specified ``reference_levels[contrast_term]``.
    """
    out = dict(reference_levels or {})
    if contrast_mode == "specified" and contrast is not None and contrast_term not in out:
        out[contrast_term] = str(contrast[1])  # ref
    return out or None


def cast_formula_categoricals(meta: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Cast bare formula identifiers that exist in meta to category dtype."""
    rhs = formula.split("~", 1)[-1]
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", rhs))
    skip = {"expr", "I", "Q", "C", "Treatment", "Sum", "Standard", "Poly", "Bs"}
    out = meta.copy()
    for tok in tokens - skip:
        if tok in out.columns and not pd.api.types.is_numeric_dtype(out[tok]):
            out[tok] = out[tok].astype("string").astype("category")
    return out

def resolve_obs_meta(adata, summary: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    """Merge requested columns from adata.obs and ``.summary``.

    When ``.summary`` shares ``adata.obs`` row index (the usual case after
    ``_merge_obs``), columns are aligned by index. The legacy ``Sample``-column
    join is only used when indices differ.
    """
    meta = adata.obs.copy()
    if summary is not None and len(summary):
        summ = summary
        if summ.index.equals(meta.index):
            for col in columns:
                if col in summ.columns:
                    meta[col] = summ[col]
        elif "Sample" in summ.columns:
            summ_by_sample = summ.set_index("Sample", drop=False)
            if "Sample" in meta.columns:
                sample_keys = meta["Sample"].astype(str)
                for col in columns:
                    if col in summ_by_sample.columns:
                        meta[col] = sample_keys.map(
                            lambda s, c=col: (
                                summ_by_sample.loc[s, c] if s in summ_by_sample.index else np.nan
                            )
                        )
            else:
                for col in columns:
                    if col in summ_by_sample.columns:
                        meta[col] = meta.index.map(
                            lambda x, c=col: (
                                summ_by_sample.loc[x, c] if x in summ_by_sample.index else np.nan
                            )
                        )
        else:
            for col in columns:
                if col in summ.columns and len(summ) == len(meta):
                    meta[col] = summ[col].values
    missing = [c for c in columns if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata column(s) not found in .obs or .summary: {missing}")
    return meta

def expr_looks_non_log(
    expr: np.ndarray,
    *,
    median_threshold: float = 100.0,
    max_threshold: float = 1000.0,
) -> bool:
    """Heuristic: expression on a linear intensity / count scale (not log2)."""
    finite = expr[np.isfinite(expr)]
    if finite.size == 0:
        return False
    return float(np.median(finite)) > median_threshold or float(np.max(finite)) > max_threshold


def _layer_log_base_label(adata: Any, layer: str) -> str:
    resolved = resolve_input_layer(adata, layer)
    rec = adata.uns.get("layer_provenance", {}).get(resolved, {})
    base = str(rec.get("base", "2"))
    if base == "10":
        return "log10"
    if base == "e":
        return "loge"
    return "log2"


def prepare_expr_for_mixed_de(
    adata: Any,
    layer: str,
    *,
    auto_log2: bool = True,
    log_pseudocount: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load expression for mixed DE on log2 scale.

  Goralski-style workflow: normalize on linear scale, then ``log2`` before LMM /
    pseudobulk. Coefficients and mean differences are reported as log2 fold change.

    If ``layer`` is not registered as log-transformed and values look linear-scale,
    applies ``log2(x + pseudocount)`` in memory when ``auto_log2=True``.
    """
    layer_is_log = infer_layer_is_log(layer, adata)
    resolved_layer = resolve_input_layer(adata, layer)
    expr = np.asarray(get_adata_layer(adata, layer), dtype=float)
    meta: dict[str, Any] = {
        "input_layer": layer,
        "resolved_layer": resolved_layer,
        "layer_is_log": layer_is_log,
        "expr_auto_log2": False,
        "log_pseudocount": None,
        "log_base": _layer_log_base_label(adata, layer) if layer_is_log else None,
    }

    if layer_is_log:
        return expr, meta

    looks_non_log = expr_looks_non_log(expr)
    if not looks_non_log:
        finite = expr[np.isfinite(expr)]
        med = float(np.median(finite)) if finite.size else float("nan")
        resolved_note = (
            f" (resolved {layer!r} → {resolved_layer!r})"
            if resolved_layer != layer
            else ""
        )
        print(
            f"{format_log_prefix('info')} Layer {layer!r}{resolved_note} is not "
            f"registered as log-transformed but values look log-like "
            f"(median={med:.2g}). Proceeding. For provenance tracking, run "
            f"`pdata.log_transform(..., base=2)` (sets "
            f"``.uns['current_X_layer']`` when ``set_X=True``) or pass "
            f"`layer={resolved_layer!r}` / ``layer='X_log2'``."
        )
        return expr, meta

    finite = expr[np.isfinite(expr)]
    med = float(np.median(finite)) if finite.size else float("nan")
    if not auto_log2:
        raise ValueError(
            f"Layer {layer!r} does not appear to be log2-transformed (median={med:.2g}). "
            "mixed_de() fits on log2 scale and reports log2 fold changes. "
            "Normalize on linear scale, log2-transform, then pass that layer — e.g.\n\n"
            f"    pdata.log_transform(layer={layer!r}, base=2)\n"
            f"    pdata.mixed_de(..., layer='X_log2')\n\n"
            "Or set auto_log2=True (default) to apply log2(x + pseudocount) in memory "
            "for this run only."
        )

    n_negative = int(np.sum(finite < 0)) if finite.size else 0
    if n_negative > 0:
        print(
            f"{format_log_prefix('warn')} {n_negative} value(s) < 0 in layer {layer!r}. "
            f"log2(x + {log_pseudocount}) will be applied."
        )
    print(
        f"{format_log_prefix('warn')} Layer {layer!r} appears non-log (median={med:.2g}). "
        f"Applying log2(x + {log_pseudocount}) in memory for this mixed_de run only. "
        "To persist: "
        f"`pdata.log_transform(layer={layer!r}, base=2)` then `layer='X_log2'`."
    )
    expr = np.log2(expr + log_pseudocount)
    meta["expr_auto_log2"] = True
    meta["log_pseudocount"] = log_pseudocount
    meta["log_base"] = "log2"
    return expr, meta


def format_expr_layer_summary(expr_meta: dict[str, Any]) -> str:
    """Short layer description for mixed_de run logs."""
    layer = expr_meta["input_layer"]
    resolved = expr_meta.get("resolved_layer", layer)
    layer_disp = (
        f"{layer!r}→{resolved!r}" if resolved != layer else f"{layer!r}"
    )
    if expr_meta.get("expr_auto_log2"):
        pc = expr_meta.get("log_pseudocount", 1.0)
        return f"{layer_disp} (auto log2, pseudocount={pc})"
    if expr_meta.get("layer_is_log"):
        base = expr_meta.get("log_base") or "log2"
        return f"{layer_disp} ({base}-transformed)"
    return f"{layer_disp} (assumed log2-scale)"


def subset_meta_mask(meta: pd.DataFrame, subset: dict[str, Any] | None) -> pd.Series:
    if not subset:
        return pd.Series(True, index=meta.index)
    mask = pd.Series(True, index=meta.index)
    for key, val in subset.items():
        mask &= meta[key].astype(str) == str(val)
    return mask

def preflight_donor_design(
    meta: pd.DataFrame,
    *,
    donor_col: str,
    group_col: str,
    require_paired_donors: bool,
) -> dict[str, Any]:
    """Compute donor pairing diagnostics."""
    sub = meta[[donor_col, group_col]].dropna()
    levels = sorted(sub[group_col].astype(str).unique())
    # Keep native donor dtype for groupby index alignment (int vs str donors).
    donors = sorted(sub[donor_col].unique(), key=str)

    donor_level_counts = (
        sub.groupby([donor_col, group_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    n_cells_per = donor_level_counts.to_dict(orient="index")

    donors_with_both = []
    donors_only: dict[str, list[str]] = {lv: [] for lv in levels}
    for donor in donors:
        if donor not in donor_level_counts.index:
            continue
        row = donor_level_counts.loc[donor]
        present = [lv for lv in levels if row.get(lv, 0) > 0]
        if len(present) == len(levels):
            donors_with_both.append(donor)
        else:
            for lv in present:
                donors_only[lv].append(str(donor))

    paired_mask = meta[donor_col].isin(donors_with_both)
    frac_paired = float(paired_mask.mean()) if len(meta) else 0.0

    diag: dict[str, Any] = {
        "n_donors_total": len(donors),
        "n_donors_paired": len(donors_with_both),
        "group_levels": levels,
        "n_cells_per_donor_group": n_cells_per,
        "fraction_cells_paired_donors": frac_paired,
        "donors_with_both_conditions": donors_with_both,
    }
    for lv in levels:
        diag[f"n_donors_{lv}_only"] = len(donors_only[lv])

    if require_paired_donors and len(donors_with_both) < len(donors):
        raise ValueError(
            "`require_paired_donors=True` but some donors lack all group levels. "
            f"Paired donors: {len(donors_with_both)}/{len(donors)}."
        )
    return diag

def _attempted_mixed_model(*, method: str, observation_level: str) -> bool:
    if observation_level == "pseudobulk":
        return False
    if method == "pseudobulk":
        return False
    return method in ("auto", "mixedlm")


def summarize_per_feature_testing(
    volcano_df: pd.DataFrame,
    *,
    method: str,
    observation_level: str,
) -> tuple[str | None, str | None]:
    """Summarize per-feature test routing and aggregate failure reasons."""
    if volcano_df.empty or "de_method" not in volcano_df.columns:
        return None, None

    attempted_mixed = _attempted_mixed_model(
        method=method, observation_level=observation_level
    )
    succeeded_mask = volcano_df["p_value"].notna()
    failed_mask = volcano_df["p_value"].isna()
    method_counts: dict[str, int] = {}
    if attempted_mixed:
        method_counts["mixed model"] = int(
            ((volcano_df["de_method"] == "mixedlm") & succeeded_mask).sum()
        )

    for internal, label in (
        ("pseudobulk_paired", "pseudobulk (paired)"),
        ("pseudobulk_unpaired", "pseudobulk (unpaired)"),
    ):
        n = int(((volcano_df["de_method"] == internal) & succeeded_mask).sum())
        if n:
            method_counts[label] = n

    n_failed = int(failed_mask.sum())
    if n_failed:
        method_counts["failed"] = n_failed

    if not method_counts:
        return None, None

    counts_str = ", ".join(f"{label}={count}" for label, count in method_counts.items())
    if method == "auto":
        prefix = "Per-feature testing (auto):"
    elif method == "mixedlm":
        prefix = "Per-feature testing (mixed model):"
    else:
        prefix = "Per-feature testing (pseudobulk):"
    testing_line = f"{prefix} {counts_str}"

    reason_parts: dict[str, int] = {}
    if "mixedlm_failure_reason" in volcano_df.columns:
        for reason, cnt in (
            volcano_df["mixedlm_failure_reason"].dropna().astype(str).value_counts().items()
        ):
            label = DE_FAILURE_LABELS.get(reason, reason)
            reason_parts[label] = reason_parts.get(label, 0) + int(cnt)
    if "de_failure_reason" in volcano_df.columns:
        for reason, cnt in (
            volcano_df.loc[failed_mask, "de_failure_reason"]
            .dropna()
            .astype(str)
            .value_counts()
            .items()
        ):
            label = DE_FAILURE_LABELS.get(reason, reason)
            reason_parts[label] = reason_parts.get(label, 0) + int(cnt)

    failure_line = None
    if reason_parts:
        detail = ", ".join(f"{label}={count}" for label, count in reason_parts.items())
        failure_line = f"Failure details: {detail}"
    return testing_line, failure_line


def summarize_per_feature_testing_for_results(
    contrast_results: dict[str, pd.DataFrame],
    *,
    contrast_mode: ContrastMode,
    method: str,
    observation_level: str,
) -> tuple[str | None, str | None]:
    """Summarize per-feature testing across contrast result tables."""
    if contrast_mode == "specified":
        return summarize_per_feature_testing(
            next(iter(contrast_results.values())),
            method=method,
            observation_level=observation_level,
        )
    testing_parts: list[str] = []
    failure_parts: list[str] = []
    for label, vdf in contrast_results.items():
        testing_line, failure_line = summarize_per_feature_testing(
            vdf,
            method=method,
            observation_level=observation_level,
        )
        if testing_line:
            testing_parts.append(f"{label}: {testing_line}")
        if failure_line:
            failure_parts.append(f"{label}: {failure_line}")
    testing_summary = "; ".join(testing_parts) if testing_parts else None
    failure_summary = "; ".join(failure_parts) if failure_parts else None
    return testing_summary, failure_summary


def print_mixed_de_info_section(
    *,
    diag: dict[str, Any],
    group_col: str,
    group_sizes: str,
    n_features_tested: int,
    n_features_total: int,
    per_feature_testing: str | None = None,
    per_feature_failures: str | None = None,
) -> None:
    """Print indented design diagnostics (same indent level as Mixed DE complete)."""
    # Match ``format_log_prefix('result_only', indent=2)`` spacing used by de()/mixed complete.
    prefix = format_log_prefix("info", 2)
    bullet = "       🔸 "
    print(f"{prefix} Mixed DE design diagnostics")
    levels = diag["group_levels"]
    only_counts = [int(diag.get(f"n_donors_{lv}_only", 0)) for lv in levels]
    only_detail = ", ".join(f"{cnt} {lv}-only" for lv, cnt in zip(levels, only_counts))
    donor_line = (
        f"Donors: {diag['n_donors_total']} total | "
        f"{diag['n_donors_paired']}/{diag['n_donors_total']} paired "
        f"({only_detail}) | "
        f"{diag['fraction_cells_paired_donors']:.0%} cells from paired"
    )
    print(f"{bullet}{donor_line}")
    n_excluded = n_features_total - n_features_tested
    features_line = f"Features: {n_features_tested}/{n_features_total}"
    if n_excluded:
        features_line += f" ({n_excluded} excluded)"
    features_line += f" | Group sizes after filtering: {group_sizes}"
    print(f"{bullet}{features_line}")
    if per_feature_testing:
        print(f"{bullet}{per_feature_testing}")
    if per_feature_failures:
        print(f"{bullet}{per_feature_failures}")

def format_comparing_groups(
    *,
    contrast_mode: ContrastMode,
    contrasts: list[tuple[str, str, str]],
    values: list[dict[str, Any]] | None,
    focal_level: str | None,
) -> str:
    """Human-readable contrast description for run logs."""
    if contrast_mode == "specified":
        if values is not None:
            return format_de_comparison_label(values[0], values[1])
        if contrasts:
            return contrasts[0][2]
        return "specified contrast"
    if contrast_mode == "pairwise":
        labels = [label for _, _, label in contrasts]
        return "pairwise: " + ", ".join(labels)
    if contrast_mode == "one_vs_rest":
        return f"one_vs_rest: {focal_level!r} vs all other levels"
    return contrast_mode

def format_group_sizes(
    meta: pd.DataFrame,
    group_col: str,
    *,
    contrast_mode: ContrastMode,
    contrasts: list[tuple[str, str, str]],
    values: list[dict[str, Any]] | None,
) -> str:
    """Observation counts per compared group (after ``subset`` and contrast filtering)."""
    col = meta[group_col].astype(str)
    if contrast_mode == "specified" and contrasts:
        label = contrasts[0][2]
        parts = [p.strip() for p in label.split(" vs ", 1)]
        if len(parts) == 2:
            n0 = int((col == parts[0]).sum())
            n1 = int((col == parts[1]).sum())
            return f"{n0} vs {n1} samples"
    if values is not None and contrast_mode == "specified":
        counts = []
        for val in values:
            if group_col in val:
                level = val[group_col]
            else:
                # Fallback: first differing-looking entry (legacy multi-key dicts).
                level = next(iter(val.values()))
            counts.append(int((col == str(level)).sum()))
        return f"{counts[0]} vs {counts[1]} samples"
    level_counts = col.value_counts().sort_index()
    breakdown = ", ".join(f"{lv}={int(cnt)}" for lv, cnt in level_counts.items())
    return f"{len(meta)} samples ({breakdown})"

def summarize_de_methods(volcano_df: pd.DataFrame) -> str | None:
    """Return raw ``de_method`` breakdown string for one volcano table."""
    if "de_method" not in volcano_df.columns:
        return None
    methods = volcano_df["de_method"].dropna().astype(str)
    if methods.empty:
        return None
    counts = methods.value_counts()
    if len(counts) == 1:
        return str(counts.index[0])
    return ", ".join(f"{name}={int(cnt)}" for name, cnt in counts.items())


def format_mixed_de_formula_display(
    *,
    fixed_formula: str,
    observation_level: str,
    random_effects: RandomEffects,
    donor_col: str,
    group_col: str,
    slope_col: str,
) -> str:
    """Human-readable model description for mixed_de run logs."""
    if observation_level == "pseudobulk":
        return f"Pseudobulk: donor-averaged by {donor_col} × {group_col}"
    if random_effects == "intercept_slope":
        re_term = f"(1 + {slope_col} | {donor_col})"
    else:
        re_term = f"(1 | {donor_col})"
    return f"{fixed_formula} + {re_term}"


def format_mixed_de_donor_covariates_line(
    *,
    fixed_covariates: list[str] | None,
    random_effects: RandomEffects,
    donor_col: str,
    observation_level: str,
) -> str:
    """Donor blocking and fixed covariates for run logs (biologist-friendly)."""
    if fixed_covariates:
        covariates = ", ".join(fixed_covariates)
    else:
        covariates = "(none)"
    if observation_level == "pseudobulk":
        blocking = f"{donor_col} (pseudobulk averaging)"
    elif random_effects == "intercept_slope":
        blocking = f"{donor_col} (intercept + slope)"
    else:
        blocking = f"{donor_col} (intercept)"
    return f"Donor blocking: {blocking} | Fixed covariates: {covariates}"


def format_mixed_de_comparing_line(
    *,
    comparing: str,
    group_sizes_before: str | None,
    path: str,
    contrast_mode: ContrastMode,
) -> str:
    """Compact comparing / sample-size / path summary for run logs."""
    parts = [f"Comparing: {comparing}"]
    if group_sizes_before is not None:
        parts.append(f"{group_sizes_before} (before filter)")
    parts.append(f"{path} / {contrast_mode}")
    return " | ".join(parts)


def print_mixed_de_donor_warnings(
    *,
    n_donors_total: int,
    n_donors_paired: int,
    method: str,
    observation_level: str,
    random_effects: RandomEffects,
) -> None:
    """Warn when donor count may destabilize mixed models or paired pseudobulk."""
    messages: list[str] = []
    attempts_mixed = _attempted_mixed_model(
        method=method, observation_level=observation_level
    )
    if n_donors_paired < MIN_DONORS_MIXEDLM_WARN:
        if attempts_mixed:
            messages.append(
                f"Only {n_donors_paired} paired donor(s): cell-level mixed models "
                "often fail to converge"
                + (
                    "; with method='auto', most features fall back to "
                    "donor-averaged pseudobulk"
                    if method == "auto"
                    else ""
                )
                + "."
            )
        if observation_level in ("pseudobulk", "auto") or method == "pseudobulk":
            messages.append(
                f"Paired pseudobulk tests require >= {MIN_DONORS_PAIRED_PSEUDOBULK} "
                f"paired donors; unpaired donor-level tests will be used."
            )
    if (
        random_effects == "intercept_slope"
        and n_donors_paired < MIN_DONORS_MIXEDLM_WARN
    ):
        messages.append(
            f"random_effects='intercept_slope' with only {n_donors_paired} paired "
            "donor(s) may be singular or unstable."
        )
    if n_donors_total < 2:
        messages.append("Fewer than 2 donors total; donor-blocked testing is not reliable.")

    for message in messages:
        print(f"{format_log_prefix('warn')} {message}")


def print_mixed_de_run_header(
    *,
    on: str,
    comparing: str,
    group_sizes_before: str | None,
    path: str,
    contrast_mode: ContrastMode,
    formula: str,
    donor_col: str,
    group_col: str,
    slope_col: str,
    layer_summary: str,
    method: str,
    observation_level: str,
    random_effects: RandomEffects,
    fixed_covariates: list[str] | None,
    subset: dict[str, Any] | None,
    correct_fdr: bool,
    threshold: float,
    log2fc_thresh: float,
    n_donors_total: int,
    n_donors_paired: int,
) -> None:
    """Print pre-run summary mirroring ``de()``."""
    log_prefix = format_log_prefix("user")
    formula_display = format_mixed_de_formula_display(
        fixed_formula=formula,
        observation_level=observation_level,
        random_effects=random_effects,
        donor_col=donor_col,
        group_col=group_col,
        slope_col=slope_col,
    )
    donor_covariates = format_mixed_de_donor_covariates_line(
        fixed_covariates=fixed_covariates,
        random_effects=random_effects,
        donor_col=donor_col,
        observation_level=observation_level,
    )
    comparing_line = format_mixed_de_comparing_line(
        comparing=comparing,
        group_sizes_before=group_sizes_before,
        path=path,
        contrast_mode=contrast_mode,
    )
    print(f"{log_prefix} Running mixed differential expression [{on}]")
    print(f"   🔸 {comparing_line}")
    if subset:
        subset_tag = ", ".join(f"{k}={v!r}" for k, v in subset.items())
        print(f"   🔸 Subset: {subset_tag}")
    print(f"   🔸 Formula: {formula_display}")
    print(f"   🔸 {donor_covariates}")
    print(
        f"   🔸 Layer: {layer_summary} | Method: {method} | "
        f"Level: {observation_level}"
    )
    print_de_threshold_line(
        correct_fdr=correct_fdr,
        threshold=threshold,
        log2fc_thresh=log2fc_thresh,
    )
    print_mixed_de_donor_warnings(
        n_donors_total=n_donors_total,
        n_donors_paired=n_donors_paired,
        method=method,
        observation_level=observation_level,
        random_effects=random_effects,
    )
    # Cell-level mixed models (and auto→LMM) are per-feature and often slow.
    if observation_level in ("cells", "subsample") and method in ("auto", "mixedlm"):
        print(
            f"{format_log_prefix('warn')} Cell-level mixed DE fits one model per "
            "feature and can take 5+ minutes depending on dataset size and CPU. "
            "For a faster run (especially with few donors), try "
            "observation_level='pseudobulk' or method='pseudobulk'."
        )


def summarize_de_methods_for_results(
    contrast_results: dict[str, pd.DataFrame],
    *,
    contrast_mode: ContrastMode,
) -> str | None:
    """Summarize raw per-feature ``de_method`` routing across contrast tables."""
    if contrast_mode == "specified":
        return summarize_de_methods(next(iter(contrast_results.values())))
    parts: list[str] = []
    for label, vdf in contrast_results.items():
        summary = summarize_de_methods(vdf)
        if summary:
            parts.append(f"{label}: {summary}")
    return "; ".join(parts) if parts else None


def _mixed_de_column_hint(*, correct_fdr: bool) -> str:
    return format_de_column_hint(
        correct_fdr=correct_fdr,
        extra_cols=("contrast", "de_method", "n_donors_paired", "random_effect_var"),
    )


def print_mixed_de_result_summary(
    volcano_df: pd.DataFrame,
    *,
    stats_location: str,
    correct_fdr: bool,
    contrast_mode: ContrastMode = "specified",
    contrast_label: str | None = None,
    include_storage_guide: bool = True,
) -> None:
    """Print post-run summary mirroring ``de()``."""
    extra_footer: list[str] | None = None
    if include_storage_guide and contrast_mode == "specified":
        extra_footer = [
            "",
            f"{stats_location}.attrs['mixed_de'] — run configuration "
            "(formula, donors, layer, observation level, …)",
        ]
    print_de_result_summary(
        volcano_df,
        title="Mixed DE",
        stats_location=stats_location,
        correct_fdr=correct_fdr,
        column_hint=_mixed_de_column_hint(correct_fdr=correct_fdr),
        contrast_label=contrast_label,
        include_storage_guide=include_storage_guide,
        extra_footer_lines=extra_footer,
    )


def print_mixed_de_collection_summary(
    contrast_results: dict[str, pd.DataFrame],
    *,
    stats_location: str,
    correct_fdr: bool,
    contrast_mode: ContrastMode,
) -> None:
    """Print summary for ``contrast_mode`` collection results (pairwise / one_vs_rest)."""
    print(f"{format_log_prefix('result_only', indent=2)} Mixed DE complete. Results stored in:")
    print(f"       • {stats_location}")
    print(f"       • Columns: {_mixed_de_column_hint(correct_fdr=correct_fdr)}")
    for label, vdf in contrast_results.items():
        print_mixed_de_result_summary(
            vdf,
            stats_location=stats_location,
            correct_fdr=correct_fdr,
            contrast_mode=contrast_mode,
            contrast_label=label,
            include_storage_guide=False,
        )
    print("")
    print(
        f"       • {stats_location}[\"meta\"] — run configuration "
        "(formula, donors, layer, observation level, …)"
    )


def feature_detection_mask(
    expr: np.ndarray,
    *,
    min_detected_fraction: float,
    min_cells_detected: int,
) -> np.ndarray:
    finite = np.isfinite(expr) & (expr != 0)
    n_det = finite.sum(axis=0)
    n_obs = expr.shape[0]
    frac = n_det / n_obs if n_obs else 0.0
    return (n_det >= min_cells_detected) & (frac >= min_detected_fraction)

def subsample_observations(
    meta: pd.DataFrame,
    expr: np.ndarray,
    *,
    donor_col: str,
    group_col: str,
    max_cells_per_stratum: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    keep_idx: list[int] = []
    for (_, _), idx in meta.groupby([donor_col, group_col], observed=True).groups.items():
        idx_list = list(idx)
        if len(idx_list) <= max_cells_per_stratum:
            keep_idx.extend(idx_list)
        else:
            chosen = rng.choice(idx_list, size=max_cells_per_stratum, replace=False)
            keep_idx.extend(chosen)
    keep_idx = sorted(keep_idx, key=lambda x: meta.index.get_loc(x))
    pos = meta.index.get_indexer(keep_idx)
    return meta.loc[keep_idx].copy(), expr[pos, :]

def _match_param_name(param_names: list[str], term: str, level: str) -> str | None:
    level = str(level)
    for name in param_names:
        if term in name and level in name:
            return name
    return None

def _build_wald_contrast(
    param_names: list[str],
    *,
    contrast_term: str,
    ref: str,
    test: str,
    contrast_at: dict[str, str] | None,
) -> np.ndarray | None:
    """Build contrast vector for test-vs-ref on contrast_term (optionally at contrast_at)."""
    vec = np.zeros(len(param_names))
    found = False

    main_name = _match_param_name(param_names, contrast_term, test)
    if main_name is not None:
        vec[param_names.index(main_name)] = 1.0
        found = True

    if contrast_at:
        for factor, level in contrast_at.items():
            inter_name = None
            for name in param_names:
                if contrast_term in name and factor in name and str(level) in name:
                    inter_name = name
                    break
            if inter_name is not None:
                vec[param_names.index(inter_name)] = 1.0
                found = True
    elif not found:
        return None
    return vec if found else None

def extract_contrast_result(
    fit,
    *,
    contrast_term: str,
    ref: str,
    test: str,
    contrast_at: dict[str, str] | None,
) -> tuple[float, float]:
    """Return (log2fc, p_value) from a fitted MixedLM or OLS result.

    Handles the usual treatment coding cases:
    - ``ref`` is the model baseline → use ``test`` coefficient
    - ``test`` is the model baseline → use ``-ref`` coefficient
    - neither is baseline (3+ levels) → Wald contrast ``test - ref``
    """
    param_names = list(fit.fe_params.index)

    if contrast_at is None:
        test_name = _match_param_name(param_names, contrast_term, test)
        ref_name = _match_param_name(param_names, contrast_term, ref)
        if test_name is not None and ref_name is None:
            return float(fit.fe_params[test_name]), float(fit.pvalues[test_name])
        if test_name is None and ref_name is not None:
            return -float(fit.fe_params[ref_name]), float(fit.pvalues[ref_name])
        if test_name is not None and ref_name is not None:
            vec = np.zeros(len(param_names))
            vec[param_names.index(test_name)] = 1.0
            vec[param_names.index(ref_name)] = -1.0
            ttest = fit.t_test(np.asarray(vec, dtype=float).reshape(1, -1))
            return float(np.atleast_1d(ttest.effect)[0]), float(np.atleast_1d(ttest.pvalue)[0])

    vec = _build_wald_contrast(
        param_names,
        contrast_term=contrast_term,
        ref=ref,
        test=test,
        contrast_at=contrast_at,
    )
    if vec is None:
        # If test is the baseline under treatment coding, fall back to -ref.
        ref_name = _match_param_name(param_names, contrast_term, ref)
        if contrast_at is None and ref_name is not None:
            return -float(fit.fe_params[ref_name]), float(fit.pvalues[ref_name])
        raise ValueError(
            f"Could not locate coefficient for {contrast_term}={test!r} "
            f"(vs ref={ref!r}) in model parameters: {param_names}"
        )

    ttest = fit.t_test(np.asarray(vec, dtype=float).reshape(1, -1))
    log2fc = float(np.atleast_1d(ttest.effect)[0])
    pval = float(np.atleast_1d(ttest.pvalue)[0])
    return log2fc, pval

def fit_mixedlm_gene(
    df: pd.DataFrame,
    *,
    formula: str,
    donor_col: str,
    random_effects: RandomEffects,
    re_slope_col: str | None,
    test_level: str,
    contrast_term: str,
    ref: str,
    test: str,
    contrast_at: dict[str, str] | None,
) -> tuple[float, float, bool, float | None, str | None]:
    """Fit one gene; return log2fc, pval, converged, random_effect_var, failure_reason."""
    work = df.dropna(subset=["expr", donor_col]).copy()
    if work[donor_col].nunique() < 2:
        return np.nan, np.nan, False, None, "insufficient_donors"
    if work["expr"].nunique() < 2:
        return np.nan, np.nan, False, None, "constant_expression"

    re_formula = None
    if random_effects == "intercept_slope":
        slope_col = re_slope_col or contrast_term
        work = work.copy()
        work["_mixed_slope_num"] = (
            work[slope_col].astype(str) == str(test_level)
        ).astype(float)
        re_formula = "~_mixed_slope_num"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                formula,
                work,
                groups=work[donor_col],
                re_formula=re_formula,
            )
            fit = model.fit(reml=True, disp=False)
        if not bool(fit.converged):
            re_var = float(fit.cov_re.iloc[0, 0]) if hasattr(fit, "cov_re") else None
            return np.nan, np.nan, False, re_var, "not_converged"
        log2fc, pval = extract_contrast_result(
            fit,
            contrast_term=contrast_term,
            ref=ref,
            test=test,
            contrast_at=contrast_at,
        )
        re_var = float(fit.cov_re.iloc[0, 0]) if hasattr(fit, "cov_re") else None
        if not np.isfinite(pval):
            return log2fc, np.nan, True, re_var, "non_finite_pvalue"
        return log2fc, pval, True, re_var, None
    except ValueError as exc:
        msg = str(exc).lower()
        if "could not locate coefficient" in msg:
            return np.nan, np.nan, False, None, "missing_contrast_coefficient"
        return np.nan, np.nan, False, None, "fit_error"
    except Exception:
        return np.nan, np.nan, False, None, "fit_error"

def fit_pseudobulk_gene(
    df: pd.DataFrame,
    *,
    donor_col: str,
    group_col: str,
    ref: str,
    test: str,
    paired_donors: list[str],
) -> tuple[float, float, str, int, str | None]:
    """Donor-level pseudobulk test for one gene.

    Returns log2fc, pval, method tag, n_donors, and failure_reason when pval is NaN.
    """
    work = df.dropna(subset=["expr", donor_col, group_col]).copy()
    if work.empty:
        return np.nan, np.nan, "pseudobulk", 0, "no_observations"

    pb = (
        work.groupby([donor_col, group_col], observed=True)["expr"]
        .mean()
        .unstack()
    )
    if ref not in pb.columns or test not in pb.columns:
        return np.nan, np.nan, "pseudobulk", 0, "missing_contrast_levels"

    if paired_donors:
        pb_pair = pb.loc[[d for d in paired_donors if d in pb.index]].dropna()
        if len(pb_pair) >= MIN_DONORS_PAIRED_PSEUDOBULK:
            a = pb_pair[ref].values
            b = pb_pair[test].values
            stat, pval = ttest_rel(b, a, nan_policy="omit")
            log2fc = float(np.nanmean(b - a))
            return log2fc, float(pval), "pseudobulk_paired", len(pb_pair), None

    ref_vals = pb[ref].dropna()
    test_vals = pb[test].dropna()
    if len(ref_vals) < 2 or len(test_vals) < 2:
        return (
            np.nan,
            np.nan,
            "pseudobulk_unpaired",
            int(len(ref_vals) + len(test_vals)),
            "insufficient_donors_per_arm",
        )
    stat, pval = ttest_ind(test_vals.values, ref_vals.values, equal_var=False, nan_policy="omit")
    log2fc = float(np.nanmean(test_vals) - np.nanmean(ref_vals))
    return log2fc, float(pval), "pseudobulk_unpaired", int(len(ref_vals) + len(test_vals)), None

def build_stats_key(
    *,
    contrast_label: str,
    subset: dict[str, Any] | None,
    donor_col: str,
    contrast_mode: ContrastMode,
    group_col: str,
) -> str:
    if contrast_mode != "specified":
        subset_tag = " | ".join(f"{k}={v}" for k, v in (subset or {}).items())
        base = f"mixed: {contrast_mode} | group_col={group_col}"
        return f"{base} | {subset_tag}" if subset_tag else base
    subset_tag = " | ".join(f"{k}={v}" for k, v in (subset or {}).items())
    parts = [f"mixed: {contrast_label}"]
    if subset_tag:
        parts.append(subset_tag)
    parts.append(f"donor={donor_col}")
    return " | ".join(parts)

def contrast_label_from_pair(test: str, ref: str) -> str:
    """Return ``'{test} vs {ref}'`` label (log2fc = test − ref)."""
    return f"{test} vs {ref}"

def list_contrasts_for_mode(
    levels: list[str],
    *,
    contrast_mode: ContrastMode,
    contrast: tuple[str, str] | None,
    focal_level: str | None,
) -> list[tuple[str, str, str]]:
    """Return list of ``(test, ref, label)`` contrasts to run.

    Public ``contrast`` is ``(test, ref)`` — same order as ``values=[group1, group2]``
    and DESeq2-style numerator/denominator (log2fc = test − ref).
    """
    if contrast_mode == "specified":
        if contrast is None:
            raise ValueError("`contrast` is required when contrast_mode='specified'.")
        test, ref = contrast
        return [(test, ref, contrast_label_from_pair(test, ref))]

    if contrast_mode == "one_vs_rest":
        if focal_level is None:
            raise ValueError("`focal_level` is required when contrast_mode='one_vs_rest'.")
        out = []
        for lv in levels:
            if lv == focal_level:
                continue
            # test=focal, ref=other
            out.append((focal_level, lv, contrast_label_from_pair(focal_level, lv)))
        return out

    # pairwise: alphabetical earlier level as ref, later as test
    out = []
    for i, ref in enumerate(levels):
        for test in levels[i + 1 :]:
            out.append((test, ref, contrast_label_from_pair(test, ref)))
    return out

def compile_volcano_dataframe(
    results: list[dict[str, Any]],
    feature_names: pd.Index,
    var: pd.DataFrame,
    *,
    contrast_label: str,
    n_donors_paired: int,
    correct_fdr: bool,
    threshold: float,
    log2fc_thresh: float,
    fdr_scope: FdrScope,
    global_pvals: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble volcano DataFrame matching de() conventions."""
    df = pd.DataFrame(index=feature_names)
    for key in ("log2fc", "p_value", "n_obs", "converged", "random_effect_var"):
        df[key] = np.nan
    df["de_method"] = pd.Series(index=feature_names, dtype=object)
    df["de_failure_reason"] = pd.Series(index=feature_names, dtype=object)
    df["mixedlm_failure_reason"] = pd.Series(index=feature_names, dtype=object)
    df["n_donors_paired"] = n_donors_paired
    df["contrast"] = contrast_label

    for row in results:
        feat = row["feature"]
        if feat not in df.index:
            continue
        for key in (
            "log2fc",
            "p_value",
            "de_method",
            "de_failure_reason",
            "mixedlm_failure_reason",
            "n_obs",
            "converged",
            "random_effect_var",
        ):
            if key in row:
                df.loc[feat, key] = row[key]

    if "Genes" in var.columns:
        df["Genes"] = var["Genes"].reindex(df.index)
    else:
        df["Genes"] = df.index

    df["-log10(p_value)"] = -np.log10(df["p_value"].replace(0, np.nan).astype(float))
    df["significance_score"] = df["-log10(p_value)"] * df["log2fc"]

    if correct_fdr:
        df["adj_p_value"] = bh_adjust_pvalues(df["p_value"].values)
        df["-log10(adj_p_value)"] = -np.log10(df["adj_p_value"].replace(0, np.nan).astype(float))
        if fdr_scope in ("global", "both") and global_pvals is not None:
            df["adj_p_value_global"] = global_pvals
        p_for_sig = df["adj_p_value"]
    else:
        p_for_sig = df["p_value"]

    df["significance"] = "not significant"
    mask_nc = df["log2fc"].isna() | df["p_value"].isna()
    df.loc[mask_nc, "significance"] = "not comparable"
    df.loc[(p_for_sig < threshold) & (df["log2fc"] > log2fc_thresh), "significance"] = "upregulated"
    df.loc[(p_for_sig < threshold) & (df["log2fc"] < -log2fc_thresh), "significance"] = "downregulated"
    cat = ["upregulated", "downregulated", "not significant", "not comparable"]
    df["significance"] = pd.Categorical(df["significance"], categories=cat, ordered=True)
    return df.sort_values(by="significance")

def run_single_contrast(
    expr: np.ndarray,
    meta: pd.DataFrame,
    feature_names: pd.Index,
    detect_mask: np.ndarray,
    *,
    formula: str,
    contrast_term: str,
    ref: str,
    test: str,
    contrast_at: dict[str, str] | None,
    donor_col: str,
    random_effects: RandomEffects,
    re_slope_col: str | None,
    observation_level: ObservationLevel,
    method: DeMethod,
    paired_donors: list[str],
    n_donors_paired: int,
    min_donors_for_slope_warn: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run per-feature models for one contrast. Returns (volcano_rows, diagnostic_rows)."""
    volcano_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    # One shared frame; only the expr column changes per feature (fits copy internally).
    gene_df = meta.copy()

    for j, feat in enumerate(feature_names):
        if not detect_mask[j]:
            continue
        gene_df["expr"] = expr[:, j]

        use_pseudobulk = observation_level == "pseudobulk"
        mixedlm_failure_reason: str | None = None
        de_failure_reason: str | None = None

        if use_pseudobulk or (method == "pseudobulk" and not use_pseudobulk):
            log2fc, pval, tag, n_don, de_failure_reason = fit_pseudobulk_gene(
                gene_df,
                donor_col=donor_col,
                group_col=contrast_term,
                ref=ref,
                test=test,
                paired_donors=paired_donors,
            )
            volcano_rows.append(
                {
                    "feature": feat,
                    "log2fc": log2fc,
                    "p_value": pval,
                    "de_method": tag,
                    "de_failure_reason": de_failure_reason,
                    "n_obs": int(gene_df["expr"].notna().sum()),
                }
            )
            diag_rows.append({"feature": feat, "converged": not np.isnan(pval), "n_donors_used": n_don})
            continue

        if method == "auto":
            log2fc, pval, converged, re_var, mixedlm_failure_reason = fit_mixedlm_gene(
                gene_df,
                formula=formula,
                donor_col=donor_col,
                random_effects=random_effects,
                re_slope_col=re_slope_col,
                test_level=test,
                contrast_term=contrast_term,
                ref=ref,
                test=test,
                contrast_at=contrast_at,
            )
            if mixedlm_failure_reason is None and np.isfinite(pval):
                de_method_used = "mixedlm"
                de_failure_reason = None
            else:
                log2fc, pval, tag, n_don, de_failure_reason = fit_pseudobulk_gene(
                    gene_df,
                    donor_col=donor_col,
                    group_col=contrast_term,
                    ref=ref,
                    test=test,
                    paired_donors=paired_donors,
                )
                de_method_used = tag
            volcano_rows.append(
                {
                    "feature": feat,
                    "log2fc": log2fc,
                    "p_value": pval,
                    "de_method": de_method_used,
                    "de_failure_reason": de_failure_reason,
                    "mixedlm_failure_reason": mixedlm_failure_reason,
                    "n_obs": int(gene_df["expr"].notna().sum()),
                }
            )
            diag_rows.append(
                {
                    "feature": feat,
                    "converged": converged,
                    "random_effect_var": re_var,
                }
            )
            continue

        log2fc, pval, converged, re_var, de_failure_reason = fit_mixedlm_gene(
            gene_df,
            formula=formula,
            donor_col=donor_col,
            random_effects=random_effects,
            re_slope_col=re_slope_col,
            test_level=test,
            contrast_term=contrast_term,
            ref=ref,
            test=test,
            contrast_at=contrast_at,
        )
        volcano_rows.append(
            {
                "feature": feat,
                "log2fc": log2fc,
                "p_value": pval,
                "de_method": "mixedlm",
                "de_failure_reason": de_failure_reason,
                "n_obs": int(gene_df["expr"].notna().sum()),
            }
        )
        diag_rows.append(
            {"feature": feat, "converged": converged, "random_effect_var": re_var}
        )

    return volcano_rows, diag_rows
