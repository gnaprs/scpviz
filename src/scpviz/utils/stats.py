"""Statistical and differential-expression helpers for scpviz."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control, ttest_ind, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA

from scpviz.utils.de_reporting import format_de_group_label
from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

def bh_adjust_pvalues(pvals: np.ndarray | list[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjustment for a 1-D array of p-values.

    NaN and non-finite entries are left unchanged; only finite values are corrected.
    """
    pvals_arr = np.asarray(pvals, dtype=float)
    adj_pvals = np.full(pvals_arr.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(pvals_arr)
    n_valid = int(valid_mask.sum())
    if n_valid > 1:
        adj_pvals[valid_mask] = false_discovery_control(
            pvals_arr[valid_mask], method="bh"
        )
    elif n_valid == 1:
        adj_pvals[valid_mask] = pvals_arr[valid_mask]
    return adj_pvals

def pairwise_log2fc(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise median log2 fold change (log2FC) between two groups.

    This function calculates all pairwise log2 ratios between features in
    two groups of samples and returns the median value per feature. It is
    primarily used as a helper for fold-change strategies in `pAnnData.de()`.

    Args:
        data1 (numpy.ndarray): Array of shape `(n_samples_group1, n_features)`
            containing abundance values for group 1.
        data2 (numpy.ndarray): Array of shape `(n_samples_group2, n_features)`
            containing abundance values for group 2.

    Returns:
        median_log2fc (numpy.ndarray): Array of shape `(n_features,)` containing
        the median pairwise log2 fold change for each feature.

    Note:
        This is an internal helper for differential expression calculations.
        End users should call `pAnnData.de()` instead of using this function directly.

    Related Functions:
        - pAnnData.de: Differential expression analysis with multiple fold change strategies.
    """
    n1, n2 = data1.shape[0], data2.shape[0]

    # data1[:, None, :] has shape (n1, 1, n_features)
    # data2[None, :, :] has shape (1, n2, n_features)
    # The result is an array of shape (n1, n2, n_features)
    with np.errstate(divide='ignore', invalid='ignore'):
        pairwise_ratios = np.log2(data1[:, None, :] / data2[None, :, :])  # (n1, n2, features)
        pairwise_flat = pairwise_ratios.reshape(-1, data1.shape[1])

    # Identify columns that are entirely NaN
    mask_all_nan = np.all(np.isnan(pairwise_flat), axis=0)
    median_fc = np.full(data1.shape[1], np.nan, dtype=float)

    # Compute only on valid columns
    if not np.all(mask_all_nan):
        valid_cols = ~mask_all_nan
        median_fc[valid_cols] = np.nanmedian(pairwise_flat[:, valid_cols], axis=0)

    # # Reshape to (n1*n2, n_features) and compute the median along the first axis.
    # median_fc = np.nanmedian(pairwise_ratios.reshape(-1, data1.shape[1]), axis=0)
    return median_fc

def de_adata(
    adata: ad.AnnData,
    values: list[dict[str, Any] | list[str]] | None = None,
    class_type: str | list[str] | None = None,
    method: str = "ttest",
    fold_change_mode: str = "mean",
    layer: str = "X",
    threshold: float = 0.05,
    log2fc: float = 1.0,
    correct_fdr: bool = False,
    equal_var: bool = True,
    pval: float | None = None,
    data_is_log: bool = False,
    log_base: float = 2.0,
    pseudocount: float = 1.0,
    gene_col: str | None = None,
) -> pd.DataFrame:
    """
    Standalone DE analysis for AnnData. Produces a volcano-ready DataFrame identical to pdata.de().

    Supports:
        - Legacy-style: class_type="condition", values=["A","B"]
        - Legacy multi-col: class_type=["cellline","treatment"],
                            values=[["HCT116","DMSO"], ["HCT116","Drug"]]
        - Dictionary-style: values=[{"cellline":"HCT116","treatment":"DMSO"}, {...}]    

    Args:
        adata (AnnData): AnnData object.
        values (list of dict or list of list): Sample group filters to compare.

            - Dictionary-style (recommended): [{'cellline': 'HCT116', 'treatment': 'DMSO'}, {...}]
            - Legacy-style (if `class_type` is provided): [['HCT116', 'DMSO'], ['HCT116', 'DrugX']]

        class_type (str or list of str, optional): Legacy-style class label(s) to interpret `values`.
        method (str): 'ttest', 'mannwhitneyu', 'wilcoxon'.
        fold_change_mode (str): 'mean' or 'pairwise_median'.
        layer (str): Layer to use. Default is 'X'.
        threshold (float): Significance cutoff. Applied to raw ``p_value`` when
            ``correct_fdr=False``, and to ``adj_p_value`` when ``correct_fdr=True``.
        pval (float, optional): Deprecated alias for ``threshold``.
        log2fc (float): Minimum absolute log2 fold change for significance labeling.
        correct_fdr (bool): If True, apply Benjamini-Hochberg FDR correction and
            label significance using adjusted p-values.
        equal_var (bool): Passed to :func:`scipy.stats.ttest_ind` when ``method='ttest'``.
            ``True`` (default) uses Student's t-test; ``False`` uses Welch's t-test.
        data_is_log (bool): If True, treat `layer` as log-transformed and
            un-log to compute fold changes.
        log_base (float): Base of the log used in `layer`. Default 2.0.
        pseudocount (float): If data is log of (x + pseudocount), provide that
            here (e.g., 1.0 for log2(x+1)).
        gene_col (str, optional): Column in `adata.var` to use for the "Genes"
            field in the output. Will use:
            - `adata.var['Genes']` by default,
            - `adata.var[<gene_col>]` if provided by the user, otherwise
            - `adata.var_names` if the above do not exist.

    Returns:
        pandas.DataFrame: DE results with volcano-ready columns.
    """
    if pval is not None:
        print(
            f"{format_log_prefix('warn')} `pval` is deprecated in de_adata(); "
            f"use `threshold` instead (applied pval={pval})."
        )
        if pval != threshold:
            print(
                f"{format_log_prefix('warn')} Both `threshold`={threshold} and "
                f"`pval={pval}` were passed to de_adata(); using `pval`."
            )
        threshold = pval

    def to_dict_list(class_type, val):
        """Convert legacy values into a list of dictionary filters."""
        if isinstance(val, dict):
            return [val]

        # if class_type is singular
        if isinstance(class_type, str):
            return [{class_type: val}]

        # if class_type is list (multi-column)
        if isinstance(class_type, list) and isinstance(val, list):
            if len(class_type) != len(val):
                raise ValueError("Length mismatch: class_type and values.")
            return [dict(zip(class_type, val))]

        raise ValueError("Invalid legacy DE input format.")

    def _unlog(data, data_is_log, log_base=2.0, pseudocount=0.0):
        """Convert log-transformed data back to linear scale for FC calc."""
        if not data_is_log:
            return data

        # data are log_base(x + pseudocount)
        with np.errstate(over='ignore', invalid='ignore'):
            if log_base == 2.0:
                lin = np.power(2.0, data) - pseudocount
            elif log_base == np.e:
                lin = np.exp(data) - pseudocount
            else:
                lin = np.power(log_base, data) - pseudocount

        # Clamp small negatives due to numerical noise
        lin[lin < 0] = 0.0
        return lin
    
    # identify sample indices for each group
    def filter_indices(adata, filters):
        """Return sample indices matching a list of dict filters."""
        mask = np.ones(len(adata), dtype=bool)
        for f in filters:
            for col, val in f.items():
                mask &= (adata.obs[col].astype(str) == str(val))
        return np.where(mask)[0]
    
    if values is None:
        raise ValueError("Please supply `values` (2 groups) for DE.")

    if len(values) != 2:
        raise ValueError("`values` must contain exactly two group definitions.")

    if values[0] == values[1]:
        raise ValueError("Both groups in `values` refer to the same condition. Please provide two distinct groups.")

    # convert values to standardized dict format
    if isinstance(values[0], dict):
        group1_filters = [values[0]]
        group2_filters = [values[1]]
    else:
        if class_type is None:
            raise ValueError("class_type must be provided for legacy DE format.")
        group1_filters = to_dict_list(class_type, values[0])
        group2_filters = to_dict_list(class_type, values[1])


    idx1 = filter_indices(adata, group1_filters)
    idx2 = filter_indices(adata, group2_filters)

    if len(idx1) == 0 or len(idx2) == 0:
        raise ValueError("One of the groups has zero samples.")

    # extract matrices
    if layer == "X":
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]

    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    data1 = X[idx1, :]
    data2 = X[idx2, :]

    data1_fc = _unlog(data1, data_is_log=data_is_log, log_base=log_base, pseudocount=pseudocount)
    data2_fc = _unlog(data2, data_is_log=data_is_log, log_base=log_base, pseudocount=pseudocount)

    # log2FC computation

    if fold_change_mode == 'mean':
        with np.errstate(all='ignore'):
            m1 = np.nanmean(data1_fc, axis=0)
            m2 = np.nanmean(data2_fc, axis=0)
            mask_invalid = (m1 == 0) | (m2 == 0) | np.isnan(m1) | np.isnan(m2)
            log2fc_vals = np.log2(m1 / m2)
            log2fc_vals[mask_invalid] = np.nan

    elif fold_change_mode == 'pairwise_median':
        mask_invalid = ( # Detect invalid features (any 0 or NaN in either group)
            np.any((data1 == 0) | np.isnan(data1), axis=0) |
            np.any((data2 == 0) | np.isnan(data2), axis=0)
        )
        # Compute median pairwise log2FC
        log2fc_vals = pairwise_log2fc(data1, data2)
        log2fc_vals[mask_invalid] = np.nan # Mark invalid features as NaN
        n_invalid = np.sum(mask_invalid)
        if n_invalid > 0:
            print(f"{format_log_prefix('info',2)} {n_invalid} proteins were not comparable (zero or NaN mean in one group).")

    else:
        raise ValueError(f"Unsupported fold_change_mode '{fold_change_mode}'")

    # statistical test

    pvals = []
    stats = []

    for i in range(X.shape[1]):
        x1, x2 = data1[:, i], data2[:, i]
        if method not in {"ttest", "mannwhitneyu", "wilcoxon"}:
            raise ValueError(f"Unsupported method '{method}'")

        try:
            if method == 'ttest':
                res = ttest_ind(x1, x2, equal_var=equal_var, nan_policy='omit')
            elif method == 'mannwhitneyu':
                res = mannwhitneyu(x1, x2, alternative='two-sided')
            elif method == 'wilcoxon':
                res = wilcoxon(x1, x2)
            pvals.append(res.pvalue)
            stats.append(res.statistic)
        except Exception:
            pvals.append(np.nan)
            stats.append(np.nan)


    pvals = np.array(pvals)

    # mean abundance
    mean1 = np.nanmean(data1, axis=0)
    mean2 = np.nanmean(data2, axis=0)

    group1_label = format_de_group_label(group1_filters)
    group2_label = format_de_group_label(group2_filters)

    # assemble DataFrame (pAnnData-compatible)
    df = pd.DataFrame(index=adata.var_names)

    if gene_col is not None:
        # User-specified or default "Genes"
        if gene_col in adata.var.columns:
            df["Genes"] = adata.var[gene_col].astype(str).values
        else:
            raise KeyError(
                f"Requested gene_col='{gene_col}', but this column is not in adata.var.\n"
                f"Available columns: {list(adata.var.columns)}"
            )
    else:
        # Fallback logic: use adata.var['Genes'] if it exists
        if "Genes" in adata.var.columns:
            df["Genes"] = adata.var["Genes"].astype(str).values
        else:
            df["Genes"] = adata.var_names.astype(str)

    df[group1_label] = mean1
    df[group2_label] = mean2
    df["log2fc"] = log2fc_vals
    df["p_value"] = pvals
    df["test_statistic"] = stats
    df["-log10(p_value)"] = -np.log10(np.where(pvals == 0, np.nan, pvals))
    df["significance_score"] = df["-log10(p_value)"] * df["log2fc"]

    if correct_fdr:
        df["adj_p_value"] = bh_adjust_pvalues(pvals)
        df["-log10(adj_p_value)"] = -np.log10(
            df["adj_p_value"].replace(0, np.nan).astype(float)
        )
        p_for_sig = df["adj_p_value"]
    else:
        p_for_sig = df["p_value"]

    # significance classification
    df["significance"] = "not significant"
    df.loc[df["log2fc"].isna(), "significance"] = "not comparable"
    df.loc[(p_for_sig < threshold) & (df["log2fc"] > log2fc), "significance"] = "upregulated"
    df.loc[(p_for_sig < threshold) & (df["log2fc"] < -log2fc), "significance"] = "downregulated"

    df["significance"] = pd.Categorical(
        df["significance"],
        categories=["upregulated", "downregulated", "not significant", "not comparable"],
        ordered=True,
    )

    # group labels for plotting annotation
    df.attrs["group1_label"] = group1_label
    df.attrs["group2_label"] = group2_label

    return df

def get_pca_importance(
    model: dict[str, Any] | PCA,
    initial_feature_names: list[str],
    n: int = 1,
) -> pd.DataFrame:
    """
    Identify the most important features for each principal component.

    This function ranks features by their absolute PCA loading values and
    extracts the top contributors for each principal component.

    Args:
        model (sklearn.decomposition.PCA or dict): Either a fitted PCA model
            from scikit-learn, or a dictionary with key `"PCs"`
            (array-like, shape: `(n_components, n_features)`).
        initial_feature_names (list of str): Names of the features, typically
            `adata.var_names`.
        n (int): Number of top features to return per principal component
            (default = 1).

    Returns:
        df (pandas.DataFrame): DataFrame with one row per principal component,
        listing the top contributing features.

    Example:
        Retrieve the top 5 features contributing to each PC:
            ```python
            from scpviz import utils as scutils
            pdata.pca(n_components=5)
            df = scutils.get_pca_importance(
                pdata.prot.uns['pca'],
                pdata.prot.var_names,
                n=5
            )
            ```
    """

    if isinstance(model, dict):
        pcs = np.asarray(model["PCs"])  # shape: n_components x n_features
    else:
        pcs = np.asarray(model.components_)  # shape: n_components x n_features

    n_pcs = pcs.shape[0]

    most_important = [
        np.abs(pcs[i]).argsort()[-n:][::-1] for i in range(n_pcs)
    ]
    most_important_names = [
        [initial_feature_names[idx] for idx in row] for row in most_important
    ]

    result = {
        f"PC{i + 1}": most_important_names[i] for i in range(n_pcs)
    }
    df = pd.DataFrame(result.items(), columns=["Principal Component", "Top Features"])
    return df
    
def get_protein_clusters(
    pdata: pAnnData,
    on: str = "prot",
    layer: str = "X",
    t: int = 5,
    criterion: str = "maxclust",
) -> dict[Any, list[str]] | None:
    """
    Retrieve hierarchical clusters of proteins from stored linkage.

    This function uses linkage information stored in `pdata.stats` to
    partition proteins into clusters.

    Args:
        pdata (pAnnData): Input object containing `.stats` with clustering results.
        on (str): Data level to use, `"prot"` (default) or `"pep"`.
        layer (str): Data layer name used when the linkage was computed (default = `"X"`).
        t (int or float): Number of clusters (if `criterion="maxclust"`) or distance
            threshold for clustering.
        criterion (str): Clustering criterion passed to `scipy.cluster.hierarchy.fcluster`,
            e.g. `"maxclust"` or `"distance"`.

    Returns:
        clusters (dict): Mapping of `cluster_id → list of proteins`.
        None: If no linkage is found in `pdata.stats`.

    Note:
        Requires that a clustermap has been previously computed and linkage
        stored under `pdata.stats[f"{on}_{layer}_clustermap"]`.

    Related Functions:
        - plot_clustermap: Generates clustered heatmaps and stores linkage.
    """
    from scipy.cluster.hierarchy import fcluster
    
    key = f"{on}_{layer}_clustermap"
    stats = pdata.stats.get(key)
    if not stats or "row_linkage" not in stats:
        print(f"No linkage found for {key} in pdata.stats.")
        return None

    linkage = stats["row_linkage"]
    labels = fcluster(linkage, t=t, criterion=criterion)
    order = stats["row_order"]

    from collections import defaultdict
    clusters = defaultdict(list)
    for label, prot in zip(labels, order):
        clusters[label].append(prot)

    return dict(clusters)


def correlation_linkage(
    X: np.ndarray,
    method: str = "pearson",
    linkage_method: str = "average",
    optimal_ordering: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a hierarchical clustering linkage matrix using correlation distance.

    Distance is defined as ``1 - correlation(x, y)`` for each pair of rows in ``X``.
    Rows are typically feature profiles across samples (e.g. z-scored protein
    abundance). Aggregate to group means before calling if per-cell relationships
    are not the intent.

    Args:
        X: Array of shape ``(n_rows, n_features)``. Each row is a profile to
            compare pairwise. Must not contain NaN (impute upstream).
        method: Correlation method, either ``"pearson"`` or ``"spearman"``.
            Spearman is Pearson correlation on rank-transformed rows (ties via
            average rank).
        linkage_method: Linkage rule passed to ``scipy.cluster.hierarchy.linkage``.
            Defaults to ``"average"`` (UPGMA).
        optimal_ordering: If True, reorders the linkage so adjacent leaves are as
            similar as possible (display only; does not change cluster membership).

    Returns:
        A tuple of:

        - ``Z``: ``(n_rows - 1, 4)`` linkage matrix for ``dendrogram()``.
        - ``condensed_dist``: Condensed pairwise distance vector used to build ``Z``.

    Raises:
        ValueError: If ``method`` is invalid, or any row of ``X`` has zero variance
            (correlation undefined).

    Note:
        Uses signed ``1 - r``, not ``1 - |r|`` — anti-correlated profiles are treated
        as dissimilar.

    Example:
        Compute linkage for group-averaged profiles:
            ```python
            from scpviz.utils import correlation_linkage
            from scipy.cluster.hierarchy import dendrogram

            Z, dist = correlation_linkage(group_means, method="pearson")
            dendrogram(Z, labels=group_labels)
            ```
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from scipy.stats import rankdata

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    X = np.asarray(X, dtype=float)
    row_var = X.var(axis=1)
    if np.any(row_var == 0):
        bad_rows = np.where(row_var == 0)[0].tolist()
        raise ValueError(
            f"Rows {bad_rows} have zero variance; correlation is undefined"
        )

    if method == "spearman":
        X = np.apply_along_axis(rankdata, axis=1, arr=X)

    # pdist 'correlation' metric computes 1 - Pearson r directly
    condensed_dist = pdist(X, metric="correlation")
    Z = linkage(
        condensed_dist, method=linkage_method, optimal_ordering=optimal_ordering
    )
    return Z, condensed_dist
