from matplotlib.pylab import f
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import variation, ttest_ind, mannwhitneyu, wilcoxon, spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import nan_euclidean_distances
from scpviz import utils
from scpviz.utils import format_log_prefix
from scpviz.utils.stats import bh_adjust_pvalues
import warnings
from scipy import sparse
import gseapy as gp


class AnalysisMixin:
    """
    Provides core statistical and dimensionality reduction tools for analyzing single-cell proteomics data.

    This mixin includes functionality for:

    - Differential expression (DE) analysis using t-tests, Mann–Whitney U, or Wilcoxon signed-rank tests  
    - Ranking proteins or peptides by abundance within groups  
    - Coefficient of Variation (CV) computation  
    - Missing value imputation (global or group-wise) using statistical or KNN-based methods  
    - Dimensionality reduction and clustering using PCA, UMAP, and Leiden  
    - Neighbor graph construction for downstream manifold learning  
    - Cleaning `.X` matrices by replacing NaNs  
    - Row-wise normalization across multiple strategies  

    All functions are compatible with both protein- and peptide-level data and support use of AnnData layers.

    Functions:
        cv: Compute coefficient of variation (CV) for each feature across or within sample groups.
        de: Perform differential expression analysis between two sample groups.
        rank: Rank features by mean abundance, compute standard deviation and numeric rank.
        impute: Impute missing values globally or within groups using mean, median, min, or KNN.
        neighbor: Compute neighborhood graph using PCA (or another embedding) for clustering or UMAP.
        leiden: Run Leiden clustering on neighborhood graph, storing labels in `.obs['leiden']`.
        umap: Perform UMAP dimensionality reduction using previously computed neighbors.
        pca: Run PCA on normalized expression matrix, handling NaN exclusion and reinsertion of features.
        clean_X: Replace NaNs in `.X` or a specified layer, optionally backing up the original.
        _normalize_helper: Internal helper to compute per-sample scaling across multiple normalization methods.
        log_transform: Apply log transform with pseudocount; register layer provenance.
    """

    def cv(self, classes = None, on = 'protein', layer = "X", debug = False):
        """
        Compute the coefficient of variation (CV) for each feature across sample groups.

        This method calculates CV for each protein or peptide across all samples in each group,
        storing the result as new columns in `.var`, one per group.

        Args:
            classes (str or list of str, optional): Sample-level class or list of classes used to define groups.
            on (str): Whether to compute CV on "protein" or "peptide" data.
            layer (str): Data layer to use for computation (default is "X").
            debug (bool): If True, prints debug information while filtering groups.

        Returns:
            None

        Example:
            Compute per-group CV for proteins using a custom normalization layer:
                ```python
                pdata.cv(classes=["group", "condition"], on="protein", layer="X_norm")
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)

        for j, class_value in enumerate(classes_list):
            data_filtered = utils.resolve_class_filter(adata, classes, class_value)

            cv_data = data_filtered.X.toarray() if layer == "X" else data_filtered.layers[layer].toarray() if layer in data_filtered.layers else None
            if cv_data is None:
                raise ValueError(f"Layer '{layer}' not found in adata.layers.")

            adata.var['CV: '+ class_value] = variation(cv_data, axis=0)

        self._history.append(f"{on}: Coefficient of Variation (CV) calculated for {layer} data by {classes}. CV stored in var['CV: {class_value}'].") # type: ignore[attr-defined]

    def de(self, values=None, class_type=None, method='ttest', layer='X', threshold=0.05, log2fc=1.0, fold_change_mode='mean', correct_fdr=False, equal_var=True, pval=None):
        """
        Perform differential expression (DE) analysis on proteins across sample groups.

        This method compares protein abundance between two sample groups using a specified
        statistical test and fold change method. Input groups can be defined using either
        legacy-style (`class_type` + `values`) or dictionary-style filters.

        Args:
            values (list of dict or list of list): Sample group filters to compare.

                - Dictionary-style (recommended): [{'cellline': 'HCT116', 'treatment': 'DMSO'}, {...}]
                - Legacy-style (if `class_type` is provided): [['HCT116', 'DMSO'], ['HCT116', 'DrugX']]

            class_type (str or list of str, optional): Legacy-style class label(s) to interpret `values`.
            method (str): Statistical test to use. Options: "ttest", "mannwhitneyu", "wilcoxon".
            layer (str): Name of the data layer to use (default is "X").
            threshold (float): Significance cutoff. Applied to raw ``p_value`` when
                ``correct_fdr=False``, and to ``adj_p_value`` when ``correct_fdr=True``. Defaults to 0.05.
            pval (float, optional): Deprecated alias for ``threshold``.

            log2fc (float): Minimum log2 fold change threshold for significance labeling.

            correct_fdr (bool): If True, apply Benjamini-Hochberg FDR correction and
                label significance using adjusted p-values. Default is False.

            equal_var (bool): Passed to :func:`scipy.stats.ttest_ind` when ``method='ttest'``.
                ``True`` (default) uses Student's t-test; ``False`` uses Welch's t-test.

            fold_change_mode (str): Strategy for computing fold change. Options:

                - "mean": log2(mean(group1) / mean(group2))
                - "pairwise_median": median of all pairwise log2 ratios
                - "pep_pairwise_median": median of peptide-level pairwise log2 ratios, aggregated per protein

        Returns:
            pd.DataFrame: DataFrame with DE statistics including log2 fold change, p-values, and significance labels.

        Example:
            Legacy-style DE comparison using class types and value combinations:
                ```python
                pdata.de(
                    class_type=["cellline", "treatment"],
                    values=[["HCT116", "DMSO"], ["HCT116", "DrugX"]]
                )
                ```

            Dictionary-style (recommended) DE comparison:
                ```python
                pdata.de(
                    values=[
                        {"cellline": "HCT116", "treatment": "DMSO"},
                        {"cellline": "HCT116", "treatment": "DrugX"}
                    ]
                )
                ```

            DE comparison adjusted for multiple testing with BH correction, and with unequal sample size:
                ```python
                pdata.de(
                    values=[
                        {"cellline": "HCT116", "treatment": "DMSO"},
                        {"cellline": "HCT116", "treatment": "DrugX"}
                    ],
                    correct_fdr=True,
                    equal_var=False
                )
                ```
        """

        if pval is not None:
            print(
                f"{format_log_prefix('warn')} `pval` is deprecated in de(); "
                f"use `threshold` instead (applied pval={pval})."
            )
            if pval != threshold:
                print(
                    f"{format_log_prefix('warn')} Both `threshold`={threshold} and "
                    f"`pval={pval}` were passed to de(); using `pval`."
                )
            threshold = pval

        # --- Handle legacy input ---
        if values is None:
            raise ValueError("Please provide `values` (new format) or both `class_type` and `values` (legacy format).")

        if class_type is not None:
            values = utils.format_class_filter(class_type, values, exact_cases=True)

        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("`values` must be a list of two group dictionaries (or legacy value pairs).")
                
        if values[0] == values[1]:
            raise ValueError("Both groups in `values` refer to the same condition. Please provide two distinct groups.")

        group1_dict, group2_dict = (
            [values[0]] if not isinstance(values[0], list) else values[0],
            [values[1]] if not isinstance(values[1], list) else values[1]
        )


        # --- Sample filtering ---
        pdata_case1 = self._filter_sample_values(values=group1_dict, exact_cases=True, return_copy=True, verbose=False, cleanup=False) # type: ignore[attr-defined], FilteringMixin
        pdata_case2 = self._filter_sample_values(values=group2_dict, exact_cases=True, return_copy=True, verbose=False, cleanup=False) # type: ignore[attr-defined], FilteringMixin

        def _label(d):
            if isinstance(d, dict):
                return '_'.join(str(v) for v in d.values())
            return str(d)

        group1_string = _label(group1_dict)
        group2_string = _label(group2_dict)
        comparison_string = f'{group1_string} vs {group2_string}'

        # --- Get layer data ---
        data1 = utils.get_adata_layer(pdata_case1.prot, layer)
        data2 = utils.get_adata_layer(pdata_case2.prot, layer)

        # Shape: (samples, features)
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        _layer_is_log = utils.infer_layer_is_log(layer, pdata_case1.prot)
        _finite_de = data1[np.isfinite(data1)]
        _median_val = float(np.nanmedian(_finite_de)) if _finite_de.size > 0 else 0.0

        if not _layer_is_log and _median_val > 1e4:
            print(
                f"{format_log_prefix('warn')} Layer {layer!r} appears to contain "
                f"non-log intensities (median={_median_val:.2e}). Statistical tests "
                "assume approximately normal distributions — consider log_transform() "
                "before DE analysis."
            )

        log_prefix = format_log_prefix("user")
        n1, n2 = len(pdata_case1.prot), len(pdata_case2.prot)
        print(f"{log_prefix} Running differential expression [protein]")
        print(f"   🔸 Comparing groups: {comparison_string}")
        print(f"   🔸 Group sizes: {n1} vs {n2} samples")
        print(
            f"   🔸 Layer: {layer} "
            f"{'(log-transformed)' if _layer_is_log else '(non-log)'}"
        )
        print(f"   🔸 Method: {method} | Fold Change: {fold_change_mode}")
        if correct_fdr:
            print(f"   🔸 Adj p-value threshold: {threshold} | Log2FC threshold: {log2fc}")
        else:
            print(f"   🔸 P-value threshold: {threshold} | Log2FC threshold: {log2fc}")

        # --- Compute fold change ---
        if fold_change_mode == 'mean':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                
                group1_mean = np.nanmean(data1, axis=0)
                group2_mean = np.nanmean(data2, axis=0)

                # Identify zeros or NaNs in either group
                mask_invalid = (group1_mean == 0) | (group2_mean == 0) | np.isnan(group1_mean) | np.isnan(group2_mean)

                if _layer_is_log:
                    rec = pdata_case1.prot.uns.get("layer_provenance", {}).get(layer, {})
                    _base_val = str(rec.get("base", "2"))
                    if _base_val in ("10",):
                        log2fc_vals = (group1_mean - group2_mean) / np.log10(2)
                        _base_label = "log10"
                    elif _base_val in ("e",):
                        log2fc_vals = (group1_mean - group2_mean) / np.log(2)
                        _base_label = "loge"
                    else:
                        log2fc_vals = group1_mean - group2_mean
                        _base_label = "log2"
                    log2fc_vals[mask_invalid] = np.nan
                    print(
                        f"{format_log_prefix('info')} Layer {layer!r} is log-transformed "
                        f"({_base_label}) — using mean difference for log2FC."
                    )
                else:
                    log2fc_vals = np.log2(group1_mean / group2_mean)
                    log2fc_vals[mask_invalid] = np.nan

                n_invalid = np.sum(mask_invalid)
                if n_invalid > 0:
                    print(f"{format_log_prefix('info',2)} {n_invalid} proteins were not comparable (zero or NaN mean in one group).")

        elif fold_change_mode == 'pairwise_median':
            mask_invalid = ( # Detect invalid features (any 0 or NaN in either group)
                np.any((data1 == 0) | np.isnan(data1), axis=0) |
                np.any((data2 == 0) | np.isnan(data2), axis=0)
            )
            # Compute median pairwise log2FC
            log2fc_vals = utils.pairwise_log2fc(data1, data2)
            log2fc_vals[mask_invalid] = np.nan # Mark invalid features as NaN
            n_invalid = np.sum(mask_invalid)
            if n_invalid > 0:
                print(f"{format_log_prefix('info',2)} {n_invalid} proteins were not comparable (zero or NaN mean in one group).")
        
        elif fold_change_mode == 'pep_pairwise_median':
            # --- Validate .pep presence ---
            if self.pep is None:
                raise ValueError("Peptide-level data (.pep) is required for fold_change_mode='pep_pairwise_median', but self.pep is None.")

            # --- Handle peptide layer fallback ---
            actual_layer = layer
            if layer != 'X' and not (hasattr(self.pep, "layers") and layer in self.pep.layers):
                warnings.warn(
                    f"Layer '{layer}' not found in .pep.layers. Falling back to 'X'.",
                    UserWarning
                )
                actual_layer = 'X'

            # Get peptide data
            pep_data1 = np.asarray(utils.get_adata_layer(pdata_case1.pep, actual_layer))
            pep_data2 = np.asarray(utils.get_adata_layer(pdata_case2.pep, actual_layer))

            # Detect invalid peptides (any 0 or NaN in either group)
            mask_invalid_pep = (
                np.any((pep_data1 == 0) | np.isnan(pep_data1), axis=0) |
                np.any((pep_data2 == 0) | np.isnan(pep_data2), axis=0)
            )

            # Compute per-peptide pairwise log2FCs
            pep_log2fc = utils.pairwise_log2fc(pep_data1, pep_data2)
            pep_log2fc[mask_invalid_pep] = np.nan  # mark invalids

            n_invalid_pep = np.sum(mask_invalid_pep)
            if n_invalid_pep > 0:
                print(f"{format_log_prefix('info',2)} {n_invalid_pep} peptides were not comparable (zero or NaN mean in one group).")

            # Map peptides to proteins
            pep_to_prot = utils.get_pep_prot_mapping(self, return_series=True)

            # Aggregate peptide log2FCs into protein-level log2FCs
            prot_log2fc = pd.Series(index=self.prot.var_names, dtype=float)
            not_comparable_prot = []

            for prot in self.prot.var_names:
                matching_peptides = pep_to_prot[pep_to_prot == prot].index
                if len(matching_peptides) == 0:
                    continue

                idxs = self.pep.var_names.get_indexer(matching_peptides)
                valid_idxs = idxs[idxs >= 0]
                if len(valid_idxs) == 0:
                    continue

                valid_log2fc = pep_log2fc[valid_idxs]

                if np.all(np.isnan(valid_log2fc)):
                    prot_log2fc[prot] = np.nan
                    not_comparable_prot.append(prot)
                else:
                    prot_log2fc[prot] = np.nanmedian(pep_log2fc[valid_idxs])

            log2fc_vals = prot_log2fc.values
            if len(not_comparable_prot) > 0:
                print(f"{format_log_prefix('info',2)} {len(not_comparable_prot)} proteins were not comparable (all peptides invalid or missing).")

        else:
            raise ValueError(f"Unsupported fold_change_mode: {fold_change_mode}")

        # --- Statistical test ---
        pvals = []
        stats = []
        for i in range(data1.shape[1]):
            x1, x2 = data1[:, i], data2[:, i]
            try:
                if method == 'ttest':
                    res = ttest_ind(x1, x2, equal_var=equal_var, nan_policy='omit')
                elif method == 'mannwhitneyu':
                    res = mannwhitneyu(x1, x2, alternative='two-sided')
                elif method == 'wilcoxon':
                    res = wilcoxon(x1, x2)
                else:
                    raise ValueError(f"Unsupported test method: {method}")
                pvals.append(res.pvalue)
                stats.append(res.statistic)
            except Exception as e:
                pvals.append(np.nan)
                stats.append(np.nan)

        # --- Compile results ---
        var = self.prot.var.copy()
        df_stats = pd.DataFrame(index=self.prot.var_names)
        df_stats['Genes'] = var['Genes'] if 'Genes' in var.columns else var.index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df_stats[group1_string] = np.nanmean(data1, axis=0)
            df_stats[group2_string] = np.nanmean(data2, axis=0)
        df_stats['log2fc'] = log2fc_vals
        df_stats['p_value'] = pvals
        df_stats['test_statistic'] = stats

        df_stats['-log10(p_value)'] = -np.log10(
            df_stats['p_value'].replace(0, np.nan).astype(float)
        )
        df_stats['significance_score'] = df_stats['-log10(p_value)'] * df_stats['log2fc']

        if correct_fdr:
            df_stats['adj_p_value'] = bh_adjust_pvalues(np.asarray(pvals, dtype=float))
            df_stats['-log10(adj_p_value)'] = -np.log10(
                df_stats['adj_p_value'].replace(0, np.nan).astype(float)
            )
            p_for_sig = df_stats['adj_p_value']
        else:
            p_for_sig = df_stats['p_value']

        df_stats['significance'] = 'not significant'
        mask_not_comparable = df_stats['log2fc'].isna()
        df_stats.loc[mask_not_comparable, 'significance'] = 'not comparable'
        df_stats.loc[(p_for_sig < threshold) & (df_stats['log2fc'] > log2fc), 'significance'] = 'upregulated'
        df_stats.loc[(p_for_sig < threshold) & (df_stats['log2fc'] < -log2fc), 'significance'] = 'downregulated'
        df_stats['significance'] = pd.Categorical(df_stats['significance'], categories=['upregulated', 'downregulated', 'not significant', 'not comparable'], ordered=True)

        df_stats = df_stats.sort_values(by='significance')

        # --- Store and return ---
        self._stats[comparison_string] = df_stats # type: ignore[attr-defined]
        self._append_history(f"prot: DE for {class_type} {values} using {method} and fold_change_mode='{fold_change_mode}'. Stored in .stats['{comparison_string}'].") # type: ignore[attr-defined], HistoryMixin

        sig_counts = df_stats['significance'].value_counts().to_dict()
        n_up = sig_counts.get('upregulated', 0)
        n_down = sig_counts.get('downregulated', 0)
        n_ns = sig_counts.get('not significant', 0)

        print(f"{format_log_prefix('result_only', indent=2)} DE complete. Results stored in:")
        print(f'       • .stats["{comparison_string}"]')
        col_hint = "log2fc, p_value, adj_p_value, significance, etc." if correct_fdr else "log2fc, p_value, significance, etc."
        print(f"       • Columns: {col_hint}")
        print(f"       • Upregulated: {n_up} | Downregulated: {n_down} | Not significant: {n_ns}")

        return df_stats

    # TODO: Need to figure out how to make this interface with plot functions, probably do reordering by each class_value within the loop?
    def rank(self, classes = None, on = 'protein', layer = "X"):
        """
        Rank proteins or peptides by average abundance across sample groups.

        This method computes the average and standard deviation for each feature within 
        each group and assigns a rank (highest to lowest) based on the group-level mean.
        The results are stored in `.var` with one set of columns per group.

        Args:
            classes (str or list of str, optional): Sample-level class/grouping column(s) in `.obs`.
            on (str): Whether to compute ranks on "protein" or "peptide" data.
            layer (str): Name of the data layer to use (default is "X").

        Returns:
            None

        Example:
            Rank proteins by average abundance across treatment groups:
                ```python
                pdata.rank(classes="treatment", on="protein", layer="X_norm")
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)
        
        for class_value in classes_list:
            rank_data = utils.resolve_class_filter(adata, classes, class_value)
            if layer == "X":
                layer_data = rank_data.X.toarray()
            elif layer in rank_data.layers:
                layer_data = rank_data.layers[layer].toarray()
            else:
                raise ValueError(f"Layer '{layer}' not found in layers.")

            # Convert sparse to dense if needed
            if hasattr(layer_data, 'toarray'):
                layer_data = layer_data.toarray()

            # Transpose to get DataFrame of shape (features, samples)
            rank_df = pd.DataFrame(layer_data.T, index=rank_data.var.index, columns=rank_data.obs_names)

            # Compute stats
            avg_col = f"Average: {class_value}"
            std_col = f"Stdev: {class_value}"
            rank_col = f"Rank: {class_value}"

            with np.errstate(invalid='ignore', divide='ignore'):
                rank_df[avg_col] = np.nanmean(layer_data, axis=0)
                rank_df[std_col] = np.nanstd(layer_data, axis=0)

            # Sort by average (descending), assign rank
            rank_df.sort_values(by=avg_col, ascending=False, inplace=True)
            rank_df[rank_col] = np.where(rank_df[avg_col].isna(), np.nan, np.arange(1, len(rank_df) + 1))

            # Reindex back to original order in adata.var
            rank_df = rank_df.reindex(adata.var.index)

            adata.var[avg_col] = rank_df[avg_col]
            adata.var[std_col] = rank_df[std_col]
            adata.var[rank_col] = rank_df[rank_col]

        self._history.append(f"{on}: Ranked {layer} data. Ranking, average and stdev stored in var.") # type: ignore[attr-defined], HistoryMixin

    def impute(self, classes=None, layer="X", method='mean', on='protein', min_scale=1, set_X=True, use_zeros_as_nan=False, **kwargs):
        """
        Impute missing values across samples globally or within groups.

        This method imputes missing values in the specified data layer using one of several strategies.
        It supports both global (across all samples) and group-wise imputation based on sample classes.

        Args:
            classes (str or list of str, optional): Sample-level class/grouping column(s). If None, imputation is global.
            layer (str): Data layer to impute from (default is "X").
            method (str): Imputation strategy to use. Options include:

                - "mean": Fill missing values with the mean of each feature.
                - "median": Fill missing values with the median of each feature.
                - "min": Fill with the minimum observed value (0 if all missing).
                - "knn": Use K-nearest neighbors (only supported for global imputation).
                - "pimms_dae": PIMMS Denoising Autoencoder (global only).
                - "pimms_vae": PIMMS Variational Autoencoder (global only).
                - "pimms_cf":  PIMMS Collaborative Filtering (global only).
                
            on (str): Whether to impute "protein" or "peptide" data.
            min_scale (float): Scaled multiplication of minimum value for imputation, i.e. 0.2 would be 20% of minimum value (default is 1).
            set_X (bool): If True, updates `.X` to use the imputed result.
            use_zeros_as_nan: If True, treat 0 values as NaN before imputing. Mostly used after `directlfq` normalization, which returns NaNs as 0s. Defaults to False.
            **kwargs: Additional arguments passed to the imputer (e.g., `n_neighbors` for KNN).

        Returns:
            None

        Example:
            Globally impute missing values using the median strategy:
                ```python
                pdata.impute(method="median", on="protein")
                ```

            Group-wise imputation based on treatment:
                ```python
                pdata.impute(classes="treatment", method="mean", on="protein")
                ```

        Note:
            - KNN imputation is only supported for global (non-grouped) mode.
            - If `directlfq` was used for normalization, set `use_zeros_as_nan` flag to `True`. Else, no imputation will be performed as `directlfq` only returns 0s.
            - Features that are entirely missing within a group or across all samples are skipped and preserved as NaN.
            - Imputed results are stored in a new layer named `"X_impute_<method>"`.
            - Imputation summaries are printed to the console by group or overall.
            - PIMMS, which stands for **Proteomics Imputation Modeling Mass Spectrometry**, for more information see: the [package](https://github.com/RasmussenLab/pimms) or the [manuscript](https://www.nature.com/articles/s41467-024-48711-5#Fig1).
        """
        from sklearn.impute import SimpleImputer, KNNImputer
        from scipy import sparse
        from scpviz import utils


        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in .{on}.")

        _resolved_input = utils.resolve_input_layer(adata, layer)

        impute_data = adata.layers[layer] if layer != "X" else adata.X
        was_sparse = sparse.issparse(impute_data)
        impute_data = impute_data.toarray() if was_sparse else impute_data.copy()
        original_data = impute_data.copy()

        if use_zeros_as_nan:
            impute_data = impute_data.astype(float)
            zero_mask = (impute_data == 0)
            impute_data[zero_mask] = np.nan
            original_data[zero_mask] = np.nan
            nan_message = f"{format_log_prefix('info_only',2)} Using zeros as nans for imputation. Recommended after `directlfq` normalization."
        else:
            nan_message = None

        layer_name = f"X_impute_{method}"

        if method not in {"mean", "median", "min","knn","pimms_dae","pimms_vae","pimms_cf"}:
            raise ValueError(f"Unsupported method: {method}")
        
        if classes is None:
            # Global imputation
            if method == 'min':
                min_vals = np.nanmin(impute_data, axis=0)
                min_vals = np.where(np.isnan(min_vals), 0, min_vals)
                min_vals = min_vals * min_scale
                mask = np.isnan(impute_data)
                impute_data[mask] = np.take(min_vals, np.where(mask)[1])
            elif method == 'knn':
                n_neighbors = kwargs.get('n_neighbors', 3)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                impute_data = imputer.fit_transform(impute_data)
            elif method in {"pimms_dae", "pimms_vae", "pimms_cf"}:
                try:
                    from pimmslearn.sklearn.ae_transformer import AETransformer
                    from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
                except ImportError:
                    raise ImportError(
                        "The 'pimmslearn' package is required for PIMMS imputation.\n"
                        "Install with: pip install pimms-learn"
                    )

                df = pd.DataFrame(
                    impute_data,
                    index=adata.obs_names,
                    columns=adata.var_names
                )

                df = np.log2(df + 1)

                # --- AE (DAE / VAE) -----------------------------------------
                if method in {"pimms_dae", "pimms_vae"}:
                    model_type = "DAE" if method == "pimms_dae" else "VAE"
                    model = AETransformer(
                        model=model_type,
                        hidden_layers=kwargs.get("hidden_layers", [512]),
                        latent_dim=kwargs.get("latent_dim", 50),
                        batch_size=kwargs.get("batch_size", 10),
                    )

                    model.fit(df,
                            cuda=kwargs.get("cuda", False),
                            epochs_max=kwargs.get("epochs_max", 100))
                    df_imputed = model.transform(df)

                else:  # pimms_cf (collaborative filtering)
                    index_name = "Sample ID"
                    column_name = "protein group"
                    value_name = "intensity"

                    df.index.name = index_name       # "Sample ID"
                    df.columns.name = column_name    # "protein group"

                    series = df.stack(dropna=False)
                    series.name = value_name

                    model = CollaborativeFilteringTransformer(
                        target_column=value_name,
                        sample_column=index_name,
                        item_column=column_name,
                        n_factors=kwargs.get("n_factors", 30),
                        batch_size=kwargs.get("batch_size", 4096),
                    )

                    model.fit(series,
                            cuda=kwargs.get("cuda", False),
                            epochs_max=kwargs.get("epochs_max", 20))
                    df_imputed = model.transform(series).unstack()

                # Convert result back
                df_imputed = (2 ** df_imputed) - 1
                impute_data = df_imputed.to_numpy()

                # Store metadata
                self.stats.setdefault("imputation", {})  # type: ignore[attr-defined]
                self.stats["imputation"]["pimms"] = {
                    "method": method,
                    "params": kwargs,
                    "samples": df.shape[0],
                    "features": df.shape[1],
                }

            else: # mean or median
                imputer = SimpleImputer(strategy=method, keep_empty_features=True)
                nan_columns = np.isnan(impute_data).all(axis=0)  # features fully missing in this group
                impute_data = imputer.fit_transform(impute_data)
                impute_data[:, nan_columns] = np.nan

            min_message = "" if method != 'min' else f"Minimum scaled by {min_scale}."
            print(f"{format_log_prefix('user')} Global imputation using '{method}'. Layer saved as '{layer_name}'. {min_message}")
            if nan_message is not None:
                print(nan_message)
            skipped_features = np.sum(np.isnan(impute_data).all(axis=0))

        else:
            # Group-wise imputation
            if method == 'knn':
                raise ValueError("KNN imputation is not supported for group-wise imputation.")
            if method.startswith("pimms"):
                raise ValueError("PIMMS imputation is only supported for global (classes=None) mode.")

            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            for group in unique_groups:
                idx = np.where(sample_names == group)[0]
                group_data = impute_data[idx, :]

                if method == 'min':
                    min_vals = np.nanmin(group_data, axis=0)
                    min_vals = np.where(np.isnan(min_vals), 0, min_vals)
                    min_vals = min_vals * min_scale
                    mask = np.isnan(group_data)
                    group_data[mask] = np.take(min_vals, np.where(mask)[1])
                    imputed_group = group_data
                else:
                    imputer = SimpleImputer(strategy=method, keep_empty_features=True)
                    nan_columns = np.isnan(group_data).all(axis=0)  # features fully missing in this group
                    imputed_group = imputer.fit_transform(group_data)
                    imputed_group[:, nan_columns] = np.nan # restore fully missing features

                impute_data[idx, :] = imputed_group

            min_message = "" if method != 'min' else f"Minimum scaled by {min_scale}."
            print(f"{format_log_prefix('user')} Group-wise imputation using '{method}' on class(es): {classes}. Layer saved as '{layer_name}'. {min_message}")

        summary_lines = []
        if classes is None:
            num_imputed = np.sum(np.isnan(original_data) & ~np.isnan(impute_data))
            # Row-wise missingness
            was_missing = np.isnan(original_data).any(axis=1)
            now_complete = ~np.isnan(impute_data).any(axis=1)
            now_incomplete = np.isnan(impute_data).any(axis=1)

            fully_imputed_samples = np.sum(was_missing & now_complete)
            partially_imputed_samples = np.sum(was_missing & now_incomplete)
            skipped_features = np.sum(np.isnan(impute_data).all(axis=0))

            summary_lines.append(
                f"{format_log_prefix('result_only', indent=2)} {num_imputed} values imputed."
            )
            summary_lines.append(
                f"{format_log_prefix('info_only', indent=2)} {fully_imputed_samples} samples fully imputed, {partially_imputed_samples} samples partially imputed, {skipped_features} skipped feature(s) with all missing values."
            )

        else:
            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            counts_by_group = {}
            fully_by_group = {}
            partial_by_group = {}
            missing_features_by_group = {}
            total_samples_by_group = {}
            
            for group in unique_groups:
                idx = np.where(sample_names == group)[0]
                before = original_data[idx, :]
                after = impute_data[idx, :]

                # count imputed values
                mask = np.isnan(before) & ~np.isnan(after)
                counts_by_group[group] = np.sum(mask)

                # count fully and partially imputed samples
                was_missing = np.isnan(before).any(axis=1)
                now_complete = ~np.isnan(after).any(axis=1)
                now_incomplete = np.isnan(after).any(axis=1)
                now_missing = np.sum(np.isnan(before).all(axis=0))

                fully_by_group[group] = np.sum(was_missing & now_complete)
                partial_by_group[group] = np.sum(was_missing & now_incomplete)
                missing_features_by_group[group] = now_missing
                total_samples_by_group[group] = len(idx)

            # Compute dynamic width based on longest group name
            group_width = max(max(len(str(g)) for g in unique_groups), 20)

            # Summary totals
            total = sum(counts_by_group.values())
            summary_lines.append(f"{format_log_prefix('result_only', indent=2)} {total} values imputed total.")
            summary_lines.append(f"{format_log_prefix('info_only', indent=2)} Group-wise summary:")

            # Header row (aligned with computed width)
            header = (f"{'Group':<{group_width}} | Values Imputed | Skipped Features | Samples Imputed (Partial,Fully)/Total")
            divider = "-" * len(header)
            summary_lines.append(f"{' ' * 5}{header}")
            summary_lines.append(f"{' ' * 5}{divider}")

            # Data rows
            for group in unique_groups:
                count = counts_by_group[group]
                fully = fully_by_group[group]
                partial = partial_by_group[group]
                skipped = missing_features_by_group[group]
                total_samples = total_samples_by_group[group]
                summary_lines.append(
                    f"{' ' * 5}{group:<{group_width}} | {count:>14} | {skipped:>16} | {partial:>7}, {fully:>5} / {total_samples:<3}"
                )

        print("\n".join(summary_lines))

        actual_layer_name = utils.update_layer_provenance(
            adata,
            layer_name=layer_name,
            op="impute",
            input_layer=_resolved_input,
            method=method,
        )
        adata.layers[actual_layer_name] = (
            sparse.csr_matrix(impute_data) if was_sparse else impute_data
        )

        if set_X:
            self.set_X(layer=actual_layer_name, on=on)  # type: ignore[attr-defined], EditingMixin

        self._history.append(  # type: ignore[attr-defined]
            f"{on}: Imputed layer '{layer}' using '{method}' (grouped by {classes if classes else 'ALL'}). "
            f"Stored in '{actual_layer_name}'."
        )

    def log_transform(
        self,
        on: str = "protein",
        layer: str = "X",
        base: int | float | str = 2,
        pseudocount: float = 1.0,
        set_X: bool = True,
    ) -> None:
        """
        Log-transform intensities with a pseudocount and register layer provenance.

        Output is stored under a short fixed name: ``X_log2``, ``X_log10``, or ``X_loge``.
        If that name already exists with a different origin, a numeric suffix is applied
        (see :func:`scpviz.utils.update_layer_provenance`).

        Args:
            on: ``\"protein\"`` / ``\"prot\"`` or ``\"peptide\"`` / ``\"pep\"``.
            layer: Source layer (default ``\"X\"``).
            base: ``2``, ``10``, ``\"e\"``, or ``numpy.e``.
            pseudocount: Added before the logarithm (default ``1.0``).
            set_X: If True, point ``.X`` at the new log layer.
        """
        if not self._check_data(on):  # type: ignore[attr-defined], ValidationMixin
            return

        adata = utils.get_adata(self, on)
        SUPPORTED_BASES = {2: np.log2, 10: np.log10, "e": np.log, np.e: np.log}
        if base not in SUPPORTED_BASES:
            raise ValueError(
                f"{format_log_prefix('error')} base={base!r} not supported. "
                "Choose from: 2, 10, 'e', or numpy.e."
            )
        log_fn = SUPPORTED_BASES[base]

        base_str = (
            "e"
            if base in ("e", np.e)
            else (str(int(base)) if float(base) == int(float(base)) else str(base))
        )
        layer_name = f"X_log{base_str}"

        if layer == "X":
            raw_mx = adata.X
            data = raw_mx.toarray() if sparse.issparse(raw_mx) else raw_mx.copy()
        elif layer in adata.layers:
            raw_mx = adata.layers[layer]
            data = raw_mx.toarray() if sparse.issparse(raw_mx) else np.asarray(raw_mx)
        else:
            raise KeyError(
                f"{format_log_prefix('error')} Layer {layer!r} not found in adata.layers."
            )

        _already_log = utils.infer_layer_is_log(layer, adata)
        if _already_log:
            print(
                f"{format_log_prefix('warn')} Layer {layer!r} appears to already be "
                "log-transformed (provenance registry or layer name). "
                f"Proceeding — result stored under the resolved output layer name."
            )

        n_negative = int(np.sum(data < 0))
        if n_negative > 0:
            print(
                f"{format_log_prefix('warn')} {n_negative} value(s) < 0 detected. "
                f"Pseudocount ({pseudocount}) added — negative values become "
                f"log({pseudocount}). Consider checking your normalization."
            )

        data_log = log_fn(data + pseudocount)

        was_sparse = sparse.issparse(
            adata.layers[layer] if layer != "X" else adata.X
        )
        _resolved_input = utils.resolve_input_layer(adata, layer)
        actual_layer_name = utils.update_layer_provenance(
            adata,
            layer_name=layer_name,
            op="log_transform",
            input_layer=_resolved_input,
            base=base_str,
            pseudocount=pseudocount,
        )
        adata.layers[actual_layer_name] = (
            sparse.csr_matrix(data_log) if was_sparse else data_log
        )

        on_norm = "protein" if on in ("protein", "prot") else "peptide"
        subpdata = "prot" if on_norm == "protein" else "pep"
        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Log-transforming [{on_norm}] layer: {layer!r} → {actual_layer_name!r}")
        print(f"     🔸 base: {base_str} log base (pseudocount {pseudocount})")
        print(f"{format_log_prefix('result_only', indent=2)} Log transform complete. Results stored in:")
        print(f"       • .{subpdata}.layers[{actual_layer_name!r}]")
        if set_X:
            print(f"       • .{subpdata}.X updated")
        print(f"       • .{subpdata}.uns['layer_provenance'][{actual_layer_name!r}] updated")
        
        if set_X:
            self.set_X(layer=actual_layer_name, on=on)  # type: ignore[attr-defined]

        self._append_history(  # type: ignore[attr-defined]
            f"{on_norm}: log_transform(base={base}, pseudocount={pseudocount}) on layer={layer!r}. "
            f"Stored in {actual_layer_name!r}."
        )

    def neighbor(self, on = 'protein', layer = "X", use_rep='X_pca', user_indent=0, **kwargs):
        """
        Compute a neighbor graph based on protein or peptide data.

        This method builds a nearest-neighbors graph for downstream analysis using 
        `scanpy.pp.neighbors`. It optionally performs PCA before constructing the graph 
        if a valid representation is not already available.

        Args:
            on (str): Whether to use "protein" or "peptide" data.
            layer (str): Data layer to use (default is "X").
            use_rep (str): Key in `.obsm` to use for computing neighbors. Default is `"X_pca"`.
                If `"X_pca"` is requested, PCA will be run automatically. If an alternative rep is provided, PCA will not be re-run.
            **kwargs: Additional keyword arguments passed to `scanpy.pp.neighbors()`.

        Returns:
            None

        Example:
            Compute neighbors using default PCA representation:
                ```python
                pdata.neighbor(on="protein", layer="X")
                ```

            Use a custom representation stored in `.obsm["X_umap"]`:
                ```python
                pdata.neighbor(on="protein", use_rep="X_umap", n_neighbors=15)
                ```

        Note:
            - The neighbor graph is stored in `.obs["distances"]` and `.obs["connectivities"]`.
            - Neighbor metadata is stored in `.uns["neighbors"]`.
            - Automatically calls `self.set_X()` if a non-default layer is specified.
            - PCA is computed automatically if `use_rep='X_pca'`, else neighbor will use the rep provided by the user.

        Todo:
            Allow users to supply a custom `KNeighborsTransformer` or precomputed neighbor graph.
                ```python
                from sklearn.neighbors import KNeighborsTransformer
                transformer = KNeighborsTransformer(n_neighbors=10, metric='manhattan', algorithm='kd_tree')
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
        
        if on.lower() in ["prot", "protein"]:
            adata = self.prot
        elif on.lower() in ["pep", "peptide"]:
            adata = self.pep

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        log_prefix = format_log_prefix("user") if user_indent == 0 else format_log_prefix("user_only",2)
        print(f"{log_prefix} Computing neighbors [{on}] using layer: {layer}")

        if use_rep == 'X_pca':
            print(f"{format_log_prefix('info_only',indent=2)} Recomputing PCA for neighbor graph.")
            self.pca(on=on, layer=layer)
        else:
            if use_rep not in adata.obsm:
                raise ValueError(f"PCA key '{use_rep}' not found in obsm. Please run PCA first and specify a valid key.")
            print(f"{format_log_prefix('info_only',indent=2)} Using '{use_rep}' found in obsm for neighbor graph.")

        if use_rep == 'X_pca':
            sc.pp.neighbors(adata, **kwargs)
        else:
            sc.pp.neighbors(adata, use_rep=use_rep, **kwargs)

        self._append_history(f'{on}: Neighbors fitted on {layer}, using {use_rep}, stored in obs["distances"] and obs["connectivities"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only',indent=2)} Neighbors computed on {layer}, using {use_rep}. Results stored in:")
        print(f"       • obs['distances'] (pairwise distances)")
        print(f"       • obs['connectivities'] (connectivity graph)")
        print(f"       • uns['neighbors'] (neighbor graph metadata)")
 
    def leiden(self, on = 'protein', layer = "X", **kwargs):
        """
        Perform Leiden clustering on protein or peptide data.

        This method runs community detection using the Leiden algorithm based on a precomputed
        neighbor graph using `scanpy.tl.leiden()`. If neighbors are not already computed, they will be generated automatically.

        Args:
            on (str): Whether to use "protein" or "peptide" data.
            layer (str): Data layer to use for clustering (default is "X").
            **kwargs: Additional keyword arguments passed to `scanpy.tl.leiden()`.

        Returns:
            None

        Example:
            Perform Leiden clustering using the default PCA-based neighbors:
                ```python
                pdata.leiden(on="protein", layer="X", resolution=0.25)
                ```

        Note:
            - Cluster labels are stored in `.obs["leiden"]`.
            - Neighbor graphs are automatically computed if not present in `.uns["neighbors"]`.
            - Automatically sets `.X` to the specified layer if it is not already active.
        """
        # uses sc.tl.leiden with default resolution of 0.25
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Performing Leiden clustering [{on}] using layer: {layer}")

        if 'resolution' in kwargs:
            resolution = kwargs.pop("resolution", 0.25)

        if 'neighbors' not in adata.uns:
            print(f"{format_log_prefix('info_only', indent=2)} Neighbors not found in AnnData object. Running neighbors with default settings.")
            self.neighbor(on = on, layer = layer, **kwargs)

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        sc.tl.leiden(adata, resolution)

        self._append_history(f'{on}: Leiden clustering fitted on {layer}, stored in obs["leiden"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only', indent=2)} Leiden clustering complete. Results stored in:")
        print(f"       • obs['leiden'] (cluster labels)")

    def umap(self, on = 'protein', layer = "X", force_neighbors=False, **kwargs):
        """
        Compute UMAP dimensionality reduction on protein or peptide data.

        This method runs UMAP (Uniform Manifold Approximation and Projection) on the selected data layer using `scanpy.tl.umap()`.
        If neighbor graphs are not already computed, they will be generated automatically.

        Args:
            on (str): Whether to use "protein" or "peptide" data.
            layer (str): Data layer to use for UMAP (default is "X").
            force_neighbors (bool): If True, recompute neighbors even if they exist.
            **kwargs: Additional keyword arguments passed to `scanpy.tl.umap()`, `scanpy.tl.neighbor()` or the scpviz `pca` function. If provided, neighbor() will always be recomputed.
                Example:
                    "n_neighbors": neighbor argument
                    "min_dist": umap argument
                    "metric": neighbor argument
                    "spread": umap argument
                    "random_state": umap argument
                    "n_pcs": neighbor argument

        Returns:
            None

        Example:
            Run UMAP using default settings:
                ```python
                pdata.umap(on="protein", layer="X")
                ```
        Note:
            - UMAP coordinates are stored in `.obsm["X_umap"]`.
            - UMAP settings are stored in `.uns["umap"]`.
            - Automatically computes neighbor graphs if not already available.
            - Will call `.set_X()` if a non-default layer is used.
        """
        # uses sc.tl.umap
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
       
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Computing UMAP [{on}] using layer: {layer}")

        if "n_neighbors" in kwargs or "metric" in kwargs or "n_pcs" in kwargs:
                    n_neighbors = kwargs.pop("n_neighbors", None)
                    metric = kwargs.pop("metric", None)
                    n_pcs = kwargs.pop("n_pcs", None)

                    # Prepare a readable message
                    neighbor_args = []
                    if n_neighbors is not None:
                        neighbor_args.append(f"n_neighbors={n_neighbors}")
                    else:
                        n_neighbors = 15  # default value
                    if metric is not None:
                        neighbor_args.append(f"metric='{metric}'")
                    else:
                        metric = "euclidean"  # default value
                    if n_pcs is not None:
                        neighbor_args.append(f"n_pcs={n_pcs}")
                    else:
                        n_pcs = 50
                    arg_str = ", ".join(neighbor_args)

                    print(f"{format_log_prefix('info_only', indent=2)} {arg_str} provided. "
                        f"Re-running neighbors with these settings before UMAP.")

                    self.neighbor(on=on, layer=layer, n_neighbors=n_neighbors, metric=metric, user_indent=2)
                    self._append_history(f"{on}: Neighbors re-computed with {arg_str} before UMAP")  # type: ignore[attr-defined], HistoryMixin
        else:
            # check if neighbor has been run before, look for distances and connectivities in obsp
            if force_neighbors:
                self.neighbor(on=on, layer=layer)
                self._append_history(f"{on}: Neighbors computed with default settings before UMAP")
            elif 'neighbors' not in adata.uns:
                print(f"{format_log_prefix('info_only', indent=2)} Neighbors not found in AnnData object. Running neighbors with default settings.")
                self.neighbor(on = on, layer = layer)
                self._append_history(f"{on}: Neighbors computed with default settings before UMAP")  # type: ignore[attr-defined], HistoryMixin
            else:
                print(f"{format_log_prefix('info_only', indent=2)} Using existing neighbors found in AnnData object.")

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        sc.tl.umap(adata, **kwargs)

        self._append_history(f'{on}: UMAP fitted on {layer}, stored in obsm["X_umap"] and uns["umap"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only', indent=2)} UMAP complete. Results stored in:")
        print(f"       • obsm['X_umap'] (UMAP coordinates)")
        print(f"       • uns['umap'] (UMAP settings)")

    def pca(self, on = 'protein', layer = "X", **kwargs):
        """
        Perform PCA (Principal Component Analysis) on protein or peptide data.

        This method performs PCA on the selected data layer, after z-score normalization and removal of
        NaN-containing features. The results are stored in `.obsm["X_pca"]` and `.uns["pca"]`.

        Args:
            on (str): Whether to use "protein" or "peptide" data.
            layer (str): Data layer to use for PCA (default is "X").
            **kwargs: Additional keyword arguments passed to `scanpy.tl.pca()`. For example,
                `key_added` to store PCA in a different key.

        Returns:
            None

        Note:
            - Features (columns) with NaN values are excluded before PCA and then padded with zeros.
            - PCA scores are stored in `.obsm['X_pca']`.
            - Principal component loadings, variance ratios, and total variances are stored in `.uns['pca']`.
            - If you store PCs under a custom key using `key_added`, remember to set `use_rep` when calling `.neighbor()` or `.umap()`.
        """

        # uses sc.tl.pca
        # for kwargs can use key_added to store PCA in a different key - then for neighbors need to specify key by use_rep
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
        
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        # make sample array
        if layer == "X":
            X = adata.X.toarray()
        elif layer in adata.layers.keys():
            X = adata.layers[layer].toarray()

        _layer_is_log_pca = utils.infer_layer_is_log(layer, adata)
        _finite_pca = X[np.isfinite(X)]
        if not _layer_is_log_pca and _finite_pca.size > 0 and np.nanmedian(_finite_pca) > 1e4:
            print(
                f"{format_log_prefix('info')} Layer {layer!r} appears to contain "
                f"non-log intensities (median={np.nanmedian(_finite_pca):.2e}). "
                "PCA will proceed with z-scoring (standard practice). Optionally "
                "run pdata.log_transform() first for better low-abundance separation."
            )
        elif _layer_is_log_pca:
            print(
                f"{format_log_prefix('info')} Layer {layer!r} detected as log-transformed. "
                "Using log-space values for PCA."
            )

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Performing PCA [{on}] using layer: {layer}, removing NaN features.")
        print(f"   🔸 BEFORE (samples × proteins): {X.shape}")
        Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)
        nan_cols = np.isnan(Xnorm).any(axis=0)
        Xnorm = Xnorm[:, ~nan_cols]
        print(f"   🔸 AFTER  (samples × proteins): {Xnorm.shape}")

        # TODO: fix bug here (ValueError: n_components=59 must be between 1 and min(n_samples, n_features)=31 with svd_solver='arpack')
        pca_data = sc.tl.pca(Xnorm, return_info=True, **kwargs)
        adata.obsm['X_pca'] = pca_data[0]
        PCs = np.zeros((pca_data[1].shape[0], nan_cols.shape[0]))
        
        # fill back the 0s where column was NaN in the original data, and thus not used in PCA
        counter = 0
        for i in range(PCs.shape[1]):
            if not nan_cols[i]:
                PCs[:, i] = pca_data[1][:, counter]
                counter += 1

        adata.uns['pca'] = {'PCs': PCs, 'variance_ratio': pca_data[2], 'variance': pca_data[3]}
        
        subpdata = "prot" if on == 'protein' else "pep"

        self._append_history(f'{on}: PCA fitted on {layer}, stored in obsm["X_pca"] and uns["pca"]["PCs"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only',indent=2)} PCA complete, fitted on {layer}. Results stored in:")
        print(f"       • .{subpdata}.obsm['X_pca']")
        print(f"       • .{subpdata}.uns['pca'] (includes PCs, variance, variance ratio)")
        var_pc1, var_pc2 = pca_data[2][:2]
        print(f"       • Variance explained by PC1/PC2: {var_pc1*100:.2f}% , {var_pc2*100:.2f}%") 

    def harmony(self, key, on = 'protein'):
        """
        Perform batch correction using Harmony integration.

        This method applies Harmony-based batch correction (via `scanpy.external.pp.harmony_integrate`)
        on PCA-reduced protein or peptide data to mitigate batch effects across samples.

        Args:
            key (str): Column name in `.obs` representing the batch variable to correct.
            on (str): Whether to use "protein" or "peptide" data. Accepts "prot"/"protein" or "pep"/"peptide" (default: "protein").

        Returns:
            None

        Example:
            Perform Harmony integration on protein-level PCA embeddings:
                ```python
                pdata.harmony(key="batch", on="protein")
                ```

            Apply Harmony on peptide-level data instead:
                ```python
                pdata.harmony(key="run_id", on="peptide")
                ```

        Note:
            - Harmony requires prior PCA computation. If PCA is missing, it will be computed automatically.
            - The Harmony-corrected coordinates are stored in `.obsm["X_pca_harmony"]`.
            - Updates the processing history via `.history`.

        Todo:
            Add optional arguments for controlling Harmony parameters (e.g., `max_iter_harmony`, `theta`, `lambda`).
        """

        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
       
        if on == 'protein' or on == 'prot':
            adata = self.prot
        elif on == 'peptide' or on == 'pep':
            adata = self.pep

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Performing Harmony batch correction on [{on}] PCA.")

        # check if pca has been run before, look for distances and connectivities in obsp
        if 'pca' not in adata.uns:
            print(f"{format_log_prefix('info_only', indent=2)} PCA not found in AnnData object. Running PCA with default settings.")
            self.pca(on = on, layer = "X")

        # check that key is valid column in adata.obs
        if key not in adata.obs.columns:
            raise ValueError(f"Batch key '{key}' not found in adata.obs.")

        sc.external.pp.harmony_integrate(adata, key)

        self._append_history(f'{on}: Harmony batch correction applied on key {key}, stored in obsm["X_pca_harmony"] and uns["umap"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only', indent=2)} Harmony batch correction complete. Results stored in:")
        print(f"       • obsm['X_pca_harmony'] (PCA coordinates)")

    def pca_gsea(self, pcs=[1, 2], on="protein", gene_sets=(
        "KEGG_2026,Reactome_Pathways_2024,WikiPathways_2024_Human,"
        "GO_Biological_Process_2025,MSigDB_Hallmark_2020"
    ), gene_col="Genes",
        min_size=5, max_size=500, permutation_num=1000, weight=1, threads=4, seed=0,
        key_added="pca_gsea", verbose=True, fdr_report_cutoffs=(0.05, 0.25), **kwargs,):
        """
        Run preranked GSEA on PCA loadings for selected principal components.

        Args:
            pcs (list[int] or None): Principal components to analyze, 1-indexed.
                Defaults to [1, 2]. If None, run all available PCs.
            on (str): Whether to use "protein" or "peptide" data.
            gene_sets (str or dict): Enrichr library name(s) for GSEApy. Pass a comma-separated
                string to merge multiple libraries; pathway keys are prefixed ``LIBRARY__term``
                in merged results.
            gene_col (str): Column in `.var` containing gene symbols.
            min_size (int): Minimum gene set size.
            max_size (int): Maximum gene set size.
            permutation_num (int): Number of permutations for prerank.
            weight (float): Weighting parameter for enrichment score.
            threads (int): Number of threads for GSEApy.
            seed (int): Random seed.
            key_added (str): Key for storing results in `.uns`.
            verbose (bool): Whether to print progress.
            fdr_report_cutoffs (tuple[float, ...]): FDR thresholds for verbose per-PC reporting
                (counts of terms with ``FDR q-val`` at or below each cutoff). Default ``(0.05, 0.25)``.
                Pass an empty tuple to omit.
            **kwargs: Additional keyword arguments passed to `gseapy.prerank()`.

        Returns:
            None

        Note:
            Each DataFrame in ``.{prot|pep}.uns[key_added]['results']`` includes gseapy columns
            plus ``library`` and ``pathway`` parsed from ``Term`` when using merged libraries
            (``LIBRARY__pathway_name``).
        """
        if not self._check_data(on):  # type: ignore[attr-defined]
            return

        if on == "protein":
            adata = self.prot
        elif on == "peptide":
            adata = self.pep
        else:
            raise ValueError("`on` must be either 'protein' or 'peptide'.")

        subpdata = "prot" if on == "protein" else "pep"

        if "pca" not in adata.uns or "PCs" not in adata.uns["pca"]:
            print(f"{format_log_prefix('warn')} PCA results not found in .{subpdata}.uns['pca'].")
            print(f"{format_log_prefix('blank', 3)} Please run `.pca()` first, then rerun `pca_gsea()`.")
            return

        # additional helper functions
        import logging
        import io
        from contextlib import contextmanager, redirect_stdout, redirect_stderr

        @contextmanager
        def _suppress_gseapy_output():
            previous_disable = logging.root.manager.disable
            sink = io.StringIO()
            logging.disable(logging.ERROR)
            try:
                with redirect_stdout(sink), redirect_stderr(sink):
                    yield
            finally:
                logging.disable(previous_disable)

        def _percent_tied_values(values):
            """Return percent of entries that are part of a duplicated value group."""
            s = pd.Series(values)
            if s.empty:
                return 0.0
            return 100 * s.duplicated(keep=False).sum() / len(s)
        
        def _build_pca_rank_df(adata, genes, loadings, use_abs_for_duplicates=False):
            rank_df = pd.DataFrame({
                "feature": adata.var_names.astype(str),
                "gene": genes.values,
                "loading": loadings,
            }).dropna(subset=["gene"]).copy()

            dup_genes = sorted(rank_df.loc[rank_df["gene"].duplicated(keep=False), "gene"].unique())

            if len(dup_genes) > 0:
                if use_abs_for_duplicates:
                    idx = rank_df.groupby("gene")["loading"].agg(lambda x: x.abs().idxmax())
                else:
                    idx = rank_df.groupby("gene")["loading"].idxmax()
                rank_df = rank_df.loc[idx].copy()

            rank_df["abs_loading"] = rank_df["loading"].abs()
            rank_df = rank_df.sort_values("loading", ascending=False).reset_index(drop=True)
            return rank_df, dup_genes
        
        # ----------------------------------

        genes = _gseapy_resolve_uppercase_genes(
            adata,
            gene_col=gene_col,
        )

        if genes is None:
            context="PCA GSEA"
            print(f"{format_log_prefix('warn')} `.var[{gene_col!r}]` not found.")
            print(f"{format_log_prefix('blank',3)} {context} requires resolved gene symbols.")
            print(f"{format_log_prefix('blank',3)} Please annotate gene names first, then rerun.")
            return

        pcs_all = adata.uns["pca"]["PCs"]
        n_pcs_available = pcs_all.shape[0]

        if pcs is None:
            pcs = list(range(1, n_pcs_available + 1))
            print(f"{format_log_prefix('warn')} `pcs=None` detected – running GSEA on all available PCs.")
            print(f"{format_log_prefix('blank', 3)} This may take a long time, especially with many PCs or large gene set libraries.")

        if not isinstance(pcs, (list, tuple, np.ndarray)):
            raise ValueError("`pcs` must be a list of integers or None.")

        pcs = [int(pc) for pc in pcs]

        invalid_pcs = [pc for pc in pcs if pc < 1 or pc > n_pcs_available]
        if invalid_pcs:
            raise ValueError(
                f"Invalid PCs requested: {invalid_pcs}. "
                f"Available PCs are 1 to {n_pcs_available}."
            )

        # Precompute tied-loading percentages once per requested PC
        _, dup_genes = _build_pca_rank_df(adata, genes, pcs_all[pcs[0] - 1, :], use_abs_for_duplicates=True)
        tied_pct_by_pc = {}
        for pc in pcs:
            loadings = pcs_all[pc - 1, :]
            rank_df_tmp, _ = _build_pca_rank_df(adata, genes, loadings, use_abs_for_duplicates=True)
            rnk_tmp = rank_df_tmp.set_index("gene")["loading"]
            tied_pct_by_pc[pc] = _percent_tied_values(rnk_tmp.values)

        if verbose:
            print(f"{format_log_prefix('user')} Running PCA GSEA [{on}] on PCs: {pcs}", flush=True)
            print(f"   🔸 Gene set library: {gene_sets}")
            print(f"   🔸 Gene column: {gene_col}")

            if len(dup_genes) > 0:
                preview = ", ".join(dup_genes[:10])
                suffix = " ..." if len(dup_genes) > 10 else ""
                print(f"{format_log_prefix('warn_only',2)} Duplicated genes ({len(dup_genes)}): {preview}{suffix}")
                print(f"     Using maximum absolute loading for duplicated genes.")

                base_params = dict(min_size=min_size, max_size=max_size, permutation_num=permutation_num, weight=weight, threads=threads, seed=seed)
                params = {**base_params, **kwargs}
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())

                print(f"{format_log_prefix('api',2)} gseapy.prerank({param_str})")

            tied_msg = ", ".join([f"PC{pc} {tied_pct_by_pc[pc]:.2f}%" for pc in pcs])
            print(f"{format_log_prefix('info_only',3)} Tied prerank stats: {tied_msg}. GSEApy may order tied genes arbitrarily.")

        stored_results = {}
        stored_rankings = {}

        for pc in pcs:
            loadings = pcs_all[pc - 1, :]

            rank_df, _ = _build_pca_rank_df(adata, genes, loadings, use_abs_for_duplicates=True)
            rnk = rank_df.set_index("gene")["loading"]

            if verbose:
                print(
                    f"{format_log_prefix('blank',3)}🔹 PC{pc}: "
                    f"{len(loadings)} features -> "
                    f"{rank_df.shape[0]} unique uppercase genes"
                )

            with _suppress_gseapy_output():
                pre_res = gp.prerank(
                    rnk=rnk,
                    gene_sets=gene_sets,
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=permutation_num,
                    weight=weight,
                    threads=threads,
                    seed=seed,
                    outdir=None,
                    verbose=False,
                    **kwargs,
                )

            res_df = _annotate_pca_gsea_result_df(pre_res.res2d.copy())
            stored_results[f"PC{pc}"] = res_df
            stored_rankings[f"PC{pc}"] = rank_df.copy()

            if verbose:
                msg = f"{format_log_prefix('result_only', 4)} PC{pc}: {res_df.shape[0]} enriched terms returned"
                fdr_col = "FDR q-val"
                if fdr_report_cutoffs and fdr_col in res_df.columns:
                    s = pd.to_numeric(res_df[fdr_col], errors="coerce")
                    parts = [f"n(FDR<={c})={int((s <= c).sum())}" for c in fdr_report_cutoffs]
                    msg += "; " + ", ".join(parts)
                elif fdr_report_cutoffs and fdr_col not in res_df.columns:
                    msg += f"; (no '{fdr_col}' column for FDR counts)"
                print(msg)

        adata.uns[key_added] = {
            "params": {
                "pcs": pcs,
                "gene_sets": gene_sets,
                "gene_col": gene_col,
                "min_size": min_size,
                "max_size": max_size,
                "permutation_num": permutation_num,
                "weight": weight,
                "threads": threads,
                "seed": seed,
                "on": on,
            },
            "results": stored_results,
            "rankings": stored_rankings,
        }

        self._append_history(  # type: ignore[attr-defined]
            f'{on}: PCA GSEA run on PCs {pcs}, stored in .{subpdata}.uns["{key_added}"]'
        )

        if verbose:
            print(f"{format_log_prefix('result')} PCA GSEA complete.")
            print(f"   • Results stored in: .{subpdata}.uns['{key_added}']['results']")
            print(f"   • Keys: {list(stored_results.keys())}")

    def ssgsea(self, on="protein", layer="X", gene_sets="GO_Biological_Process_2023", gene_col="Genes", 
               min_size=5, max_size=500, threads=4, seed=0, key_added="ssgsea", verbose=True, **kwargs,):
        """
        Compute per-sample ssGSEA pathway scores.

        Args:
            on (str): Whether to use "protein" or "peptide" data.
            layer (str): Data layer to use.
            gene_sets (str or dict): Gene set library for GSEApy.
            gene_col (str): Column in `.var` containing gene symbols.
            min_size (int): Minimum gene set size.
            max_size (int): Maximum gene set size.
            threads (int): Number of threads for GSEApy.
            seed (int): Random seed.
            key_added (str): Base key for storing results.
            verbose (bool): Whether to print progress.
            **kwargs: Additional keyword arguments passed to `gseapy.ssgsea()`.

        Returns:
            None
        """
        if not self._check_data(on):  # type: ignore[attr-defined]
            return

        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep
        else:
            raise ValueError("`on` must be either 'protein' or 'peptide'.")

        genes = _gseapy_resolve_uppercase_genes(
            adata,
            gene_col=gene_col,
        )
        
        if genes is None:
            context="ssGSEA"
            print(f"{format_log_prefix('warn')} `.var[{gene_col!r}]` not found.")
            print(f"{format_log_prefix('blank',3)} {context} requires resolved gene symbols.")
            print(f"{format_log_prefix('blank',3)} Please annotate gene names first, then rerun.")
            return

        X = utils.get_adata_layer(adata, layer=layer)
        expr_df = pd.DataFrame(
            X.T,
            index=genes.values,
            columns=adata.obs_names.astype(str),
        )

        expr_df = expr_df[~pd.isna(expr_df.index)].copy()

        dup_genes = sorted(pd.Index(expr_df.index)[pd.Index(expr_df.index).duplicated(keep=False)].unique())
        if len(dup_genes) > 0:
            _print_duplicate_gene_warning(dup_genes, method_desc="mean abundance")
            expr_df = expr_df.groupby(expr_df.index).mean()

        log_prefix = format_log_prefix("user")
        subpdata = "prot" if on == "protein" else "pep"

        if verbose:
            print(f"{log_prefix} Running ssGSEA [{on}] using layer: {layer}")
            print(f"   🔸 Gene set library: {gene_sets}")
            print(f"   🔸 Gene column: {gene_col}")
            print(f"   🔸 Input matrix after cleanup (genes × samples): {expr_df.shape}")

        ss = gp.ssgsea(
            data=expr_df,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            threads=threads,
            seed=seed,
            outdir=None,
            verbose=False,
            **kwargs,
        )

        res_df = ss.res2d.copy()

        # GSEApy ssGSEA returns long-form results with:
        # Name = sample, Term = pathway, ES = enrichment score
        score_df = res_df.pivot(index="Term", columns="Name", values="ES").T
        score_df = score_df.reindex(adata.obs_names.astype(str))

        adata.obsm[f"X_{key_added}"] = score_df
        adata.uns[key_added] = {
            "params": {
                "layer": layer,
                "gene_sets": gene_sets,
                "gene_col": gene_col,
                "min_size": min_size,
                "max_size": max_size,
                "threads": threads,
                "seed": seed,
                "on": on,
            },
            "long_results": res_df,
            "pathway_names": list(score_df.columns),
        }

        self._append_history(  # type: ignore[attr-defined]
            f'{on}: ssGSEA run on {layer}, stored in .{subpdata}.obsm["X_{key_added}"] and .{subpdata}.uns["{key_added}"]'
        )

        if verbose:
            print(f"{format_log_prefix('result_only', indent=2)} ssGSEA complete. Results stored in:")
            print(f"       • .{subpdata}.obsm['X_{key_added}'] (samples × pathways)")
            print(f"       • .{subpdata}.uns['{key_added}']")
            print(f"       • Pathways scored: {score_df.shape[1]}")

    def nanmissingvalues(self, on = 'protein', limit = 0.5):
        """
        Set columns (proteins or peptides) with excessive missing values to NaN.

        This method scans all features and replaces their corresponding columns with NaN
        if the fraction of missing values exceeds the given threshold. It helps ensure
        downstream normalization and imputation steps are applied to meaningful features only.

        Args:
            on (str): Whether to use "protein" or "peptide" data. Accepts "prot"/"protein" or "pep"/"peptide" (default: "protein").
            limit (float): Proportion threshold for missing values (default: 0.5). 
                Features with more than `limit × 100%` missing values are set entirely to NaN.

        Returns:
            None

        !!! warning "Deprecation Notice"
            This function may be deprecated in future releases.  
            Use [`annotate_found`](reference/pAnnData/editing_mixins/#src.scpviz.pAnnData.editing_mixins.annotate_found)  
            and [`filter_prot_found`](reference/pAnnData/editing_mixins/#src.scpviz.pAnnData.editing_mixins.filter_prot_found)  
            for more robust and configurable detection-based filtering.
            
        Example:
            Mask proteins with more than 50% missing values:
                ```python
                pdata.nanmissingvalues(on="protein", limit=0.5)
                ```

            Apply the same filter for peptide-level data:
                ```python
                pdata.nanmissingvalues(on="peptide", limit=0.3)
                ```

        Note:
            - The missing-value fraction is computed per feature across all samples.
            - This operation modifies the `.X` matrix in-place.
            - The updated data are stored back into `.prot` or `.pep`.
        """
        import scipy.sparse
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        if on == 'protein':
            adata = self.prot

        elif on == 'peptide':
            adata = self.pep

        if scipy.sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        missing_proportion = np.isnan(X).mean(axis=0)
        columns_to_nan = missing_proportion > limit
        X[:, columns_to_nan] = np.nan
        adata.X = scipy.sparse.csr_matrix(X) if scipy.sparse.issparse(adata.X) else X

        if on == 'protein':
            self.prot = adata
        elif on == 'peptide':
            self.pep = adata

    def normalize(self, classes = None, layer = "X", method = 'sum', on = 'protein', set_X = True, force = False, use_nonmissing = False, **kwargs):  
        """
        Normalize sample intensities across protein or peptide data.

        This method performs global or group-wise normalization of the selected data layer.
        It supports multiple normalization strategies ranging from simple scaling
        (e.g., sum, median) to advanced approaches such as `reference_feature` and
        [`directlfq`]((https://doi.org/10.1016/j.mcpro.2023.100581)).

        Args:
            classes (str or list, optional): Sample-level grouping column(s) in `.obs` to
                perform group-wise normalization. If None, normalization is applied globally.
            layer (str, optional): Data layer to normalize from (default: `"X"`).
            method (str, optional): Normalization strategy to apply. Options include:
                `'sum'`, `'median'`, `'mean'`, `'max'`, `'reference_feature'`,
                `'robust_scale'`, `'quantile_transform'`, `'directlfq'`.
            on (str, optional): Whether to use `"protein"` or `"peptide"` data.
            set_X (bool, optional): Whether to set `.X` to the normalized result (default: True).
            force (bool, optional): Proceed with normalization even if samples exceed the
                allowed fraction of missing values (default: False).
            use_nonmissing (bool, optional): If True, only use columns with no missing values
                across all samples when computing scaling factors (default: False).
            **kwargs: Additional keyword arguments for normalization methods.
                - `reference_columns` (list): For `'reference_feature'`, specify columns or
                gene names to normalize against.
                - `max_missing_fraction` (float): Maximum allowed fraction of missing values
                per sample (default: 0.5).
                - `n_neighbors` (int): For methods requiring neighbor-based computations.
                - `input_type_to_use` (str): For `'directlfq'`, specify `'pAnnData'`,
                `'diann_precursor_ms1'`, or `'diann_precursor_ms1_and_ms2'`.
                - `path` (str): For `'directlfq'`, path to the `report.tsv` or `report.parquet`
                file from DIA-NN output.
                - `strict` (bool): For `'directlfq'`, whether to use unique + shared peptides or only unique peptides. Defaults to False (unique + shared).

        Returns:
            None

        Example:
            Perform global normalization using the median intensity:
                ```python
                pdata.normalize(on="protein", method="median")
                ```

            Apply group-wise normalization by treatment class using sum-scaling:
                ```python
                pdata.normalize(classes="treatment", method="sum", on="protein")
                ```

            Run reference-feature normalization using specific genes:
                ```python
                pdata.normalize(
                    on="protein",
                    method="reference_feature",
                    reference_columns=["ACTB", "GAPDH"]
                )
                ```

        !!! tip "About `directlfq` normalization"
            - The `directlfq` method aggregates peptide-level data to protein-level intensities
            and stores results in a new protein-layer (e.g. `'X_norm_directlfq'`).
            - It does not support group-wise normalization.
            - Processing time may scale with dataset size.
            - For algorithmic and benchmarking details, see:  
            **Ammar, Constantin et al. (2023)**  
            *Accurate Label-Free Quantification by directLFQ to Compare Unlimited Numbers of Proteomes.*  
            *Molecular & Cellular Proteomics*, 22(7):100581.  
            [https://doi.org/10.1016/j.mcpro.2023.100581](https://doi.org/10.1016/j.mcpro.2023.100581)



        Note:
            - Results are stored in a new layer named `'X_norm_<method>'`.
            - The normalized layer replaces `.X` if `set_X=True`.
            - Normalization operations are recorded in `.history`.
            - For consistency across runs, consider running `.impute()` before normalization.

        Todo:
            - Add optional z-score and percentile normalization modes.
            - Add support for specifying external scaling factors.
        """

        
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in .{on}.")
       
        normalize_data = adata.layers[layer] if layer != "X" else adata.X
        was_sparse = sparse.issparse(normalize_data)
        normalize_data = normalize_data.toarray() if was_sparse else normalize_data.copy()
        original_data = normalize_data.copy()

        layer_name = 'X_norm_' + method
        normalize_funcs = ['sum', 'median', 'mean', 'max', 'reference_feature', 'robust_scale', 'quantile_transform','directlfq']

        if method not in normalize_funcs:
            raise ValueError(f"Unsupported normalization method: {method}")

        # Special handling for directlfq
        if method == "directlfq":
            if classes is not None:
                print(f"{format_log_prefix('warn')} 'directlfq' does not support group-wise normalization. Proceeding with global normalization.")
                classes = None

            print(f"{format_log_prefix('user')} Running directlfq normalization on peptide-level data.")
            print(f"{format_log_prefix('info_only', indent=2)} Note: please be patient, directlfq can take a minute to run depending on data size. Output files will be produced.")
            normalize_data = self._normalize_helper_directlfq(**kwargs)

            adata = self.prot  # directlfq always outputs protein-level intensities
            _resolved_input = utils.resolve_input_layer(adata, layer)
            actual_layer_name = utils.update_layer_provenance(
                adata,
                layer_name=layer_name,
                op="normalize",
                input_layer=_resolved_input,
                method="directlfq",
            )
            adata.layers[actual_layer_name] = (
                sparse.csr_matrix(normalize_data) if was_sparse else normalize_data
            )

            if set_X:
                self.set_X(layer=actual_layer_name, on="protein")  # type: ignore[attr-defined]

            self._append_history(  # type: ignore[attr-defined]
                f"protein: Normalized layer using directlfq (input_type={kwargs.get('input_type_to_use', 'default')}). Stored in `{actual_layer_name}`."
            )
            print(
                f"{format_log_prefix('result_only', indent=2)} directlfq normalization complete. "
                f"Results are stored in layer '{actual_layer_name}'."
            )
            print(f"{format_log_prefix('warn_only',3)} Downstream imputation should be performed with the flag `use_zeros_as_nan` set to True due to directlfq output format returning NaNs as 0s.")
            return
    
        # --- standard normalization ---
        # Build the header message early
        if classes is None:
            msg = f"{format_log_prefix('user')} Global normalization using '{method}'"
        else:
            msg = f"{format_log_prefix('info_only')} Group-wise normalization using '{method}' on class(es): {classes}"

        if use_nonmissing and method in {'sum', 'mean', 'median', 'max'}:
            msg += " (using only fully observed columns)"
        msg += f". Layer will be saved as '{layer_name}'."

        # ✅ Print message before checking for missing values
        print(msg)

        # Check for bad rows (too many missing values)
        missing_fraction = np.isnan(normalize_data).sum(axis=1) / normalize_data.shape[1]
        max_missing_fraction = kwargs.pop("max_missing_fraction", 0.5)
        bad_rows_mask = missing_fraction > max_missing_fraction

        if np.any(bad_rows_mask):
            n_bad = np.sum(bad_rows_mask)
            print(f"{format_log_prefix('warn',2)} {n_bad} sample(s) have >{int(max_missing_fraction*100)}% missing values.")
            print("     Try running `.impute()` before normalization. Suggest to use the flag `use_nonmissing=True` to normalize using only consistently observed proteins.")
            if not force:
                print("     ➡️ Use `force=True` to proceed anyway.")
                return
            print(f"{format_log_prefix('warn',2)} Proceeding with normalization despite bad rows (force=True).")

        if classes is None:
            normalize_data = self._normalize_helper(normalize_data, method, use_nonmissing=use_nonmissing, **kwargs)
        else:
            # Group-wise normalization
            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            for group in unique_groups:
                idx = np.where(sample_names == group)[0]
                group_data = normalize_data[idx, :]

                normalized_group = self._normalize_helper(group_data, method=method, use_nonmissing=use_nonmissing, **kwargs)
                normalize_data[idx, :] = normalized_group

        # summary printout
        summary_lines = []
        if classes is None:
            summary_lines.append(f"{format_log_prefix('result_only', indent=2)} Normalized all {normalize_data.shape[0]} samples.")
        else:
            for group in unique_groups:
                count = np.sum(sample_names == group)
                summary_lines.append(f"   - {group}: {count} samples normalized")
            summary_lines.insert(0, f"{format_log_prefix('result_only', indent=2)} Normalized {normalize_data.shape[0]} samples total.")
        print("\n".join(summary_lines))

        _resolved_input = utils.resolve_input_layer(adata, layer)
        actual_layer_name = utils.update_layer_provenance(
            adata,
            layer_name=layer_name,
            op="normalize",
            input_layer=_resolved_input,
            method=method,
        )
        adata.layers[actual_layer_name] = (
            sparse.csr_matrix(normalize_data) if was_sparse else normalize_data
        )

        if set_X:
            self.set_X(layer=actual_layer_name, on=on)  # type: ignore[attr-defined], EditingMixin

        # Determine if use_nonmissing note should be added
        note = ""
        if use_nonmissing and method in {'sum', 'mean', 'median', 'max'}:
            note = " (using only fully observed columns)"

        self._append_history(  # type: ignore[attr-defined], HistoryMixin
            f"{on}: Normalized layer {layer} using {method}{note} (grouped by {classes}). "
            f"Stored in `{actual_layer_name}`."
        )

    def _normalize_helper(self, data, method, use_nonmissing, **kwargs):
        """
        Perform row-wise normalization using a selected method.

        Used internally by `normalize()` to compute per-sample scaling.
        Supports reference feature scaling, robust methods, and quantile normalization.

        Args:
            data (np.ndarray): Sample × feature data matrix.
            method (str): Normalization strategy. Options:
                - 'sum'
                - 'mean'
                - 'median'
                - 'max'
                - 'reference_feature'
                - 'robust_scale'
                - 'quantile_transform'
            use_nonmissing (bool): If True, computes scaling using only columns with no NaNs.

        Returns:
            np.ndarray: Normalized data matrix.
        """

        if method in {'sum', 'mean', 'median', 'max'}:
            reducer = {
                    'sum': np.nansum,
                    'mean': np.nanmean,
                    'median': np.nanmedian,
                    'max': np.nanmax
                }[method]

            if use_nonmissing:
                fully_observed_cols = ~np.isnan(data).any(axis=0)
                if not np.any(fully_observed_cols):
                    raise ValueError("No fully observed columns available for normalization with `use_nonmissing=True`.")
                used_cols = np.where(fully_observed_cols)[0]
                print(f"{format_log_prefix('info_only',2)} Normalizing using only fully observed columns: {len(used_cols)}")
                row_vals = reducer(data[:, fully_observed_cols], axis=1)
            else:
                row_vals = reducer(data, axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.nanmax(row_vals) / row_vals
            scale = np.where(np.isnan(scale), 1.0, scale) # metaboanalyst: scale = 1.0 / row_vals
            data_norm = data * scale[:, None]

        elif method == 'reference_feature':
            # norm by reference feature: scale each row s.t. the reference column is the same across all rows (scale to max value of reference column)
            reference_columns = kwargs.get('reference_columns', [2])
            reference_method = kwargs.get('reference_method', 'median')  # default to median

            reducer_map = {
                'mean': np.nanmean,
                'median': np.nanmedian,
                'sum': np.nansum
            }

            if reference_method not in reducer_map:
                raise ValueError(f"Unsupported reference method: {reference_method}. Supported methods are: {list(reducer_map.keys())}")
            reducer = reducer_map[reference_method]

            # resolve reference column names if needed
            if isinstance(reference_columns[0], str):
                gene_to_acc, _ = self.get_gene_maps(on='protein') # type: ignore[attr-defined], IdentifierMixin
                resolved = utils.resolve_accessions(self.prot, reference_columns, gene_map=gene_to_acc)
                reference_acc = [ref for ref in resolved if ref in self.prot.var.index]
                reference_columns = [self.prot.var.index.get_loc(ref) for ref in reference_acc]
                print(f"{format_log_prefix('info')} Normalizing using found reference columns: {reference_acc}")
                self._history.append(f"Used reference_feature normalization with resolved accessions: {resolved}") # type: ignore[attr-defined]
            else:
                reference_columns = [int(ref) for ref in reference_columns]
                reference_acc = [self.prot.var.index[ref] for ref in reference_columns if ref < self.prot.shape[1]]
                print(f"{format_log_prefix('info')} Normalizing using reference columns: {reference_acc}")
                self._history.append(f"Used reference_feature normalization with resolved accessions: {reference_acc}") # type: ignore[attr-defined]

            scaling_factors = np.nanmean(np.nanmax(data[:, reference_columns], axis=0) / (data[:, reference_columns]), axis=1)

            nan_rows = np.where(np.isnan(scaling_factors))[0]
            if nan_rows.size > 0:
                print(f"{format_log_prefix('warn')} Rows {list(nan_rows)} have all missing reference values.")
                print(f"{format_log_prefix('info')} Falling back to row median normalization for these rows.")

                fallback = np.nanmedian(data[nan_rows, :], axis=1)
                fallback[fallback == 0] = np.nan  # avoid division by 0
                fallback_scale = np.nanmax(fallback) / fallback
                fallback_scale = np.where(np.isnan(fallback_scale), 1.0, fallback_scale)  # default to 1.0 if all else fails

                scaling_factors[nan_rows] = fallback_scale

            scaling_factors = np.where(np.isnan(scaling_factors), np.nanmean(scaling_factors), scaling_factors)
            data_norm = data * scaling_factors[:, None]

        elif method == 'robust_scale':
            # norm by robust_scale: Center to the median and component wise scale according to the interquartile range. See sklearn.preprocessing.robust_scale for more information.
            from sklearn.preprocessing import robust_scale
            data_norm = robust_scale(data, axis=1)

        elif method == 'quantile_transform':
            # norm by quantile_transform: Transform features using quantiles information. See sklearn.preprocessing.quantile_transform for more information.
            from sklearn.preprocessing import quantile_transform
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data_norm = quantile_transform(data, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        return data_norm

    def _normalize_helper_directlfq(self, input_type_to_use="pAnnData", path=None, strict=False, **kwargs):
        """
        Run directlfq normalization and return normalized protein-level intensities.

        Args:
            input_type_to_use (str): Either 'pAnnData' (default) or 
                'diann_precursor_ms1_and_ms2'.
            path (str, optional): Path to DIA-NN report file (required if 
                input_type_to_use='diann_precursor_ms1_and_ms2').
            strict (bool): Whether to be strict and use only unique peptides, in which shared peptides are ignored for normalization. Defaults to False.
            **kwargs: Passed to directlfq.lfq_manager.run_lfq().

        Returns:
            np.ndarray: Normalized data (samples × proteins).
        """
        import directlfq.lfq_manager as lfq_manager
        import os

        strict = kwargs.pop("strict", False)

        if input_type_to_use == "diann_precursor_ms1_and_ms2":
            if path is None:
                raise ValueError("For input_type_to_use='diann_precursor_ms1_and_ms2', please provide the DIA-NN report path via `path`.")
            lfq_manager.run_lfq(path, input_type_to_use=input_type_to_use, **kwargs)

        else:
            # check if pep exists
            if self.pep is None:
                raise ValueError("Peptide-level data not found. Please load peptide data before running directlfq normalization.")
            
            # Build peptide-level input table from .pep
            X = self.pep.layers.get("X_precursor", self.pep.X)
            if not isinstance(X, pd.DataFrame):
                X = X.toarray() if hasattr(X, "toarray") else X
            X_df = pd.DataFrame(
                X.T,
                index=self.pep.var_names,
                columns=self.pep.obs_names
            )
            prot_col = "Protein.Group" if "Protein.Group" in self.pep.var.columns else "Master Protein Accessions"
            X_df.insert(0, "protein", self.pep.var[prot_col].to_list())
            X_df.insert(1, "ion", X_df.index.to_list())

            if not strict:
                # e.g. "P03995; P03995-2" → two rows with same intensities, one per protein
                # ensure directLFQ sees only single-accession protein IDs
                n_before = X_df.shape[0]
                X_df["protein"] = X_df["protein"].astype(str).str.split(";")
                X_df = X_df.explode("protein")
                X_df["protein"] = X_df["protein"].str.strip()
                X_df = X_df[X_df["protein"] != ""]
                
                n_after = X_df.shape[0]
                if n_after > n_before:
                    print(f"{format_log_prefix('info',2)} Expanded multi-protein peptide groups: {n_before} → {n_after} rows.")
                else:
                    print(f"{format_log_prefix('info',2)} No multi-protein peptide groups detected for directlfq input.")

            X_df.reset_index(drop=True, inplace=True)
            tmp_file = "peptide_matrix.aq_reformat.tsv"
            X_df.to_csv(tmp_file, sep="\t", index=False)
            lfq_manager.run_lfq(tmp_file, **kwargs)

        # Load directlfq output (look for protein_intensities file)
        out_file = None
        for f in os.listdir("."):
            if f.endswith("protein_intensities.tsv"):
                out_file = f
        if out_file is None:
            raise FileNotFoundError("directlfq did not produce a '*protein_intensities.tsv' file in current directory.")

        norm_prot = pd.read_csv(out_file, sep="\t").set_index("protein")
        aligned = norm_prot.reindex(
            index=self.prot.var_names,
            columns=self.prot.obs_names
        )

        return aligned.T.to_numpy()

    def pairwise_correlation(
        self,
        classes: str | list[str],
        on: str = "protein",
        layer: str = "X",
        method: str = "pearson",
        order: list | None = None,
        compute_sample_matrix: bool = False,
        force: bool = False,
        subset_mask: np.ndarray | list | None = None,
    ) -> None:
        """
        Compute pairwise proteome correlations across groups defined by `.obs` metadata.

        For each group label (one column, or combined labels from multiple columns),
        computes the mean expression profile (nanmean), drops features with NaN in any
        group, then computes an (n_groups × n_groups) correlation or distance matrix.
        Optionally also computes a sample-level matrix.
        Results are stored in `adata.uns["pairwise_corr"]`.

        Args:
            classes (str or list of str): One `.obs` column name, or a list of column
                names, defining sample groups (same convention as `normalize()` and
                `plot_pca()`). Multiple columns are combined per sample using
                :func:`~scpviz.utils.get_samplenames` (comma-space join, same as
                group-wise ``normalize``).
            on (str): Whether to use `"protein"` or `"peptide"` data (default: `"protein"`).
            layer (str): Data layer to use. `"X"` uses `.X`; any other value uses
                `.layers[layer]`. Default: `"X"`.
            method (str): Correlation/distance metric. One of:
                `"pearson"`, `"spearman"`, `"euclidean"`. Default: `"pearson"`.
            order (list or None): Preferred ordering of group labels. Unknown values
                are dropped (with a warning); groups not listed are appended after,
                sorted alphabetically (with a warning). If None, order is sorted
                alphabetically.
            compute_sample_matrix (bool): If True, also compute an (n_samples × n_samples)
                matrix sorted to match the resolved group order. Default: False.
            force (bool): If True, recompute even if results are cached in
                `adata.uns["pairwise_corr"]`. Default: False.
            subset_mask (array-like or None): Boolean mask of length `n_obs` selecting
                samples to include. If None, all samples are used.

        Returns:
            None

        Raises:
            ValueError: If `classes` is invalid or missing from `.obs`, `method` is
                invalid, or `subset_mask` has the wrong shape or is all-False.
            KeyError: If `layer` is not `"X"` and not in `adata.layers`.

        Note:
            Features where **any** group mean is NaN are dropped (complete cases only).
            A cache hit still appends a history entry noting that cached results were used.
            For multi-column ``classes``, combined labels use a comma-space separator;
            see ``uns['pairwise_corr']['separator']`` and ``classes_list`` for plotting.

        Example:
            Combined cell line and treatment (labels like ``"AS, kd"``)::

                pdata.pairwise_correlation(
                    classes=["cellline", "treatment"],
                    method="pearson",
                )
        """
        if not self._check_data(on):  # type: ignore[attr-defined], ValidationMixin
            pass

        adata = utils.get_adata(self, on)
        on_norm = "protein" if on in ("protein", "prot") else "peptide"
        subpdata = "prot" if on_norm == "protein" else "pep"

        _classes_list: list[str] = [classes] if isinstance(classes, str) else list(classes)
        if not _classes_list:
            raise ValueError(
                f"{format_log_prefix('error')} classes must be a non-empty string or "
                "list of strings."
            )
        _missing_cols = [c for c in _classes_list if c not in adata.obs.columns]
        if _missing_cols:
            raise ValueError(
                f"{format_log_prefix('error')} classes column(s) not found in "
                f"adata.obs: {_missing_cols}"
            )
        _separator: str | None = ", " if len(_classes_list) > 1 else None

        if method not in ("pearson", "spearman", "euclidean"):
            raise ValueError(
                f"{format_log_prefix('error')} method={method!r} is not supported. "
                "Choose from: 'pearson', 'spearman', 'euclidean'."
            )

        subset_indices: tuple[int, ...] | None
        if subset_mask is None:
            subset_indices = None
        else:
            sm = np.asarray(subset_mask, dtype=bool)
            if sm.ndim != 1 or sm.size != adata.n_obs:
                raise ValueError(
                    f"{format_log_prefix('error')} subset_mask must be a 1D boolean "
                    f"array of length n_obs={adata.n_obs}."
                )
            if not np.any(sm):
                raise ValueError(
                    f"{format_log_prefix('error')} subset_mask selects zero samples."
                )
            subset_indices = tuple(np.flatnonzero(sm).tolist())

        if layer != "X" and layer not in adata.layers:
            raise KeyError(
                f"{format_log_prefix('error')} Layer {layer!r} not found in adata.layers."
            )

        X_full = utils.get_adata_layer(adata, layer)
        X_full = np.asarray(X_full)

        _layer_is_log_pw = utils.infer_layer_is_log(layer, adata)
        _finite_pw = X_full[np.isfinite(X_full)]
        if not _layer_is_log_pw and _finite_pw.size > 0 and np.nanmedian(_finite_pw) > 1e4:
            print(
                f"{format_log_prefix('info')} Layer {layer!r} appears to contain "
                f"non-log intensities (median={np.nanmedian(_finite_pw):.2e}). "
                "Pairwise correlations on non-log data are dominated by highly abundant "
                "proteins. Consider normalize() then log_transform(), then rerun with "
                "the resulting log layer."
            )
        elif _layer_is_log_pw:
            print(
                f"{format_log_prefix('info')} Layer {layer!r} detected as log-transformed. "
                "Using log-space values for correlation."
            )

        raw_labels = utils.get_samplenames(adata, classes)
        obs_labels = pd.Series(
            raw_labels, index=adata.obs_names, name="_combined_class"
        )

        if subset_indices is not None:
            row_sel = np.array(subset_indices, dtype=int)
            X = X_full[row_sel]
            obs_names_used = adata.obs_names[row_sel]
            obs_labels = obs_labels.iloc[row_sel]
        else:
            X = X_full
            obs_names_used = adata.obs_names

        unique_groups = pd.unique(obs_labels).tolist()
        if len(unique_groups) == 0:
            raise ValueError(
                f"{format_log_prefix('error')} No groups found for classes={_classes_list!r} "
                "after applying subset_mask."
            )

        ug_set = set(unique_groups)
        if order is None:
            final_order = sorted(unique_groups)
        else:
            invalid_in_order = [v for v in order if v not in ug_set]
            if invalid_in_order:
                print(
                    f"{format_log_prefix('warn')} order contains values not present in "
                    f"combined labels from {_classes_list!r}; removing: {invalid_in_order}"
                )
            seen: set = set()
            valid_user: list = []
            for v in order:
                if v not in ug_set or v in seen:
                    continue
                seen.add(v)
                valid_user.append(v)
            omitted = sorted(ug_set - set(valid_user))
            if omitted:
                print(
                    f"{format_log_prefix('warn')} groups not listed in order will be "
                    f"appended after your order (alphabetically): {omitted}"
                )
            final_order = valid_user + omitted

        prev = adata.uns.get("pairwise_corr")
        cache_hit = False
        if (
            not force
            and isinstance(prev, dict)
            and prev.get("classes") == classes
            and prev.get("method") == method
            and prev.get("layer") == layer
            and prev.get("compute_sample_matrix") == compute_sample_matrix
            and prev.get("subset_indices") == subset_indices
            and prev.get("order") == final_order
        ):
            cache_hit = True

        if cache_hit:
            print(
                f"{format_log_prefix('info')} Pairwise correlation already computed "
                f"(classes={classes!r}, method={method}). Use force=True to recompute."
            )
            self._append_history(  # type: ignore[attr-defined], HistoryMixin
                f'{on_norm}: pairwise_correlation — used last cached result '
                f'(classes={classes!r}, method={method}, layer={layer}).'
            )
            return

        if not force and isinstance(prev, dict):
            print(
                f"{format_log_prefix('warn')} Recomputing pairwise correlation: "
                "parameters differ from cached result."
            )

        n_groups = len(final_order)
        group_means = np.zeros((n_groups, X.shape[1]), dtype=float)
        for i, g in enumerate(final_order):
            mask = (obs_labels == g).to_numpy()
            rows = X[mask, :]
            if rows.shape[0] == 0:
                group_means[i, :] = np.nan
            else:
                group_means[i, :] = np.nanmean(rows, axis=0)

        nan_mask = np.isnan(group_means).any(axis=0)
        group_means_clean = group_means[:, ~nan_mask]
        n_features_used = int(group_means_clean.shape[1])
        n_features_dropped = int(nan_mask.sum())

        if n_groups < 2:
            if method == "euclidean":
                group_corr_arr = np.zeros((1, 1), dtype=float)
            else:
                group_corr_arr = np.ones((1, 1), dtype=float)
        elif method == "pearson":
            group_corr_arr = np.corrcoef(group_means_clean)
        elif method == "spearman":
            sp = spearmanr(group_means_clean.T).statistic
            sp = np.asarray(sp)
            if sp.ndim == 0:
                group_corr_arr = np.array([[1.0, float(sp)], [float(sp), 1.0]])
            else:
                group_corr_arr = sp
        else:
            group_corr_arr = cdist(
                group_means_clean, group_means_clean, metric="euclidean"
            )

        group_corr_df = pd.DataFrame(
            group_corr_arr, index=final_order, columns=final_order
        )

        sample_corr_df = None
        if compute_sample_matrix:
            sorted_obs_names: list = []
            for group in final_order:
                sorted_obs_names.extend(
                    obs_labels[obs_labels == group].index.tolist()
                )
            if subset_indices is not None:
                _allowed = set(obs_names_used)
                assert all(n in _allowed for n in sorted_obs_names), (
                    "pairwise_correlation: sample-matrix obs names must stay within "
                    "subset_mask; obs_labels should remain sliced after subset_indices."
                )
            row_ix = adata.obs_names.get_indexer(sorted_obs_names)
            X_sorted = X_full[row_ix]
            X_sorted_clean = X_sorted[:, ~nan_mask]
            if method == "pearson":
                sample_corr_df = pd.DataFrame(X_sorted_clean).T.corr(method="pearson")
                sample_corr_df.index = sorted_obs_names
                sample_corr_df.columns = sorted_obs_names
            elif method == "spearman":
                sample_corr_df = pd.DataFrame(X_sorted_clean).T.corr(method="spearman")
                sample_corr_df.index = sorted_obs_names
                sample_corr_df.columns = sorted_obs_names
            else:
                # cdist treats NaNs as NaN output; raw abundance rows usually have missing
                # values, which yields an all-NaN matrix and a blank heatmap. Use nan-aware
                # Euclidean distance between sample vectors (sklearn).
                dist = nan_euclidean_distances(X_sorted_clean)
                sample_corr_df = pd.DataFrame(
                    dist, index=sorted_obs_names, columns=sorted_obs_names
                )

        adata.uns["pairwise_corr"] = {
            "group_matrix": group_corr_df,
            "sample_matrix": sample_corr_df,
            "classes": classes,
            "classes_list": _classes_list,
            "separator": _separator,
            "order": final_order,
            "method": method,
            "layer": layer,
            "compute_sample_matrix": compute_sample_matrix,
            "n_features_used": n_features_used,
            "n_features_dropped": n_features_dropped,
            "subset_indices": subset_indices,
        }

        _classes_display = classes if isinstance(classes, str) else list(classes)
        self._append_history(  # type: ignore[attr-defined], HistoryMixin
            f"{on_norm}: pairwise_correlation on classes={_classes_display!r}, "
            f"method={method}, layer={layer}"
            + (
                f", subset_indices={subset_indices}"
                if subset_indices is not None
                else ""
            )
        )

        print(
            f"{format_log_prefix('user')} Computing pairwise correlation [{on_norm}] "
            f"using layer: {layer}"
        )
        _fp_fc = format_log_prefix("filter_conditions")
        if isinstance(classes, str):
            print(f"{_fp_fc}classes: {classes}")
        else:
            print(f"{_fp_fc}classes: {list(classes)}")
        for _col in _classes_list:
            _u = pd.unique(adata.obs[_col])
            _head = [str(_v) for _v in _u[:5]]
            _more = len(_u) - 5
            _suffix = f" (+{_more} more)" if _more > 0 else ""
            print(f"{_fp_fc}{_col}: {', '.join(_head)}{_suffix}")
        print(f"{_fp_fc}method: {method}")
        order_preview = " | ".join(str(x) for x in final_order)
        print(
            f"{_fp_fc}order (N={len(final_order)}): "
            f"{order_preview}"
        )
        print(
            f"{_fp_fc}Features: {n_features_used} used / "
            f"{adata.n_vars} total ({n_features_dropped} dropped — NaN in ≥1 group mean)"
        )
        _ng = len(final_order)
        print(
            f"{format_log_prefix('result_only', indent=2)} Pairwise correlation complete. "
            "Results stored in:"
        )
        print(f"       • .{subpdata}.uns['pairwise_corr']")
        if compute_sample_matrix and sample_corr_df is not None:
            _ns = int(sample_corr_df.shape[0])
            print(
                f"       • Group matrix: ({_ng} × {_ng}) | "
                f"Sample matrix: ({_ns} × {_ns})"
            )
        else:
            print(f"       • Group matrix: ({_ng} × {_ng})")

    def clean_X(self, on='prot', inplace=True, set_to=0, layer=None, to_sparse=False, backup_layer="X_preclean", verbose=True):
        """
        Replace NaNs in `.X` or a specified layer with a given value (default: 0).

        Optionally backs up the original data to a layer (default: `'X_preclean'`) before overwriting.
        Typically used to prepare data for scanpy or sklearn functions that cannot handle missing values.

        Args:
            on (str): Target data to clean, either `'protein'` or `'peptide'`.
            inplace (bool): If True, update `.X` or `.layers[layer]` in place. If False, return cleaned matrix.
            set_to (float): Value to replace NaNs with (default: 0.0).
            layer (str or None): If specified, applies to `.layers[layer]`; otherwise uses `.X`.
            to_sparse (bool): If True, returns a sparse matrix.
            backup_layer (str or None): If `inplace=True` and `layer=None`, saves the original `.X` to this layer.
            verbose (bool): Whether to print summary messages.

        Returns:
            np.ndarray: Cleaned matrix if `inplace=False`, otherwise `None`.
        """
        if not self._check_data(on):
            return
        if on == 'prot' or on == 'protein':
            adata = self.prot
        elif on == 'pep' or on == 'peptide': 
            adata = self.pep

        print(f'{format_log_prefix("user")} Cleaning {on} data: making scanpy compatible, replacing NaNs with {set_to} in {"layer " + layer if layer else ".X"}.')

        # Choose source matrix
        X = adata.layers[layer] if layer else adata.X
        is_sparse = sparse.issparse(X)

        # Copy for manipulation
        X_clean = X.copy()
        nan_count = 0

        if is_sparse:
            nan_mask = np.isnan(X_clean.data)
            nan_count = np.sum(nan_mask)
            if nan_count > 0:
                X_clean.data[nan_mask] = set_to
        else:
            nan_mask = np.isnan(X_clean)
            nan_count = np.sum(nan_mask)
            X_clean[nan_mask] = set_to

        if to_sparse and not is_sparse:
            X_clean = sparse.csr_matrix(X_clean)

        # Apply result
        if inplace:
            if layer:
                self.prot.layers[layer] = X_clean
            else:
                # Save original .X if requested and not already backed up
                if backup_layer and backup_layer not in self.prot.layers:
                    self.prot.layers[backup_layer] = self.prot.X.copy()
                    if verbose:
                        print(f"{format_log_prefix('info')} Backed up .X to .layers['{backup_layer}']")
                self.prot.X = X_clean
            if verbose:
                print(f"{format_log_prefix('result')} Cleaned {'layer ' + layer if layer else '.X'}: replaced {nan_count} NaNs with {set_to}.")
        else:
            if verbose:
                print(f"{format_log_prefix('result')} Returning cleaned matrix: {nan_count} NaNs replaced with {set_to}.")
            return X_clean 

# helper functions for analysis methods
def _annotate_pca_gsea_result_df(res_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``library`` and ``pathway`` columns from gseapy ``Term`` when using merged Enrichr
    libraries (``LIBRARY__pathway_name``). Leaves ``Term`` unchanged.
    """
    if "Term" not in res_df.columns:
        return res_df
    out = res_df.copy()
    term = out["Term"].astype(str)
    lib = term.map(lambda x: x.split("__", 1)[0] if "__" in x else "")
    pw = term.map(lambda x: x.split("__", 1)[1] if "__" in x else x)
    idx = list(out.columns).index("Term") + 1
    out.insert(idx, "library", lib.values)
    out.insert(idx + 1, "pathway", pw.values)
    return out

def _gseapy_resolve_uppercase_genes(adata, gene_col="Genes"):
    """
    For use with gseapy prerank and ssgsea functions. Resolves gene symbols from `.var[gene_col]`, uppercase them, and return as a Series.

    Hard-stops with a warning if gene names are unavailable.
    """
    if gene_col not in adata.var.columns:
        return None

    genes = adata.var[gene_col].copy()

    # Normalize empties to NaN
    genes = genes.replace("", np.nan)
    genes = genes.replace("nan", np.nan)

    # Uppercase non-missing values
    genes = genes.astype("object")
    genes = genes.where(genes.isna(), genes.astype(str).str.upper())

    return genes

def _print_duplicate_gene_warning(dup_genes, method_desc="mean"):
    """Print a warning listing duplicated gene symbols and chosen collapse method."""
    if len(dup_genes) == 0:
        return

    dup_text = ", ".join(dup_genes)
    print(f"{format_log_prefix('warn')} Found duplicated gene symbols.")
    print(f"{format_log_prefix('blank',3)} Duplicated genes ({len(dup_genes)}): {dup_text}")
    print(f"{format_log_prefix('blank',3)} Using {method_desc} for duplicated genes.")
