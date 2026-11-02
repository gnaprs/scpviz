# TODO

**Priority key:** H = high | M = medium | L = backlog  
**Status key:** [ ] = todo | [~] = in progress | [x] = done

H4C1: add to histone housekeeping list

---

## High priority (H)

Quick view of what to work on. Full tables below.
*Regenerate the H table from sections below when priorities change.*

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | base-1 | Allow users to input "obs" for their own metadata, own transformer for pdata.neighbor |
| H | [ ] | base-4 | Make wildcard '*' for plotting volcano plots — pair w search annotation? |
| H | [~] | viz-3 | Add gsea — one for PCA, one for ssGSEA |
| H | [ ] | ref-4 | Docstring hover: hovering over functions doesn't show docstring all the time (pdata class inherits mixin; .plotting/.utils are fine) — fix ASAP |
| H | [ ] | bug-6 | Parquet import: verify peptide-to-protein splicing (ProtA; ProtB). Maybe apply directlfq fix to import? |
| H | [x] | doc-1 | Tutorial: integrate with scanpy features |
| H | [~] | mod-5 | Web app for other users — check out dash-webapp branch by Baiyi |

---

## Features

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | base-1 | Allow users to input "obs" for their own metadata, own transformer for pdata.neighbor |
| L | [ ] | base-2 | Add sharedPeptides function on get_CV() |
| L | [ ] | base-3 | Concating multiple datasets? — see ad.concat, maybe use adata.obs_names_make_unique() |
| H | [ ] | base-4 | Make wildcard '*' for plotting volcano plots — pair w search annotation? |
| L | [ ] | viz-1 | Dot plot: expression of indicated genes in three clusters (dot size = % cells per cluster; color = cluster average normalized expression). Partially doable w scanpy. |
| M | [ ] | viz-2 | Compare two comparisons: log(fc) vs log(fc) with significance coloring |
| H | [x] | viz-3 | Add gsea — one for PCA, one for ssGSEA |
| M | [x] | viz-4 | Hypothesis testing — ANOVA with Tukey and BH correction |
| M | [x] | viz-5 | plot_grouped_heatmap + plot_clustered_heatmap (publication heatmaps) |
| M | [ ] | enr-1 | String DB values rank API — check API key; per-user or local copy in package? [API help](https://string-db.org/cgi/help?subpage=api%23valuesranks-enrichment-api) |
| L | [ ] | feat-qc-1 | Consider QC metrics beyond what we already have |
| H | [x] | base-5 | Save/load: `pdata.save()` / `pAnnData.load()` (single `.pdata` HDF5 file, lossless, per-value pickle fallback) + `pdata.export_h5ad(on=...)`; `.metadata` moved to object level (`_metadata`). Encoding in `utils/serialization.py` |

---

## Enhancements

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| M | [ ] | enh-1 | When updating .summary/obs, move prot/pep details to the right (prioritize metadata). Affects users, fix soon. |
| L | [x] | enh-2 | get_pca_importance: accept pdata input (and prot/pep) — get uns['pca'] and var_names from it. Pair with PCA GSEA? |
| L | [ ] | enh-3 | Double-check peptide export format: Gene Name, Peptide name, peptide AA start/end in protein, Charge, sample-name columns for intensities |
| L | [ ] | enh-4 | Check for DIA/DDA and suggest preprocessing methods? |
| M | [ ] | enh-5 | Fuzzy match for get_abundance matches |
| M | [ ] | enh-6 | Move Found In and Significant In from var/obs? Too many columns — or express more concisely (e.g. .obsm/.varm) |
| L | [ ] | enh-7 | Add modification print for diann import |
| L | [x] | enh-8 | When impute errors on wrong obs column, pretty-format the error message |
| L | [ ] | enh-9 | Standardize history logging (datetime stamp, formatting) |

---

## Refactor & code quality

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| M | [~] | ref-1 | Sync DE for adata and pdata into a combined workflow, and volcano code |
| L | [ ] | ref-2 | Standardize internal terminology: `classes` vs `class_types` for sample-level grouping |
| L | [ ] | ref-3 | Add typing to variables — low priority, should be easy |
| H | [ ] | ref-4 | Docstring hover: hovering over functions doesn't show docstring all the time (pdata class inherits mixin; .plotting/.utils are fine) — fix ASAP |
| L | [ ] | ref-5 | Move CSV exports (`export`, `export_layer`, `export_morpheus`) from `EditingMixin` to `IOMixin` so all persistence (import, save/load, `export_h5ad`, CSV) lives in one place; update `CODEBASE_PRIMER.md` §Export + §10 task map |
| M | [ ] | ref-6 | Move `pdata.stats` into per-modality `.prot.uns` / `.pep.uns` (e.g. `uns["scpviz_stats"]`), keeping `pdata.stats` as a read-through view for back-compat. Fixes key collisions: `mixed_de` keys omit `on` (protein vs peptide runs overwrite), `stats["imputation"]["pimms"]` not modality-tagged. ~60 call sites (DE, mixed_de, enrichment, volcano, clustering, `get_protein_clusters`, `__repr__`) — needs dedicated DE regression tests. Bump `.pdata` schema_version + migrate in `load()` |

---

## Bugs & known fixes

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| L | [ ] | bug-1 | Clustermap bug when linkage |
| M | [x] | bug-2 | plot_abundnace_housekeeping throws error when no housekeeping gene is found |
| L | [ ] | bug-3 | Verify we sync rs and filter rs matrix for every filter operation (not only filter sample by condition?) |
| H | [ ] | bug-4 | Parquet import: verify peptide-to-protein splicing (ProtA; ProtB). Maybe apply directlfq fix to import? |
| M | [ ] | bug-5 | History logging inconsistencies (found during save/load testing): raw `self._history.append(...)` (no timestamp) instead of `_append_history` in `set_X` (`editing.py:72`), `annotate_found` / `annotate_significant` (`filtering.py:2084`, `:2182`), `cv` (`analysis.py:91`), `rank` (`:1033`), `impute` (`:1334`), and reference-feature `normalize` (`:2512`, `:2517`; also no `{on}:` prefix) instead of `_append_history`; mixed `prot:` / `protein:` / `{on}:` prefixes across methods; `copy()` appends a "Pushed summary values back to obs" entry (summary setter → `update_summary(sync_back=True)`), so every `return_copy=True` operation logs it |
| M | [x] | bug-6 | `leiden()` (and possibly other `.obs` writers) adds `obs['leiden']` without refreshing `.summary` or marking it stale, so `.summary` is out of sync until the next `update_summary()`; `copy()` then also pushes it into `.pep.obs`. Fixed for `leiden()`; the `copy()` push is part of bug-5 |
| L | [x] | bug-7 | `get_peptide_properties()` stores numeric columns (e.g. `pep.var['gravy']`) as object dtype mixing floats and `pd.NA`; use nullable `Float64`. Fixed with plain float64/NaN (anndata can't write nullable `Float64` to h5ad) |
| L | [x] | bug-8 | `harmony()` takes no harmonypy kwargs (docstring TODO); with small sample counts harmonypy picks `n_clusters=0` and fails (tests xfail). Pass through `nclust` / `max_iter_harmony` / `theta`. Fixed: `harmony(**kwargs)` → harmonypy |
| M | [ ] | bug-9 | py311-dev (Windows): `torch` fails with WinError 1114 (`c10.dll`) when imported after `scpviz`, so `impute(method='pimms_*')` fails in normal use; importing torch first works. Find the conflicting DLL (likely an OpenMP/MKL import in scpviz's dependency chain) |
| H | [x] | bug-10 | `filter_prot(..., return_copy=False)` with peptides crashes (`IndexError: boolean row index has incorrect length`): `pdata` is `self`, so `self.prot` is already filtered when passed as `original` to `_filter_sync_peptides_to_proteins` (`filtering.py:260`, `:886`), while `rs` isn't. Capture original var names before filtering. Check other in-place filters for the same pattern |
| M | [x] | bug-11 | `uns['found_metrics_*']` / `uns['significance_metrics_*']` aren't subset when features are filtered: `annotate_found` → `filter_prot` → `filter_prot_found` fails (`operands could not be broadcast together with shapes (3013,) (4924,)`, `filtering.py:491`). Subset (or reindex) the metrics in the filter path, or reindex to `var_names` when reading them |
| M | [x] | bug-12 | `pAnnData(prot=..., pep=..., rs=...)` leaves `.summary` as an empty DataFrame (`pAnnData.py:74`), so the first `copy()` (any `return_copy=True` filter, `de` with `pep_pairwise_median`) raises `Mismatch: .summary and .prot.obs have different sample indices`. Build the summary in `__init__` when `prot`/`pep` is given; workaround: `pdata.update_summary()` |



---

## Documentation

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | doc-1 | Tutorial: integrate with scanpy features |
| L | [ ] | doc-2 | QC tutorial |
| L | [ ] | doc-3 | search_annotation tutorial |
| M | [ ] | doc-4 | Finish under-construction tutorials: `imputation.md`, `de.md`, `enrichment.md` |



---

## Maintenance

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
|  | [ ] | maint-1 | Todo for maintenance goes here |

---

## New modules

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| L | [~] | mod-1 | Peptide sequence characteristics (hydrophobicity, etc.) [peptide_param module]. Can work with Baiyi. |
| L | [ ] | mod-2 | Correlation visualization [protein corr module] |
| M | [ ] | mod-3 | Omics module for comparing adata? — build on existing transcriptomics x proteomics functions |
| L | [ ] | mod-4 | Alphamap peptide mapping visualization |
| H | [~] | mod-5 | Web app for other users — check out dash-webapp branch by Baiyi |
