# Hidden Functions

Hidden functions for all MixIns.

!!! warning "Advanced / Internal"
    The functions in this section are internal utilities. They may change 
    without notice and are not guaranteed to remain stable across releases. 
    Use only if you understand the internal architecture of `pAnnData`.

<!-- !!! note "Why some targets are classes"
    `members` is resolved on the autodoc root. Mixin methods like
    `_has_data` live on `BaseMixin`, not on the `base` module, so those
    blocks target the class. Module-level helpers (e.g. `_import_diann`,
    `_pretty_vs_key`) still target the module. -->

---

### analysis

::: src.scpviz.pAnnData.analysis.AnalysisMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _normalize_helper
        - _normalize_helper_directlfq

::: src.scpviz.pAnnData.analysis
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _annotate_pca_gsea_result_df
        - _gseapy_resolve_uppercase_genes
        - _print_duplicate_gene_warning

### base

::: src.scpviz.pAnnData.base.BaseMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _has_data

### editing

::: src.scpviz.pAnnData.editing.EditingMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _set_RS

### enrichment

::: src.scpviz.pAnnData.enrichment
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _pretty_vs_key
        - _resolve_de_key

### filtering

::: src.scpviz.pAnnData.filtering
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _detect_ambiguous_input

::: src.scpviz.pAnnData.filtering.FilterMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _filter_sync_peptides_to_proteins
        - _filter_sample_poi
        - _filter_sample_condition
        - _filter_sample_values
        - _filter_sample_query
        - _cleanup_proteins_after_sample_filter
        - _apply_rs_filter
        - _format_filter_query
        - _annotate_found_samples
        - _annotate_significant_samples

### history

::: src.scpviz.pAnnData.history.HistoryMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _append_history

### identifier

::: src.scpviz.pAnnData.identifier.IdentifierMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _build_identifier_maps

### io

::: src.scpviz.pAnnData.io
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _import_proteomeDiscoverer
        - _import_diann
        - _safe_strip
        - _create_pAnnData_from_parts
        - _build_rs_matrix

### metrics

::: src.scpviz.pAnnData.metrics.MetricsMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _update_metrics
        - _update_summary_metrics

### pAnnData

::: src.scpviz.pAnnData.pAnnData.pAnnData
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _cached_identifier_maps_protein
        - _cached_identifier_maps_peptide

### summary

::: src.scpviz.pAnnData.summary.SummaryMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _update_summary
        - _merge_obs
        - _push_summary_to_obs
        - _mark_summary_stale

### validation

::: src.scpviz.pAnnData.validation.ValidationMixin
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - _check_data
        - _check_rankcol
