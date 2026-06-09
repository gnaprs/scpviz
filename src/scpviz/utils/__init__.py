"""
Utility functions for scpviz.

This package provides helper and processing functions used throughout scpviz.
Import as:

    from scpviz import utils as scutils

Submodules (for maintainers): ``formatting``, ``data``, ``class_filter``, ``id_maps``, ``stats``.

## Text / formatting

Functions:
    format_log_prefix: Return standardized log prefixes for messages.

## Data access + transformation

Functions:
    parse_filename_index: Parse sample metadata from filename columns.
    get_samplenames: Resolve sample names for given classes from ``.obs``.
    get_classlist: Return unique class values for specified ``.obs`` columns.
    get_adata_layer: Safely extract a matrix from ``.X`` or ``.layers``.
    get_adata: Retrieve the ``.prot`` or ``.pep`` AnnData from a ``pAnnData`` object.
    get_abundance: Extract abundance data from pAnnData or AnnData.
    resolve_accessions: Map gene names or accessions to ``.var_names``.
    get_pep_prot_mapping: Determine peptide-to-protein mapping column.
    get_peptides_for_accessions: Map accessions or genes to observed peptides via RS.
    get_accessions_for_peptides: Map peptide IDs or sequences to protein accessions via RS.
    update_layer_provenance: Register a matrix layer in ``adata.uns['layer_provenance']``.
    resolve_input_layer: Map ``layer='X'`` to ``uns['current_X_layer']`` for provenance.
    infer_layer_is_log: Infer log-transformed layers via provenance or name heuristic.

## Sample selection / set logic

Functions:
    format_class_filter: Standardize class/value inputs for filtering.
    filter: Legacy sample filtering (prefer ``pAnnData.filter_sample_values``).
    resolve_class_filter: Resolve class/value pairs and apply filtering.
    get_upset_contents: Build contents for UpSet plots from pAnnData.
    get_upset_query: Query features present/absent in UpSet contents.

## Identifier mappings (UniProt / STRING)

Functions:
    get_uniprot_fields_worker: Low-level UniProt REST API query function (batch up to 1024).
    get_uniprot_fields: High-level UniProt API wrapper with batching.
    standardize_uniprot_columns: Normalize UniProt column names for stable downstream use.
    get_string_mappings: Map UniProt accessions to STRING IDs (UniProt + STRING fallback).
    convert_identifiers: Convert between accession / gene / STRING / organism_id.

## Statistics

Functions:
    pairwise_log2fc: Compute pairwise median log2 fold change between groups.
    de_adata: Differential expression helper over AnnData matrices.
    get_pca_importance: Identify most important features for PCA components.
    get_protein_clusters: Retrieve hierarchical clusters from stored linkage.

!!! warning
    Many functions here are internal helpers. For common workflows (filtering, plotting, enrichment),
    prefer the corresponding ``pAnnData`` methods when available.
"""

# Match legacy ``scpviz.utils`` module: test and notebook attributes
import upsetplot
import scpviz.setup as _setup
from scpviz import pAnnData  # noqa: F401

from .formatting import *
from .data import *
from .stats import *
from .id_maps import *
from .class_filter import *

# Star-imports omit leading-underscore names; tests and notebooks access these on ``utils``.
from .data import _get_abundance_from_adata
from .id_maps import (
    _map_uniprot_field,
    _string_get_string_ids,
    _uniprot_get_string_ids,
)
