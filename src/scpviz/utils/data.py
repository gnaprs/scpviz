"""Data access, AnnData layers, and abundance helpers for scpviz."""
from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING, overload

import pandas as pd
import numpy as np
import anndata as ad

from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData


def parse_filename_index(
    df: pd.DataFrame,
    obs_columns: list[str],
    delimiter: str = "_",
    condition: str | None = None,
) -> pd.DataFrame:
    """
    Parse DataFrame index (filenames) into metadata columns based on a list of obs_columns. Can label a subset based on condition.

    Args:
        df (pd.DataFrame):
            DataFrame whose index contains delimited filenames.
        obs_columns (list of str):
            Names of the metadata columns to extract from the filename.
        delimiter (str):
            Character used to split the filename. Default is "_".
        condition (str or None):
            Optional boolean expression (evaluated with df.eval) that selects a
            subset of rows for parsing. If None, parse all rows. For example, `condition="parsingType == '5-tokens'"`

    Returns:
        pd.DataFrame:
            Copy of df with added metadata columns.
    """

    df_parsed = df.copy()

    if condition is None:
        mask = pd.Series(True, index=df.index)
    else:
        try:
            mask = df.eval(condition)
        except Exception as e:
            raise ValueError(f"Invalid condition '{condition}': {e}")

        if mask.dtype != bool:
            raise ValueError(f"Condition '{condition}' did not evaluate to a boolean mask.")

    # Nothing to parse
    if not mask.any():
        raise ValueError(
            f"Condition '{condition}' selected 0 rows. "
            f"Check that the column names and values in the condition are correct."
        )

    # Subset of filenames to parse
    idx_to_parse = df.index[mask]

    # Split index by delimiter
    parts = idx_to_parse.to_series().str.split(delimiter, expand=True)

    # Validate number of parts
    expected = len(obs_columns)
    actual = parts.shape[1]
    if actual != expected:
        raise ValueError(
            f"Expected {expected} parts after splitting index by '{delimiter}', "
            f"but got {actual}. Index example: '{idx_to_parse[0]}'"
        )

    # Assign parsed components
    for i, col in enumerate(obs_columns):
        # Create column if missing
        if col not in df_parsed.columns:
            df_parsed[col] = pd.NA
        # Fill only selected rows
        df_parsed.loc[mask, col] = parts.iloc[:, i].values

    return df_parsed

# ----------------
# DATA PROCESSING FUNCTIONS
# NOTE: get_samplenames and get_classlist are very similar, may want to explain better the difference (classlist is basically samplenames.unique?)
def get_samplenames(
    adata: ad.AnnData, classes: str | list[str] | None
) -> list[str] | None:
    """
    Retrieve sample names for specified class values.

    This function resolves `.obs` metadata into sample-level identifiers
    (one name per row). It is typically used for plotting functions where
    sample names are required for labeling or grouping.

    Args:
        adata (anndata.AnnData): AnnData object containing sample metadata.

        classes (str or list of str): Column(s) in `.obs` used to build sample names.

            - str: return vlaues from a single column.
            - list of str: combine multiple columns per row with `", "`.

    Returns:
        sample_names (list of str): Sample names dervied from `.obs`.

    Example:
        Get sample names from a single metadata column:
            ```python
            samples = get_samplenames(adata, "cell_type")
            ```

        Combine multiple columns into sample identifiers:
            ```python
            samples = get_samplenames(adata, ["cell_type", "treatment"])
            ```

    Related Functions:
        get_classlist: Return unique class values (not per-sample names).
    """
    if classes is None:
        return None
    elif isinstance(classes, str):
        return adata.obs[classes].values.tolist()
    elif isinstance(classes, list):
        return adata.obs[classes].apply(lambda row: ', '.join(row.values.astype(str)), axis=1).values.tolist()
    else:
        raise ValueError("Invalid input for 'classes'. It should be None, a string, or a list of strings.")
    
def get_classlist(
    adata: ad.AnnData,
    classes: str | list[str] | None = None,
    order: list[str] | None = None,
) -> list[str]:
    """
    Retrieve unique class values for specified metadata columns. Useful 
    for plot legends.

    Unlike `get_samplenames`, which returns one identifier per row/sample,
    this function extracts the set of unique class values for grouping
    purposes (e.g., plotting categories). Supports optional reordering.

    Args:
        adata (anndata.AnnData): AnnData object containing sample metadata.

        classes (str or list of str, optional): Column(s) in `.obs` to use.
            
            - None: combine all metadata columns up to the first `_quant` column.  
            - str: return unique values from one column.  
            - list of str: return unique combined values across multiple columns.  

        order (list of str, optional): Custom order of categories. Must exactly
            match the unique values; otherwise, a `ValueError` is raised.

    Returns:
        class_list (list of str): Unique class values in `.obs`, optionally reordered.

    Raises:
        ValueError: If invalid columns are provided, or if `order` does not
        match the unique class list.

    Example:
        Get unique values from one metadata column:
            ```python
            classes = get_classlist(adata, classes="cell_type")
            ```

        Combine two columns and return unique class labels:
            ```python
            classes = get_classlist(adata, classes=["cell_type", "treatment"])
            ```

        Reorder categories explicitly:
            ```python
            classes = get_classlist(
                adata, classes="cell_type", order=["A", "B", "C"]
                )
            ```

    Related Functions:
        get_samplenames: Return per-sample names (not unique class values).
    """

    if classes is None:
        # combine all .obs columns per row into one string
        # NOTE: might break, should use better method to filter out file-related columns
        quant_col_index = adata.obs.columns.get_loc(next(col for col in adata.obs.columns if "_quant" in col))
        selected_columns = adata.obs.iloc[:, :quant_col_index]
        classes_list = selected_columns.apply(lambda x: "_".join(x.astype(str)), axis=1).unique()
        classes = selected_columns.columns.tolist()
    elif isinstance(classes, str):
        # check if classes is one of the columns of adata.obs
        if classes not in adata.obs.columns:
            raise ValueError(f"Invalid value for 'classes'. '{classes}' is not a column in adata.obs.")
        classes_list = adata.obs[classes].unique()
    elif isinstance(classes, list):
        # check if list has length 1
        if len(classes) == 1:
            classes_list = adata.obs[classes[0]].unique()
        # check if all classes are columns of adata.obs
        else:
            if not all([c in adata.obs.columns for c in classes]):
                raise ValueError(f"Invalid value for 'classes'. Not all elements in '{classes}' are columns in adata.obs.")
            classes_list = adata.obs[classes].apply(lambda x: "_".join(x.astype(str)), axis=1).unique()
    else:
        raise ValueError("Invalid value for 'classes'. Must be None, a string or a list of strings.")

    if isinstance(classes_list, str):
        classes_list = [classes_list]
    if isinstance(order, str):
        order = [order]

    if order is not None:
        # check if order list matches classes_list
        missing_elements = set(classes_list) - set(order)
        extra_elements = set(order) - set(classes_list)
        # Print missing and extra elements if any
        if missing_elements or extra_elements:
            if missing_elements:
                print(f"Missing elements in 'order': {missing_elements}")
            if extra_elements:
                print(f"Extra elements in 'order': {extra_elements}")
            raise ValueError("The 'order' list does not match 'classes_list'.")
        # if they match, then reorder classes_list to match order
        classes_list = order

    return classes_list

def get_adata_layer(adata: ad.AnnData, layer: str) -> np.ndarray:
    """
    Safely extract layer data as dense numpy array.

    This helper returns the requested layer as a dense `numpy.ndarray`,
    ensuring compatibility for downstream operations. Supports both
    `.X` and `.layers[...]`.

    Args:
        adata (anndata.AnnData): AnnData object containing data matrices.

        layer (str): Layer key.  
            - `"X"`: return the main data matrix.  
            - any other str: return the corresponding entry from `.layers`. E.g. "X_norm"

    Returns:
        data (numpy.ndarray): Dense matrix representation of the requested layer.
    """
    if layer == "X":
        data = adata.X
    elif layer in adata.layers:
        data = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in .layers and is not 'X'.")

    return data.toarray() if hasattr(data, 'toarray') else data

def get_adata(pdata: pAnnData, on: str = "protein") -> ad.AnnData:
    """
    Retrieve the protein- or peptide-level AnnData object from a pAnnData container.

    Args:
        pdata (pAnnData): The parent pAnnData object containing both protein- and peptide-level data.

        on (str): Which data object to return.  
            - `"protein"`: return `pdata.prot`  
            - `"peptide"`: return `pdata.pep`  

    Returns:
        adata (anndata.AnnData): The requested AnnData object.
    """

    if on in ('protein','prot'):
        return pdata.prot
    elif on in ('peptide','pep'):
        return pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

def get_abundance(pdata: pAnnData | ad.AnnData, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Wrapper to extract abundance from either pAnnData or AnnData.

    This is a convenience wrapper that dispatches to the appropriate method:
    - If `pdata` is a `pAnnData` object, it calls `pdata.get_abundance()`.
    - If `pdata` is an `AnnData` object, it falls back to the internal
      helper `_get_abundance_from_adata`.

    Args:
        pdata (pAnnData or anndata.AnnData): Input object to extract abundance from.
        *args: Positional arguments forwarded to `get_abundance`.
        **kwargs: Keyword arguments forwarded to `get_abundance`.

    Note:
        See `pAnnData.get_abundance` for full parameter documentation. Briefly,

            - namelist (list of str, optional): List of accessions or gene names to extract.
            - layer (str): Data layer name (default = "X").
            - on (str): "protein" or "peptide".
            - classes (str or list of str, optional): Sample-level `.obs` column(s) to include.
            - log (bool): If True, applies log2 transform to abundance values.
            - x_label (str): Label features by "gene" or "accession".

    Returns:
        df (pandas.DataFrame): Long-form abundance DataFrame, optionally with
        sample metadata and protein/peptide annotations.

    See Also:
        - :func:`pAnnData.get_abundance` (EditingMixin): Full-featured version with detailed docs.
        - get_adata_layer: Helper to access abundance matrices from AnnData layers.
    """
    if hasattr(pdata, "get_abundance"):
        return pdata.get_abundance(*args, **kwargs)
    import scpviz.utils as _u

    return _u._get_abundance_from_adata(pdata, *args, **kwargs)

def _get_abundance_from_adata(
    adata: ad.AnnData,
    namelist: list[str] | str | None = None,
    layer: str = "X",
    log: bool = True,
    x_label: str = "gene",
    classes: str | list[str] | None = None,
    gene_col: str = "Genes",
    all_matches: bool = False,
) -> pd.DataFrame:
    """
    Abundance extraction for plain AnnData, including gene/accession support.
    """

    if namelist is not None:
        if isinstance(namelist, str):
            namelist = [namelist]

    # Resolve gene names → accessions
    if namelist:
        resolved = resolve_accessions(adata, namelist, gene_col=gene_col, all_matches=all_matches)
        adata = adata[:, resolved]

    # Extract matrix
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names).reset_index()
    df = df.melt(id_vars="index", var_name="accession", value_name="abundance")
    df = df.rename(columns={"index": "cell"})

    df = df.merge(adata.obs.reset_index(), left_on="cell", right_on="index")

    gene_map = adata.var["Genes"].to_dict() if "Genes" in adata.var else {}
    df['gene'] = df['accession'].map(gene_map)
    df['x_label_name'] = df['gene'].fillna(df['accession']) if x_label == 'gene' else df['accession']

    if classes:
        df['class'] = df[classes] if isinstance(classes, str) else df[classes].astype(str).agg('_'.join, axis=1)
    else:
        df['class'] = 'all'

    if log:
        df['log2_abundance'] = np.log2(np.clip(df['abundance'], 1e-6, None))

    return df

def resolve_accessions(
    adata: ad.AnnData | pAnnData,
    namelist: list[str],
    gene_col: str = "Genes",
    gene_map: dict[str, str] | None = None,
    *,
    all_matches: bool = False,
    on_empty: str = "raise",
    quiet: bool = False,
) -> list[str] | None:
    """
    Resolve gene or accession names to accession IDs from `.var_names`.

    This function maps user-specified identifiers (gene names or accession IDs)
    to the canonical accession IDs in an AnnData or pAnnData object. It first
    checks `.var_names` for exact matches, then optionally resolves gene names
    via a specified column (default `"Genes"`). Unmatched names are reported.

    Args:
        adata (AnnData or pAnnData): AnnData-like object containing `.var`.
        namelist (list of str): Input identifiers to resolve (genes or accessions).
        gene_col (str): Column in `.var` containing gene names (default: `"Genes"`).
        gene_map (dict, optional): Precomputed mapping of gene → accession. If None,
            a mapping is constructed from `gene_col`. Used only when ``all_matches=False``.
        all_matches (bool): If False (default), each gene name resolves to a single
            accession (last duplicate wins in ``gene_map``). If True, returns every
            accession whose ``gene_col`` value equals the name; logs ``info`` when
            a name expands to more than one accession.
        on_empty (str): ``"raise"`` (default) raises ``ValueError`` when nothing resolves;
            ``"return"`` returns an empty list.
        quiet (bool): If True, suppress ``info`` / ``warn`` print messages (for callers that
            embed notes in their own log output).

    Returns:
        resolved (list of str): List of accession IDs corresponding to the input names.

    Raises:
        ValueError: If none of the provided names can be resolved and ``on_empty="raise"``.

    Example:
        Resolve gene symbols to accession IDs:
            ```python
            accs = resolve_accessions(adata, namelist=["UBE4B", "GAPDH"])
            ```

        Resolve all isoforms for a gene:
            ```python
            accs = resolve_accessions(adata, namelist=["GAPDH"], all_matches=True)
            ```

        Resolve accessions directly:
            ```python
            accs = resolve_accessions(adata, namelist=["P12345", "Q67890"])
            ```

    Related Functions:
        - get_gene_maps: Build full accession → gene mapping dictionaries.
        - get_abundance: Extract abundance values by gene or accession.
    """
    import pandas as pd

    if not namelist:
        return None

    if on_empty not in ("raise", "return"):
        raise ValueError(f"on_empty must be 'raise' or 'return', got {on_empty!r}")

    var_names = adata.var_names.astype(str)

    # Use passed-in gene_map or build one (one-to-one mode only)
    if gene_map is None and not all_matches:
        gene_map = {}
        if gene_col in adata.var.columns:
            for acc, gene in zip(var_names, adata.var[gene_col]):
                if pd.notna(gene):
                    gene_map[str(gene)] = acc

    resolved, unmatched = [], []
    for name in namelist:
        name = str(name)
        if name in var_names:
            resolved.append(name)
        elif all_matches and gene_col in adata.var.columns:
            matches = [
                acc for acc, gene in zip(var_names, adata.var[gene_col])
                if pd.notna(gene) and str(gene) == name
            ]
            if matches:
                resolved.extend(matches)
                if len(matches) > 1 and not quiet:
                    print(
                        f"{format_log_prefix('info')} "
                        f"'{name}' resolved to multiple accessions: {matches}"
                    )
            else:
                unmatched.append(name)
        elif gene_map is not None and name in gene_map:
            resolved.append(gene_map[name])
        else:
            unmatched.append(name)

    if not resolved:
        if on_empty == "return":
            return []
        raise ValueError(
            f"No valid names found in `namelist`: {namelist}.\n"
            f"Check against .var_names or '{gene_col}' column."
        )

    if unmatched and not quiet:
        print(f"{format_log_prefix('warn')} A match was not found for the following:")
        for u in unmatched:
            print(f"  - {u}")

    return resolved

def resolve_peptides(
    pdata: pAnnData,
    namelist: list[str],
    *,
    all_matches: bool = True,
    on_empty: str = "return",
    quiet: bool = False,
) -> list[str] | None:
    """
    Resolve peptide IDs from ``.pep.var_names``, or expand gene/accession names to peptides.

    Direct peptide IDs in ``namelist`` are returned as-is. Otherwise each name is resolved
    to protein accessions (``resolve_accessions`` with ``all_matches``) and linked peptides
    are collected via the RS matrix when available, or via the peptide-to-protein mapping
    column otherwise.

    Args:
        pdata (pAnnData): Object with ``.pep``; protein expansion requires ``.prot``.
        namelist (list of str): Peptide IDs, gene names, or protein accessions.
        all_matches (bool): When expanding gene names to proteins, return every isoform
            (default True).
        on_empty (str): ``"raise"`` or ``"return"`` (empty list) when nothing resolves.
        quiet (bool): If True, suppress ``info`` print messages.

    Returns:
        list of str | None: Unique ``.pep.var_names`` entries, or ``None`` for empty input.
    """
    if not namelist:
        return None

    if on_empty not in ("raise", "return"):
        raise ValueError(f"on_empty must be 'raise' or 'return', got {on_empty!r}")

    if pdata.pep is None:
        raise ValueError("Peptide data (.pep) is not available.")

    pep_names = set(pdata.pep.var_names.astype(str))
    resolved: list[str] = []

    for name in namelist:
        name = str(name)
        if name in pep_names:
            resolved.append(name)
            continue

        if pdata.prot is None:
            continue

        accs = resolve_accessions(
            pdata.prot, [name], all_matches=all_matches, on_empty="return", quiet=quiet
        )
        if not accs:
            continue

        if pdata.rs is not None:
            pep_df = get_peptides_for_accessions(pdata, accs)
            matches = pep_df["peptide_id"].astype(str).tolist()
        else:
            prot_col = get_pep_prot_mapping(pdata)
            acc_set = set(accs)
            matches = []
            for pep_id, prot_val in zip(pdata.pep.var_names, pdata.pep.var[prot_col]):
                if pd.isna(prot_val):
                    continue
                linked = {p.strip() for p in str(prot_val).split(";")}
                if linked & acc_set:
                    matches.append(str(pep_id))

        if matches:
            resolved.extend(matches)
            if len(matches) > 1 and not quiet:
                print(
                    f"{format_log_prefix('info')} "
                    f"'{name}' resolved to multiple peptides: {matches}"
                )

    resolved = list(dict.fromkeys(resolved))

    if not resolved:
        if on_empty == "return":
            return []
        raise ValueError(
            f"No valid peptides found in `namelist`: {namelist}.\n"
            "Check against .pep.var_names or provide a gene/accession with linked peptides."
        )

    return resolved


@overload
def get_pep_prot_mapping(pdata: pAnnData, return_series: Literal[False] = False) -> str: ...


@overload
def get_pep_prot_mapping(pdata: pAnnData, return_series: Literal[True]) -> pd.Series: ...


def get_pep_prot_mapping(
    pdata: pAnnData, return_series: bool = False
) -> str | pd.Series:
    """
    Retrieve the peptide-to-protein mapping column or mapping values.

    This function resolves the appropriate `.pep.var` column for peptide-to-protein
    mapping based on the data source recorded in `pdata.metadata["source"]`.

    Args:
        pdata (pAnnData): The annotated proteomics object containing `.metadata` and `.pep`.
        return_series (bool): If True, return a pandas Series of peptide-to-protein
            mappings. If False (default), return the column name as a string.

    Returns:
        col (str): Column name in `.pep.var` containing peptide-to-protein mapping,
        if `return_series=False`.
        mapping (pandas.Series): Series mapping peptides to proteins,
        if `return_series=True`.

    Raises:
        ValueError: If the data source is unrecognized or no valid mapping column is found.

    Note:
        The mapping column depends on the import source:
        
        - Proteome Discoverer → `"Master Protein Accessions"`
        - DIA-NN → `"Protein.Group"`
        - MaxQuant → `"Leading razor protein"`
    """
    source = pdata.metadata.get("source", "").lower()

    if source == "proteomediscoverer":
        col = "Master Protein Accessions"
    elif source == "diann":
        col = "Protein.Group"
    elif source == "maxquant":
        col = "Leading razor protein"
    else:
        raise ValueError(f"Unknown data source '{source}' — cannot determine peptide-to-protein mapping.")

    if return_series:
        return pdata.pep.var[col]

    return col

_PEPTIDE_SEQUENCE_SEARCH_COLS = ("Stripped.Sequence", "Modified.Sequence", "Annotated Sequence")

def get_peptides_for_accessions(
    pdata: pAnnData,
    accessions: list[str],
    *,
    sequence_from: str = "index",
) -> pd.DataFrame:
    """
    Map protein accessions (or gene names) to observed peptides via the RS matrix.

    Args:
        pdata (pAnnData): Object with ``.prot``, ``.pep``, and ``.rs``.
        accessions (list of str): Protein accession IDs or gene names.
        sequence_from (str): Column in ``.pep.var`` for the ``sequence`` output,
            or ``"index"`` (default) to use ``.pep.var_names`` (precursor IDs for
            DIA-NN; annotated sequences for Proteome Discoverer).

    Returns:
        pd.DataFrame: Columns ``accession``, ``peptide_id``, ``sequence``.
            One row per accession–peptide pair. Unmatched inputs are skipped with
            a warning.

    Raises:
        ValueError: If ``.prot``, ``.pep``, or ``.rs`` is missing, or if
            ``sequence_from`` is not ``"index"`` and the column is absent.

    Example:
        Map accessions to observed peptides (default ``sequence`` = ``.pep.var_names``):
            ```python
            from scpviz import utils as scutils

            df = scutils.get_peptides_for_accessions(pdata, ["P34932"])
            df.head()
            #   accession  peptide_id  sequence
            # 0   P34932    ...         ...
            ```

        Resolve gene names instead of accessions:
            ```python
            df = scutils.get_peptides_for_accessions(pdata, ["HSPA4"])
            ```

        DIA-NN: return amino-acid strings instead of precursor IDs:
            ```python
            df = scutils.get_peptides_for_accessions(
                pdata,
                ["P34932"],
                sequence_from="Stripped.Sequence",
            )
            ```

        Proteome Discoverer: use annotated sequence column explicitly:
            ```python
            df = scutils.get_peptides_for_accessions(
                pdata,
                ["P34932"],
                sequence_from="Annotated Sequence",
            )
            ```

    Note:
        With ``sequence_from="index"``, DIA-NN datasets return precursor IDs in
        both ``peptide_id`` and ``sequence``; use ``"Stripped.Sequence"`` or
        ``"Modified.Sequence"`` for amino-acid strings.

    Related Functions:
        - get_accessions_for_peptides: Reverse lookup (peptide → accession).
        - resolve_accessions: Resolve gene or accession names to protein ``.var_names``.
        - filter_prot: Filter ``pAnnData`` by accession list (also syncs peptides via RS).
    """
    columns = ["accession", "peptide_id", "sequence"]
    if not accessions:
        return pd.DataFrame(columns=columns)

    if pdata.prot is None or pdata.pep is None or pdata.rs is None:
        raise ValueError(
            "get_peptides_for_accessions requires protein, peptide, and RS data."
        )

    if sequence_from != "index" and sequence_from not in pdata.pep.var.columns:
        available = ", ".join(map(str, pdata.pep.var.columns))
        raise ValueError(
            f"Column '{sequence_from}' not found in .pep.var. Available: {available}"
        )

    rs = pdata.rs
    pep_names = np.array(pdata.pep.var_names)
    prot_var_names = pdata.prot.var_names.astype(str)
    gene_to_acc, _ = pdata.get_identifier_maps(on="protein")

    rows = []
    unmatched = []
    for query in accessions:
        query = str(query)
        if query in prot_var_names.values:
            accession = query
        elif query in gene_to_acc:
            accession = gene_to_acc[query]
        else:
            unmatched.append(query)
            continue

        prot_idx = prot_var_names.get_loc(accession)
        pep_idx = rs[prot_idx, :].nonzero()[1]
        for j in pep_idx:
            peptide_id = pep_names[j]
            if sequence_from == "index":
                sequence = peptide_id
            else:
                sequence = pdata.pep.var.iloc[j][sequence_from]
            rows.append({"accession": accession, "peptide_id": peptide_id, "sequence": sequence})

    if unmatched:
        print(f"{format_log_prefix('warn')} A match was not found for the following:")
        for u in unmatched:
            print(f"  - {u}")

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["accession", "peptide_id"], kind="mergesort")
        .reset_index(drop=True)
    )

def get_accessions_for_peptides(
    pdata: pAnnData,
    peptides: list[str],
    *,
    sequence_from: str = "index",
) -> pd.DataFrame:
    """
    Map peptides to protein accessions via the RS matrix.

    Peptide inputs may be ``.pep.var_names`` or amino-acid strings matched
    against ``Stripped.Sequence``, ``Modified.Sequence``, or ``Annotated Sequence``.

    Args:
        pdata (pAnnData): Object with ``.prot``, ``.pep``, and ``.rs``.
        peptides (list of str): Peptide IDs or sequence strings.
        sequence_from (str): Column in ``.pep.var`` for the ``sequence`` output,
            or ``"index"`` (default) to use ``.pep.var_names``.

    Returns:
        pd.DataFrame: Columns ``peptide_id``, ``accession``, ``sequence``.
            One row per peptide–accession pair (shared peptides yield multiple rows).
            Unmatched inputs are skipped with a warning.

    Raises:
        ValueError: If ``.prot``, ``.pep``, or ``.rs`` is missing, or if
            ``sequence_from`` is not ``"index"`` and the column is absent.

    Example:
        Map peptide IDs (``.pep.var_names``) to protein accessions:
            ```python
            from scpviz import utils as scutils

            pep_id = pdata.pep.var_names[0]
            df = scutils.get_accessions_for_peptides(pdata, [pep_id])
            df.head()
            #   peptide_id  accession  sequence
            # 0  ...         P34932     ...
            ```

        DIA-NN: look up by stripped amino-acid sequence (may match multiple precursors):
            ```python
            seq = pdata.pep.var["Stripped.Sequence"].iloc[0]
            df = scutils.get_accessions_for_peptides(
                pdata,
                [seq],
                sequence_from="Stripped.Sequence",
            )
            ```

        Round-trip from accession → peptide → accession:
            ```python
            acc = pdata.prot.var_names[0]
            peps = scutils.get_peptides_for_accessions(pdata, [acc])
            prots = scutils.get_accessions_for_peptides(
                pdata, [peps.iloc[0]["peptide_id"]]
            )
            assert acc in prots["accession"].values
            ```

    Note:
        Shared peptides linked to multiple proteins in the RS matrix return one
        row per accession. A single sequence string can also match multiple
        precursor IDs when resolved from ``.pep.var`` columns.

    Related Functions:
        - get_peptides_for_accessions: Reverse lookup (accession → peptide).
        - get_pep_prot_mapping: Column name for peptide-to-protein metadata in ``.pep.var``.
    """
    columns = ["peptide_id", "accession", "sequence"]
    if not peptides:
        return pd.DataFrame(columns=columns)

    if pdata.prot is None or pdata.pep is None or pdata.rs is None:
        raise ValueError(
            "get_accessions_for_peptides requires protein, peptide, and RS data."
        )

    if sequence_from != "index" and sequence_from not in pdata.pep.var.columns:
        available = ", ".join(map(str, pdata.pep.var.columns))
        raise ValueError(
            f"Column '{sequence_from}' not found in .pep.var. Available: {available}"
        )

    rs = pdata.rs
    pep_var_names = pdata.pep.var_names.astype(str)
    prot_names = np.array(pdata.prot.var_names)

    rows = []
    unmatched = []
    for query in peptides:
        query = str(query)
        matched_ids = []
        if query in pep_var_names.values:
            matched_ids = [query]
        else:
            for col in _PEPTIDE_SEQUENCE_SEARCH_COLS:
                if col not in pdata.pep.var.columns:
                    continue
                hits = pep_var_names[pdata.pep.var[col].astype(str) == query]
                if len(hits):
                    matched_ids = hits.tolist()
                    break

        if not matched_ids:
            unmatched.append(query)
            continue

        for peptide_id in matched_ids:
            pep_idx = pep_var_names.get_loc(peptide_id)
            prot_idx = rs[:, pep_idx].nonzero()[0]
            if sequence_from == "index":
                sequence = peptide_id
            else:
                sequence = pdata.pep.var.loc[peptide_id, sequence_from]
            for i in prot_idx:
                rows.append(
                    {
                        "peptide_id": peptide_id,
                        "accession": prot_names[i],
                        "sequence": sequence,
                    }
                )

    if unmatched:
        print(f"{format_log_prefix('warn')} A match was not found for the following:")
        for u in unmatched:
            print(f"  - {u}")

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["peptide_id", "accession"], kind="mergesort")
        .reset_index(drop=True)
    )

def update_layer_provenance(
    adata: ad.AnnData,
    layer_name: str,
    op: str,
    input_layer: str,
    **kwargs: Any,
) -> str:
    """
    Register a layer in the provenance registry stored in ``adata.uns``.

    Preprocessing methods (``normalize``, ``impute``, ``log_transform``) call this
    before assigning ``adata.layers[...]``. Chains are reconstructable by following
    ``input_layer`` pointers.

    If ``layer_name`` already exists with a different ``op`` or ``input_layer``,
    a warning is printed and the record is stored under ``layer_name_1``, ``layer_name_2``, …

    Args:
        adata: AnnData to update (must not rely on pAnnData ``.history``; registry
            lives only in ``adata.uns``).
        layer_name: Intended output layer key.
        op: One of ``\"normalize\"``, ``\"impute\"``, ``\"log_transform\"``.
        input_layer: Source layer name, or ``\"X\"`` if read from ``adata.X``.
        **kwargs: Extra metadata (e.g. ``method=...``, ``base=...``).

    Returns:
        Actual layer key to use in ``adata.layers`` (may be suffixed on collision).
    """
    if "layer_provenance" not in adata.uns:
        adata.uns["layer_provenance"] = {}

    registry = adata.uns["layer_provenance"]
    new_record = {"op": op, "input_layer": input_layer, **kwargs}

    if layer_name in registry:
        existing = registry[layer_name]
        collision = (
            existing.get("input_layer") != input_layer or existing.get("op") != op
        )
        if collision:
            suffix_n = 1
            candidate = f"{layer_name}_{suffix_n}"
            while candidate in registry:
                suffix_n += 1
                candidate = f"{layer_name}_{suffix_n}"

            print(
                f"{format_log_prefix('warn')} Layer '{layer_name}' already exists "
                f"in the provenance registry with a different origin:\n"
                f"       existing: {existing}\n"
                f"       new:      {new_record}\n"
                f"     Storing new layer as '{candidate}' to avoid collision.\n"
                f"     Use pdata.show_layer_provenance('{layer_name}') to inspect "
                "the existing chain."
            )
            layer_name = candidate

    registry[layer_name] = new_record
    return layer_name

def infer_layer_is_log(layer: str, adata: Optional[ad.AnnData] = None) -> bool:
    """
    Infer whether a layer contains log-transformed values.

    1. **Registry** (if ``adata`` is given and ``adata.uns['layer_provenance']`` exists):
       walk ancestors via ``input_layer`` (cycle-safe). If any step has
       ``op == \"log_transform\"``, return True. If ``layer`` is registered and no
       ``log_transform`` appears, return False.
    2. **Name fallback**: ``\"log\" in layer.lower()``.

    Standalone ``AnnData`` objects (e.g. passed into low-level ``utils`` helpers)
    often have no ``layer_provenance`` and no pAnnData ``.history``; only the
    name heuristic applies unless you populate ``uns['layer_provenance']`` yourself.

    Args:
        layer: Layer name to inspect.
        adata: Optional AnnData carrying ``layer_provenance``.

    Returns:
        True if the layer is treated as log-transformed.
    """
    if adata is not None:
        registry = adata.uns.get("layer_provenance", {})
        visited: set[str] = set()
        current: str = layer
        while current in registry and current not in visited:
            visited.add(current)
            record = registry[current]
            if record.get("op") == "log_transform":
                return True
            nxt = record.get("input_layer", "")
            if not nxt:
                break
            current = nxt
        if layer in registry:
            return False

    return "log" in layer.lower()

def resolve_input_layer(adata: ad.AnnData, layer: str) -> str:
    """
    Resolve the source layer name for provenance when the user passes ``layer='X'``.

    The active matrix ``.X`` tracks its logical source in ``adata.uns['current_X_layer']``
    (maintained by ``set_X()`` and set at import). For any other ``layer`` string,
    return it unchanged.

    If ``current_X_layer`` is missing (legacy objects), falls back to ``\"X_raw\"``.
    """
    if layer == "X":
        return adata.uns.get("current_X_layer", "X_raw")
    return layer
