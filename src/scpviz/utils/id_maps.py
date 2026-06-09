"""UniProt, STRING, and identifier conversion helpers for scpviz."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Union

import io
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests

import scpviz.setup as _setup

from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

def get_uniprot_fields_worker(
    prot_list: list[str],
    search_fields: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query UniProt for a batch of protein accessions.

    This function sends requests to the UniProt REST API for up to 1024 proteins
    at a time and returns the requested fields as a DataFrame. It handles isoform
    accessions, fallback queries, and UniProt ID redirects automatically.

    Args:
        prot_list (list of str): List of protein accessions or IDs.
        search_fields (list of str): UniProt return fields.
            See: https://www.uniprot.org/help/return_fields
        verbose (bool): If True, print progress messages and missing accessions.

    Returns:
        df (pandas.DataFrame): DataFrame containing UniProt metadata for the input proteins.

    Raises:
        ValueError: If `query_type` is unknown or the data source cannot be resolved.

    !!! info
        - This function is intended as a **worker** and is usually called by
          `get_uniprot_fields`.
        - It automatically resolves canonical vs. isoform accessions and will
          attempt UniProt ID mapping if some accessions cannot be found.

    Related Functions:
        - get_uniprot_fields: High-level batch UniProt query wrapper.
    """

    base_url = 'https://rest.uniprot.org/uniprotkb/stream'
    if search_fields is None:
        raise ValueError("search_fields is required for UniProt queries.")
    fields = "%2C".join(search_fields)
    format_type = 'tsv'
    
    def query_uniprot_batch(ids, query_type="accession"):
        if not ids:
            return pd.DataFrame()

        if query_type == "accession":
            query_parts = [f"%28accession%3A{id}%29" for id in ids]
        elif query_type == "id":
            query_parts = [f"%28id%3A{id}%29" for id in ids]
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        query = "+OR+".join(query_parts)
        full_query = f"%28{query}%29"
        url = f'{base_url}?fields={fields}&format={format_type}&query={full_query}'

        if verbose:
            print(f"Querying UniProt ({query_type}, TSV mode) for {len(ids)} proteins")

        results = requests.get(url)
        results.raise_for_status()

        # Handle empty response gracefully
        if not results.text.strip():
            print(f"{format_log_prefix('warn_only', 2)} UniProt returned empty response for {len(ids)} proteins.")
            return pd.DataFrame()

        return pd.read_csv(io.StringIO(results.text), sep="\t")

    if verbose:
        print(f"{format_log_prefix('API', 1)} Querying UniProt for {len(prot_list)} total proteins [TSV mode].")
    
    def resolve_uniprot_redirects(accessions, from_db='UniProtKB_AC-ID', to_db='UniProtKB'):
        url = 'https://rest.uniprot.org/idmapping/run'
        data = {'from': from_db, 'to': to_db, 'ids': ','.join(accessions)}

        res = requests.post(url, data=data)
        res.raise_for_status()
        job_id = res.json()['jobId']

        # Poll until job is complete
        while True:
            status = requests.get(f"https://rest.uniprot.org/idmapping/status/{job_id}").json()
            if status.get("jobStatus") == "RUNNING":
                time.sleep(1)
            else:
                break

        # Get results
        results = requests.get(f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}").json()
        mapping = {item['from']: item['to']['primaryAccession'] for item in results.get('results', [])}
        return mapping

    # Split isoform vs canonical accessions
    isoform_ids = [acc for acc in prot_list if '-' in acc]
    canonical_ids = [acc for acc in prot_list if '-' not in acc]

    df_canonical = query_uniprot_batch(canonical_ids, query_type="accession")
    df_isoform = query_uniprot_batch(isoform_ids, query_type="accession")

    # Identify any isoforms that weren't found
    found_isoform_ids = set(df_isoform['Entry']) if not df_isoform.empty else set()
    missing_isoforms = [acc for acc in isoform_ids if acc not in found_isoform_ids]

    if missing_isoforms and verbose:
        print(f"{format_log_prefix('info_only', 3)} Attempting fallback query for {len(missing_isoforms)} isoform base IDs")

    # Attempt fallback query using base accessions
    fallback_ids = list(set([id.split('-')[0] for id in missing_isoforms]))
    df_fallback = query_uniprot_batch(fallback_ids, query_type="id")

    # Combine all DataFrames
    df = pd.concat([df_canonical, df_isoform, df_fallback], ignore_index=True)

    # Final pass: insert missing rows if still unresolved
    found_entries = set(df['Entry']) if 'Entry' in df.columns else set()
    still_missing = set(prot_list) - found_entries

    if still_missing:
        if verbose:
            print(f"{format_log_prefix('info_only', 3)} Attempting UniProt ID redirect for {len(still_missing)} unresolved accessions.")
        redirect_map = resolve_uniprot_redirects(list(still_missing))
        if redirect_map:
            redirected_ids = list(redirect_map.values())
            df_redirected = query_uniprot_batch(redirected_ids, query_type="accession")
            
            # Remap back to original accession
            inv_map = {v: k for k, v in redirect_map.items()}
            if 'Entry' in df_redirected.columns:
                df_redirected['Entry'] = df_redirected['Entry'].apply(lambda x: inv_map.get(x, x))

            df = pd.concat([df, df_redirected], ignore_index=True)

            resolved = set(redirect_map.keys())
            still_missing -= resolved

    # Step 5: Fill in placeholders for totally missing accessions
    if still_missing:
        print(f"{format_log_prefix('warn_only', 3)} Proteins not found in UniProt: {list(still_missing)[:5]}") if verbose else None
        missing_df = pd.DataFrame({'Entry': list(still_missing)})
        for col in search_fields:
            if col != 'accession' and col not in missing_df.columns:
                missing_df[col] = np.nan
        df = pd.concat([df, missing_df], ignore_index=True)
    
    if 'STRING' in df.columns:
        # keep first STRING ID (or join all if you prefer)
        df['xref_string'] = df['STRING'].apply(
            lambda s: str(s).split(';')[0].strip() if pd.notna(s) and str(s).strip() else np.nan
        )
        df.drop(columns=['STRING'], inplace=True)

    return df

def get_uniprot_fields(
    prot_list: list[str],
    search_fields: list[str] | None = None,
    batch_size: int = 100,
    verbose: bool = True,
    standardize: bool = True,
    worker_verbose: bool = False,
) -> pd.DataFrame:
    """
    Retrieve UniProt metadata for a list of protein accessions.

    This function wraps `get_uniprot_fields_worker` to handle batching of
    protein IDs, returning results as a single DataFrame.

    Args:
        prot_list (list of str): List of protein accessions.
        search_fields (list of str): UniProt fields to return.
            Defaults include accession, gene names, GO terms, and STRING IDs.
        batch_size (int): Number of accessions per batch (max 1024, default=100).
        verbose (bool): If True, print progress messages.
        standardize (bool): If True (default), normalize UniProt column names
            to canonical lowercase keys (e.g., "gene_primary", "organism_id",
            "xref_string") for consistent downstream processing.

    Returns:
        df (pandas.DataFrame): DataFrame containing UniProt metadata for the input proteins.

    Example:
        Query UniProt for a small set of proteins:
            ```python
            proteins = ["P40925", "P40926"]
            df = get_uniprot_fields(proteins)
            df[["Entry", "Gene Names", "Organism Id"]].head()
            ```

        Retrieve raw UniProt field names without renaming:
            >>> df_raw = get_uniprot_fields(proteins, standardize=False)

    Related Functions:
        - get_uniprot_fields_worker: Worker function that handles low-level UniProt API queries.
        - standardize_uniprot_columns: Helper used internally for column normalization.
    """
    import scpviz.utils as _u

    if search_fields is None:
        search_fields = [
            "accession",
            "id",
            "protein_name",
            "gene_primary",
            "gene_names",
            "organism_id",
            "go",
            "go_f",
            "go_c",
            "go_p",
            "cc_interaction",
            "xref_string",
        ]

    # --- Ensure 'accession' field comes first (UniProt requirement)
    search_fields = ["accession"] + [f for f in search_fields if f != "accession"]

    # --- Split IDs into batches
    batches = [prot_list[i:i + batch_size] for i in range(0, len(prot_list), batch_size)]
    all_results = []

    for i, batch in enumerate(batches, start=1):
        if verbose:
            print(
                f"{format_log_prefix('api', indent=2)} Querying UniProt for batch {i}/{len(batches)} "
                f"({len(batch)} proteins) [fields: {', '.join(search_fields)}]"
            )

            if len(batches) > 1:
                print(f"{format_log_prefix('info_only', indent=3)} Processing batch {i}/{len(batches)}...")

        try:
            batch_df = get_uniprot_fields_worker(batch, search_fields, verbose=worker_verbose)
            if standardize:
                batch_df = _u.standardize_uniprot_columns(batch_df)
            all_results.append(batch_df)
        except Exception as e:
            print(f"{format_log_prefix('warn')} Failed batch {i}: {e}")
            continue

    if not all_results:
        if verbose:
            print(f"{format_log_prefix('warn')} No results retrieved from UniProt.")
        return pd.DataFrame()

    full_method_df = pd.concat(all_results, ignore_index=True)
    if verbose:
        print(f"{format_log_prefix('result_only', 2)} Retrieved UniProt metadata for {len(full_method_df)} entries.")

    return full_method_df

def standardize_uniprot_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Normalize UniProt DataFrame column names to a consistent lowercase, snake_case schema.

    This ensures stability across UniProt REST API version changes while keeping
    the user informed only when critical fields are affected.

    Args:
        df (pd.DataFrame): Raw UniProt metadata table.

    Returns:
        pd.DataFrame: Copy of the DataFrame with standardized column names.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return df

    rename_map = {}
    aliases = {
        # identifiers
        "entry": "accession",
        "entry_name": "id",
        "accession": "accession",
        "primaryaccession": "accession",

        # gene fields
        "gene_names_primary": "gene_primary",
        "gene_name_primary": "gene_primary",
        "gene_primary_name": "gene_primary",
        "gene_primary": "gene_primary",
        "gene_primaryname": "gene_primary",
        "gene_primary_name_": "gene_primary",
        "gene_primaryname_": "gene_primary",

        # organism fields
        "organism_id": "organism_id",
        "organism_identifier": "organism_id",
        "organismid": "organism_id",

        # STRING / cross-reference
        "cross_reference_string": "xref_string",
        "xref_string_id": "xref_string",
        "crossreference_string": "xref_string",
        "string": "xref_string",
        "string_id": "xref_string",
        "xref_string": "xref_string",
    }

    # critical canonical fields we care about if changed or missing
    critical_fields = {"accession", "gene_primary", "organism_id", "xref_string"}

    # known benign patterns — don't warn if these change
    benign_patterns = {
        "gene_ontology",
        "go",
        "gene_names",      # non-primary gene list
        "protein_name",    # descriptive only
        "cc_interaction",  # crossref metadata
    }

    for col in df.columns:
        norm = (
            re.sub(r"[^a-z0-9]+", "_", col.lower())
            .strip("_")
            .replace("__", "_")
        )

        mapped = aliases.get(norm, None)

        if mapped:
            rename_map[col] = mapped
        else:
            # warn only if this looks like a drifted critical column
            if (
                any(k in norm for k in ["accession", "gene", "organism", "string"])
                and not any(p in norm for p in benign_patterns)
            ):
                warnings.warn(
                    f"[standardize_uniprot_columns] ⚠️ Unrecognized UniProt column '{col}' "
                    f"(normalized='{norm}') — may affect critical mapping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            rename_map[col] = norm  # keep normalized fallback name

    df = df.rename(columns=rename_map)
    # verify that all critical fields exist at least once
    missing_critical = [c for c in critical_fields if c not in df.columns]
    if missing_critical:
        if _setup.GLOBAL_DEBUG:
            warnings.warn(
                f"[standardize_uniprot_columns] Missing expected UniProt columns: {', '.join(missing_critical)}",
                RuntimeWarning,
                stacklevel=2,
            )

    return df.rename(columns=rename_map)

## STRING
def get_string_mappings(
    identifiers: list[str],
    use_uniprot: bool = True,
    use_string: bool = True,
    caller_identity: str = "scpviz",
    batch_size: int = 100,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Resolve STRING identifiers for a list of UniProt accessions.

    This function maps UniProt protein accessions to STRING IDs using a
    two-step strategy:
    
    1. **UniProt lookup** – retrieves STRING cross-references (`xref_string`)
       and organism IDs via the UniProt API (fast).
    2. **STRING API lookup** – queries the STRING `get_string_ids` endpoint
       for any identifiers not resolved via UniProt.

    Args:
        identifiers (list of str): List of UniProt accession IDs to map.
        use_uniprot (bool): If True (default), attempt mapping via UniProt
            `xref_string` and `organism_id` fields.
        use_string (bool): If True (default), query the STRING API for any
            identifiers still unresolved after the UniProt step.
        caller_identity (str): Identifier passed to the STRING API
            (default: "scpviz").
        batch_size (int): Number of identifiers per batch when querying
            external APIs (default=100).
        debug (bool): If True, print progress and debug information.

    Returns:
        pandas.DataFrame: Mapping table with one row per input identifier and
        the following columns:
        
        - `input_identifier`: UniProt accession provided as input  
        - `string_identifier`: Corresponding STRING ID (if resolved)  
        - `ncbi_taxon_id`: NCBI taxonomy ID inferred from UniProt or STRING  

    Example:
        Map a small set of UniProt accessions to STRING IDs:
            ```python
            proteins = ["P40925", "P40926"]
            df = get_string_mappings(proteins)
            df
            ```

        Disable the UniProt shortcut and query STRING directly (takes longer than UniProt):
            ```python
            df = get_string_mappings(proteins, use_uniprot=False)
            ```

    Related Functions:
        - get_uniprot_fields: Retrieve UniProt metadata, including STRING cross-references.
        - pAnnData.EnrichmentMixin (enrichment_functional(), enrichment_ppi())
    """
    import scpviz.utils as _u

    ids = [str(x).strip() for x in identifiers if x is not None and str(x).strip()]
    if not ids:
        return pd.DataFrame(columns=["input_identifier", "string_identifier", "ncbi_taxon_id"])

    found: Dict[str, str] = {}
    species_map: Dict[str, object] = {}

    # Step 1: UniProt xref_string
    uni_df = pd.DataFrame(columns=["input_identifier", "string_identifier"])
    if use_uniprot:
        try:
            uni_df, uni_species = _u._uniprot_get_string_ids(
                ids, batch_size=batch_size, standardize=True, debug=debug
            )
            if not uni_df.empty:
                found.update(dict(zip(uni_df["input_identifier"], uni_df["string_identifier"])))
            species_map.update(uni_species)

            print(f"{format_log_prefix('api',2)} UniProt mapped: {len(uni_df)} / {len(ids)}")
        except Exception as e:
            print(f"{format_log_prefix('error')} UniProt stream step failed: {e}") 

    # Missing after UniProt
    missing = [i for i in ids if i not in found]

    # Step 2: STRING get_string_ids
    string_df = pd.DataFrame(columns=["input_identifier", "string_identifier", "ncbi_taxon_id"])
    if use_string and missing:
        try:
            string_df = _u._string_get_string_ids(
                missing, batch_size=batch_size, caller_identity=caller_identity, debug=debug
            )
            if not string_df.empty:
                found.update(dict(zip(string_df["input_identifier"], string_df["string_identifier"])))

            print(f"{format_log_prefix('api',2)} STRING mapped: {len(string_df)} / {len(missing)} (missing after UniProt)")
        except Exception as e:
            print(f"{format_log_prefix('error')} STRING stream step failed: {e}") 

    # Build output table
    out_df = pd.DataFrame({"input_identifier": ids})
    out_df["string_identifier"] = out_df["input_identifier"].map(found)

    # Taxon: prefer UniProt organism_id, then STRING ncbi_taxon_id
    tax_from_uniprot = out_df["input_identifier"].map(lambda a: species_map.get(a, pd.NA))
    tax_from_uniprot = tax_from_uniprot.apply(scalarize_taxon)

    if not string_df.empty and "ncbi_taxon_id" in string_df.columns:
        string_tax_map = dict(zip(string_df["input_identifier"], string_df["ncbi_taxon_id"]))
        tax_from_string = out_df["input_identifier"].map(lambda a: string_tax_map.get(a, pd.NA))
        tax_from_string = tax_from_string.apply(scalarize_taxon)
    else:
        tax_from_string = pd.Series([pd.NA] * len(out_df), index=out_df.index)

    out_df["ncbi_taxon_id"] = tax_from_uniprot.combine_first(tax_from_string)

    return out_df

def _first_string_xref(x: object) -> Union[str, float]:
    """Parse the first STRING xref from UniProt `xref_string` (may be ';' delimited)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    return s.split(";")[0].strip()

def scalarize_taxon(x: object) -> object:
    """
    Normalize taxon-id values so they never contain lists or arrays.

    Returns:
        Scalar string-like taxon id, or pd.NA.
    """
    # Handle pandas missing scalar explicitly first
    if x is pd.NA:
        return pd.NA

    # Handle standard missing
    if x is None:
        return pd.NA
    if isinstance(x, float) and np.isnan(x):
        return pd.NA

    # Empty string
    if isinstance(x, str):
        s = x.strip()
        return pd.NA if s == "" else s

    # Empty container / container → first element
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return pd.NA
        return scalarize_taxon(x[0])

    # Everything else → string
    return str(x)

def _string_get_string_ids(identifiers: List[str], *, batch_size: int = 100, caller_identity: str = "scpviz", debug: bool = False,) -> pd.DataFrame:
    """
    Query STRING get_string_ids for identifiers.

    Returns:
        DataFrame with columns:
            input_identifier, string_identifier, ncbi_taxon_id
        (may be empty if nothing returned)
    """
    if not identifiers:
        return pd.DataFrame(columns=["input_identifier", "string_identifier", "ncbi_taxon_id"])

    url = "https://string-db.org/api/tsv-no-header/get_string_ids"
    all_rows = []

    for i in range(0, len(identifiers), batch_size):
        batch = identifiers[i : i + batch_size]
        params = {
            "identifiers": "\r".join(batch),
            "limit": 1,
            "echo_query": 1,
            "caller_identity": caller_identity,
        }

        try:
            t0 = time.time()
            r = requests.post(url, data=params, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="\t", header=None)
            if df.empty:
                continue

            df.columns = [
                "input_identifier",
                "input_alias",
                "string_identifier",
                "ncbi_taxon_id",
                "preferred_name",
                "annotation",
                "score",
            ]
            df = df[["input_identifier", "string_identifier", "ncbi_taxon_id"]]
            all_rows.append(df)

            if debug:
                dt = time.time() - t0
                print(f"{'✅'} STRING batch {i // batch_size + 1}: {len(batch)} ids in {dt:.2f}s")
        except Exception as e:
            if debug:
                print(f"{'⚠️'} STRING batch {i // batch_size + 1} failed: {e}")

    if not all_rows:
        return pd.DataFrame(columns=["input_identifier", "string_identifier", "ncbi_taxon_id"])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.dropna(subset=["input_identifier"]).drop_duplicates(subset=["input_identifier"], keep="first")
    return out

def _uniprot_get_string_ids(identifiers: List[str], *, batch_size: int = 100, standardize: bool = True, debug: bool = False,) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Query UniProt for xref_string + organism_id and return mapping rows.

    Returns:
        (df_map, species_map)
        df_map columns: input_identifier, string_identifier
        species_map: input_identifier -> organism_id
    """
    if not identifiers:
        return (
            pd.DataFrame(columns=["input_identifier", "string_identifier"]),
            {},
        )

    import scpviz.utils as _u

    dfu = _u.get_uniprot_fields(
        identifiers,
        search_fields=["xref_string", "organism_id"],
        batch_size=batch_size,
        standardize=standardize,
        verbose=debug,
    )
    if dfu is None or dfu.empty:
        return (
            pd.DataFrame(columns=["input_identifier", "string_identifier"]),
            {},
        )

    entry_col = "accession" if "accession" in dfu.columns else None
    xref_col = "xref_string" if "xref_string" in dfu.columns else None
    org_col = "organism_id" if "organism_id" in dfu.columns else None
    if entry_col is None or xref_col is None:
        return (
            pd.DataFrame(columns=["input_identifier", "string_identifier"]),
            {},
        )

    tmp = dfu[[entry_col, xref_col] + ([org_col] if org_col else [])].copy()
    tmp["string_identifier"] = tmp[xref_col].apply(_first_string_xref)
    tmp = tmp.rename(columns={entry_col: "input_identifier"})
    tmp = tmp.dropna(subset=["input_identifier", "string_identifier"])
    tmp = tmp.drop_duplicates(subset=["input_identifier"], keep="first")
    df_map = tmp[["input_identifier", "string_identifier"]].copy()

    species_map: Dict[str, object] = {}
    if org_col and org_col in tmp.columns:
        for _, row in tmp.iterrows():
            acc = row["input_identifier"]
            org_val = row[org_col]
            if pd.isna(org_val):
                continue
            s = str(org_val).strip()
            if not s:
                continue
            try:
                species_map[acc] = int(s)
            except Exception:
                species_map[acc] = s

    return df_map, species_map

# ----------------
def _map_uniprot_field(
    from_type: str, to_type: str | list[str]
) -> tuple[str, list[str], list[str]]:
    """
    Internal helper to resolve UniProt column names and required fields
    for a given identifier conversion request.

    Args:
        from_type (str): Source identifier type ('accession', 'gene').
        to_type (str or list of str): Target identifier type(s).
            Supported: 'gene', 'string', 'organism_id'.

    Returns:
        tuple: (from_col, to_cols, required_fields)
    """
    if isinstance(to_type, str):
        to_type = [to_type]

    # Validate allowed types
    valid_types = {"accession", "gene", "string", "organism_id"}
    if from_type not in valid_types:
        raise ValueError(f"Invalid from_type: '{from_type}'. Must be one of {valid_types}")
    if any(t not in valid_types for t in to_type):
        raise ValueError(f"Invalid to_type: {to_type}. Must be subset of {valid_types}")
    if from_type == "organism_id":
        raise ValueError("'organism_id' can only be used as a target (to_type).")

    field_map = {
        "accession": "accession",
        "gene": "gene_primary",
        "string": "xref_string",
        "organism_id": "organism_id",
    }

    from_col = field_map[from_type]
    to_cols = [field_map[t] for t in to_type]

    # Determine required UniProt fields for the query
    required_fields = set(["accession", from_col, *to_cols])
    if from_type == "gene":
        # Gene lookups usually need accession linkage
        required_fields |= {"accession"}

    return from_col, to_cols, list(required_fields)

def convert_identifiers(
    ids: list[str],
    from_type: str,
    to_type: str | list[str],
    pdata: pAnnData | None = None,
    use_cache: bool = True,
    return_type: str = "dict",
    verbose: bool = True,
) -> dict[str, dict[str, Any]] | pd.DataFrame | tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    Convert identifiers between UniProt-compatible types.

    Supports mapping between protein accessions, gene names, STRING IDs,
    and organism IDs. Multiple output types may be requested at once.

    Args:
        ids (list of str): Input identifiers.
        from_type (str): Source identifier type (``"accession"`` or ``"gene"``).
            ``"organism_id"`` cannot be used as a source.
        to_type (str or list of str): Target identifier type(s). May include
            any of ``"gene"``, ``"string"``, or ``"organism_id"``.
        pdata (pAnnData, optional): pAnnData object providing cached
            accession–gene mappings. If provided, ``use_cache`` is
            automatically set to True.
        use_cache (bool): Whether to use cached mappings from ``pdata``.
            Default is True.
        return_type (str): Output format. ``"dict"`` returns a nested dict
            mapping each input ID to requested fields; ``"df"`` returns a
            DataFrame with columns for ``from_type`` and all ``to_type``
            fields; ``"both"`` returns ``(dict, DataFrame)``.
        verbose (bool): Whether to print progress messages. Default is True.

    Returns:
        dict, pandas.DataFrame, or tuple: Depends on ``return_type``.

    Example:
        Map one accession to a gene name using cached ``pdata`` mappings
        (dict output):
            ```python
            from scpviz import utils as scutils

            scutils.convert_identifiers(
                ["P40925"], "accession", "gene", return_type="dict", pdata=pdata
            )
            ```

        Convert accessions to gene, STRING, and organism ID in one table—gene
        names come from ``pdata``, STRING and organism ID from UniProt:
            ```python
            scutils.convert_identifiers(
                ["P40925", "P55072", "P04637"],
                "accession",
                ["gene", "string", "organism_id"],
                return_type="df",
                pdata=pdata,
            )
            ```

        Reverse lookup: gene symbols to accession (cached) plus STRING and
        organism ID (UniProt):
            ```python
            scutils.convert_identifiers(
                ["MDH1", "VCP", "TP53"],
                "gene",
                ["accession", "string", "organism_id"],
                return_type="df",
                pdata=pdata,
            )
            ```

        Same multi-field conversion without ``pdata``—queries UniProt for
        all fields:
            ```python
            scutils.convert_identifiers(
                ["P40925", "P55072", "P04637"],
                "accession",
                ["gene", "string", "organism_id"],
                return_type="df",
            )
            ```

        Return a nested dict mapping each input ID to all requested fields:
            ```python
            scutils.convert_identifiers(
                ["P40925", "P55072"],
                "accession",
                ["gene", "string", "organism_id"],
                return_type="dict",
            )
            ```
    """
    import pandas as pd
    import numpy as np
    import scpviz.utils as _u

    if not ids:
        empty_df = pd.DataFrame(columns=[from_type] + ([to_type] if isinstance(to_type, str) else list(to_type)))
        return {} if return_type != "df" else empty_df

    if pdata is not None:
        use_cache = True

    from_col, to_cols, search_fields = _map_uniprot_field(from_type, to_type)
    if isinstance(to_type, str):
        to_type = [to_type]

    # canonical UniProt field map (consistent with standardize_uniprot_columns)
    _FIELD_MAP = {
        "accession": "accession",
        "gene": "gene_primary",
        "string": "xref_string",
        "organism_id": "organism_id",
    }

    # --- Logging
    if verbose:
        print(f"{format_log_prefix('search', indent=1)} Converting from '{from_type}' to {to_type} for {len(ids)} identifiers...")
        if pdata is not None:
            cacheable_types = {"accession", "gene"}
            api_needed = [t for t in to_type if t not in cacheable_types]
            if set([from_type] + to_type).issubset(cacheable_types):
                print(f"{format_log_prefix('info_only', indent=2)} Using cached mapping from pdata (no UniProt queries).")
            elif api_needed:
                api_list = ", ".join(api_needed)
                print(f"{format_log_prefix('info_only', indent=2)} Using cached mapping for gene/accession; UniProt lookup required for: {api_list}.")
        else:
            print(f"{format_log_prefix('info_only', indent=2)} No pdata provided — querying UniProt for all target fields.")

    # --- Tier 1: cache lookup (only accession <-> gene)
    resolved = {id_: {t: None for t in to_type} for id_ in ids}
    to_query = list(ids)

    if pdata is not None and use_cache and {"accession", "gene"}.issuperset({from_type, *to_type}):
        if from_type == "accession" and "gene" in to_type:
            _, acc_to_gene = pdata.get_identifier_maps(on="protein")
            for acc in ids:
                if acc in acc_to_gene:
                    resolved[acc]["gene"] = acc_to_gene[acc]
        elif from_type == "gene" and "accession" in to_type:
            gene_to_acc, _ = pdata.get_identifier_maps(on="protein")
            for gene in ids:
                if gene in gene_to_acc:
                    resolved[gene]["accession"] = gene_to_acc[gene]

        # Filter unmapped
        to_query = [x for x, v in resolved.items() if not any(vv for vv in v.values())]

    # --- Tier 3: UniProt API
    df = pd.DataFrame()
    if len(to_query) > 0:
        # Hybrid case: gene → STRING / organism_id (not gene → accession, which would recurse infinitely)
        if from_type == "gene" and to_type != ["accession"]:
            gene_to_acc = convert_identifiers(to_query, "gene", "accession", pdata=pdata, use_cache=use_cache, verbose=False)
            accs = [v.get("accession") for v in gene_to_acc.values() if v.get("accession")]
            if accs:
                df = _u.get_uniprot_fields(accs, search_fields=search_fields, standardize=True)
                df = _u.standardize_uniprot_columns(df)
                df = df.drop_duplicates(subset="accession", keep="first")

                # Build per-target maps
                per_target_maps = {}
                for t in to_type:
                    col = _FIELD_MAP[t]
                    if col in df.columns:
                        per_target_maps[t] = dict(zip(df["accession"], df[col]))
                    else:
                        per_target_maps[t] = {}

                # Assign results
                for g, acc_dict in gene_to_acc.items():
                    acc = acc_dict.get("accession")
                    for t in to_type:
                        resolved[g][t] = per_target_maps[t].get(acc) if acc else None
            else:
                for g in to_query:
                    for t in to_type:
                        resolved[g][t] = None

        else:
            # Direct mapping (accession → X)
            df = _u.get_uniprot_fields(to_query, search_fields=search_fields, standardize=True)

            # --- Clean up STRING results if present
            if not df.empty:
                if "xref_string" in df.columns and isinstance(df["xref_string"], pd.Series):
                    df["xref_string"] = (
                        df["xref_string"]
                        .astype(str)
                        .apply(lambda s: s.replace(";", "").strip() if isinstance(s, str) else np.nan)
                        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                    )
                elif "string" in to_type and verbose:
                    print(f"{format_log_prefix('warn_only', indent=3)} UniProt did not return 'xref_string' field — possible API schema drift.")

            if not df.empty and from_col in df.columns:
                per_target_maps = {}
                for t in to_type:
                    col = _FIELD_MAP[t]
                    if col in df.columns:
                        per_target_maps[t] = dict(zip(df[from_col], df[col]))
                    else:
                        per_target_maps[t] = {}

                for id_ in to_query:
                    for t in to_type:
                        resolved[id_][t] = per_target_maps[t].get(id_)
            else:
                for id_ in to_query:
                    for t in to_type:
                        resolved[id_][t] = None

    # --- Reporting
    resolved_count = sum(
        any(vv is not None and not pd.isna(vv) for vv in v.values()) for v in resolved.values()
    )
    missing = [k for k, v in resolved.items() if all(vv is None or pd.isna(vv) for vv in v.values())]

    if verbose:
        local_resolved = len(ids) - len(to_query)
        api_resolved = resolved_count - local_resolved
        print(f"{format_log_prefix('result_only', indent=2)} {resolved_count}/{len(ids)} identifiers successfully converted "
            f"({local_resolved} local, {api_resolved} via UniProt).")
        if missing:
            print(f"{format_log_prefix('warn_only', indent=2)} {len(missing)} identifiers could not be resolved:")
            print("        " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))

    # --- Output
    result_df = pd.DataFrame({from_type: list(resolved.keys())})
    for t in to_type:
        result_df[t] = [resolved[i][t] for i in result_df[from_type]]

    if return_type == "dict":
        return resolved
    elif return_type == "df":
        return result_df
    elif return_type == "both":
        return resolved, result_df
    else:
        raise ValueError("Invalid return_type. Choose from {'dict', 'df', 'both'}.")
