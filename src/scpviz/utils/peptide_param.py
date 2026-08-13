"""Peptide sequence parsing and Biopython ProtParam wrappers for scpviz."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from scpviz.utils.data import get_peptides_for_accessions
from scpviz.utils.formatting import format_log_prefix

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData

_PD_PREFIX = re.compile(r"^\[[A-Z]\]\.")
_PD_SUFFIX = re.compile(r"\.\[[A-Z]\]$")
_TRAILING_DIGITS = re.compile(r"\d+$")
_STANDARD_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

DEFAULT_PEPTIDE_PROPERTIES: tuple[str, ...] = (
    "gravy",
    "molecular_weight",
    "isoelectric_point",
)

SCALAR_PROPERTIES: frozenset[str] = frozenset(
    {
        "gravy",
        "molecular_weight",
        "aromaticity",
        "instability_index",
        "isoelectric_point",
    }
)

TUPLE_PROPERTIES: dict[str, tuple[str, ...]] = {
    "secondary_structure_fraction": ("ss_helix", "ss_turn", "ss_sheet"),
    "molar_extinction_coefficient": ("extinction_reduced", "extinction_oxidized"),
}

DICT_OR_LIST_PROPERTIES: dict[str, str] = {
    "count_amino_acids": "aa_counts",
    "get_amino_acids_percent": "aa_percent",
    "flexibility": "flexibility",
    "protein_scale": "protein_scale",
}

SUPPORTED_PROPERTIES: frozenset[str] = frozenset(
    SCALAR_PROPERTIES
    | TUPLE_PROPERTIES.keys()
    | DICT_OR_LIST_PROPERTIES.keys()
    | {"charge_at_pH", "length"}
)


def strip_peptide_sequence(peptide_id: str, *, sequence: str | None = None) -> str | None:
    """
    Extract a standard amino-acid string suitable for Biopython ``ProteinAnalysis``.

    Strips Proteome Discoverer notation (terminal ``[X].`` / ``.[X]``, ``MOD:`` suffix,
    trailing digits). When ``sequence`` is provided (e.g. DIA-NN ``Stripped.Sequence``),
    that value is cleaned instead of ``peptide_id``.

    Args:
        peptide_id (str): Raw peptide ID or annotated sequence string.
        sequence (str, optional): Pre-parsed sequence column value. When set, ``peptide_id``
            is ignored except for context in downstream tables.

    Returns:
        str or None: Uppercase sequence of standard IUPAC letters, or ``None`` if invalid.

    Example:
        Clean a Proteome Discoverer annotated string:
            ```python
            from scpviz import utils as scutils

            raw = "[R].MQHNLEQQIQAR.[N] MOD:1xOxidation [M1]"
            scutils.strip_peptide_sequence(raw)
            # 'MQHNLEQQIQAR'
            ```

        Clean a specified peptide list, then compute features (no ``pAnnData`` required):
            ```python
            from scpviz import utils as scutils

            raw_peptides = [
                "[K].KPVVDCVVSVPCFYTDAER.[R]",
                "[R].MQHNLEQQIQAR.[N] MOD:1xOxidation [M1]",
            ]
            cleaned = [scutils.strip_peptide_sequence(p) for p in raw_peptides]
            props = scutils.compute_peptide_properties(cleaned)
            props
            #    gravy  molecular_weight  isoelectric_point
            # 0  ...              ...                ...
            # 1 -1.392         1495.663              6.502
            ```

    Related Functions:
        - compute_peptide_properties: Compute ProtParam metrics from cleaned sequences.
        - resolve_peptide_sequence: Resolve sequences from ``pdata.pep`` metadata.
    """
    raw = str(sequence if sequence is not None else peptide_id)
    raw = raw.split(" MOD:")[0]
    raw = _PD_PREFIX.sub("", raw)
    raw = _PD_SUFFIX.sub("", raw)
    raw = _TRAILING_DIGITS.sub("", raw)
    raw = raw.strip().upper()
    if not raw or _STANDARD_AA.fullmatch(raw) is None:
        return None
    return raw


def resolve_peptide_sequence(
    pdata: pAnnData,
    peptide_id: str,
    *,
    sequence_from: str | None = None,
) -> str | None:
    """
    Resolve a clean amino-acid sequence for one peptide in ``pdata.pep``.

    Auto-detects ``Stripped.Sequence`` (DIA-NN), then ``Annotated Sequence`` (PD),
    otherwise parses ``peptide_id`` / ``.pep.var_names``.

    Args:
        pdata (pAnnData): Object with ``.pep``.
        peptide_id (str): Peptide ID (``.pep.var_names`` entry).
        sequence_from (str, optional): Column in ``.pep.var`` for sequence resolution,
            ``"index"`` to parse ``peptide_id`` only, or ``None`` to auto-detect.

    Returns:
        str or None: Clean amino-acid sequence, or ``None`` if parsing fails.

    Example:
        Resolve one peptide from a Proteome Discoverer import:
            ```python
            from scpviz import utils as scutils

            pep_id = pdata.pep.var_names[0]
            scutils.resolve_peptide_sequence(pdata, pep_id)
            # 'KPVVDCVVSVPCFYTDAER'
            ```

        DIA-NN: uses ``Stripped.Sequence`` automatically:
            ```python
            pep_id = pdata.pep.var_names[0]
            scutils.resolve_peptide_sequence(pdata, pep_id)
            ```

    Related Functions:
        - strip_peptide_sequence: Low-level string cleaning.
        - get_peptide_properties: Batch property computation for ``pdata``.
    """
    pep_var = pdata.pep.var
    peptide_id = str(peptide_id)

    if sequence_from is not None:
        if sequence_from == "index":
            return strip_peptide_sequence(peptide_id)
        if sequence_from not in pep_var.columns:
            available = ", ".join(map(str, pep_var.columns))
            raise ValueError(
                f"Column '{sequence_from}' not found in .pep.var. Available: {available}"
            )
        return strip_peptide_sequence(
            peptide_id, sequence=str(pep_var.loc[peptide_id, sequence_from])
        )

    if "Stripped.Sequence" in pep_var.columns:
        return strip_peptide_sequence(
            peptide_id, sequence=str(pep_var.loc[peptide_id, "Stripped.Sequence"])
        )
    if "Annotated Sequence" in pep_var.columns:
        return strip_peptide_sequence(
            peptide_id, sequence=str(pep_var.loc[peptide_id, "Annotated Sequence"])
        )
    return strip_peptide_sequence(peptide_id)


def _validate_properties(properties: Sequence[str]) -> list[str]:
    props = list(properties)
    if not props:
        raise ValueError("At least one property must be requested.")
    unknown = set(props) - SUPPORTED_PROPERTIES
    if unknown:
        raise ValueError(
            f"Unknown peptide properties: {sorted(unknown)}. "
            f"Supported: {sorted(SUPPORTED_PROPERTIES)}"
        )
    return props


def _property_columns(properties: Sequence[str], *, charge_at_pH: float | None) -> list[str]:
    cols: list[str] = []
    for prop in properties:
        if prop in SCALAR_PROPERTIES or prop == "length":
            cols.append(prop)
        elif prop in TUPLE_PROPERTIES:
            cols.extend(TUPLE_PROPERTIES[prop])
        elif prop in DICT_OR_LIST_PROPERTIES:
            cols.append(DICT_OR_LIST_PROPERTIES[prop])
        elif prop == "charge_at_pH":
            if charge_at_pH is None:
                raise ValueError(
                    "charge_at_pH must be set when 'charge_at_pH' is in properties."
                )
            cols.append(f"charge_at_pH_{charge_at_pH:g}")
    return cols


def _compute_one(
    sequence: str,
    properties: Sequence[str],
    *,
    monoisotopic: bool = False,
    charge_at_pH: float | None = None,
    protein_scale: dict[str, float] | None = None,
    protein_scale_window: int = 9,
    protein_scale_edge: float = 1.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Local metric (not a Biopython ProtParam method).
    if "length" in properties:
        out["length"] = len(sequence)

    biopython_props = [p for p in properties if p != "length"]
    if not biopython_props:
        return out

    try:
        analysis = ProteinAnalysis(sequence, monoisotopic=monoisotopic)
    except Exception:
        return out

    for prop in biopython_props:
        try:
            if prop in SCALAR_PROPERTIES:
                out[prop] = getattr(analysis, prop)()
            elif prop in TUPLE_PROPERTIES:
                values = getattr(analysis, prop)()
                for col, val in zip(TUPLE_PROPERTIES[prop], values):
                    out[col] = val
            elif prop in DICT_OR_LIST_PROPERTIES:
                col = DICT_OR_LIST_PROPERTIES[prop]
                if prop == "protein_scale":
                    if protein_scale is None:
                        raise ValueError(
                            "protein_scale must be set when 'protein_scale' is in properties."
                        )
                    out[col] = analysis.protein_scale(
                        protein_scale,
                        protein_scale_window,
                        edge=protein_scale_edge,
                    )
                else:
                    out[col] = getattr(analysis, prop)()
            elif prop == "charge_at_pH":
                out[f"charge_at_pH_{charge_at_pH:g}"] = analysis.charge_at_pH(charge_at_pH)
        except Exception:
            for col in _property_columns([prop], charge_at_pH=charge_at_pH):
                out.setdefault(col, pd.NA)
    return out


def compute_peptide_properties(
    sequences: str | Sequence[str] | pd.Series,
    properties: Sequence[str] = DEFAULT_PEPTIDE_PROPERTIES,
    *,
    monoisotopic: bool = False,
    charge_at_pH: float | None = None,
    protein_scale: dict[str, float] | None = None,
    protein_scale_window: int = 9,
    protein_scale_edge: float = 1.0,
) -> pd.DataFrame:
    """
    Compute peptide biophysical properties for one or more amino-acid sequences.

    Supports local metrics (e.g. ``length``) and Biopython ProtParam metrics.
    Does not require a ``pAnnData`` object. Raw Proteome Discoverer strings are
    cleaned automatically; DIA-NN ``Stripped.Sequence`` values can be passed directly.

    Args:
        sequences (str, list of str, or pandas.Series): Peptide sequence(s). Non-standard
            input yields ``pd.NA`` for that row.
        properties (list of str): Metric names to compute (local and/or Biopython; see
            Supported properties below). Default: ``gravy``, ``molecular_weight``,
            ``isoelectric_point``.
        monoisotopic (bool): Use monoisotopic mass for ``molecular_weight``.
        charge_at_pH (float, optional): Required when ``"charge_at_pH"`` is in ``properties``.
        protein_scale (dict, optional): Amino-acid scale dict; required when
            ``"protein_scale"`` is requested.
        protein_scale_window (int): Window size for ``protein_scale`` only.
        protein_scale_edge (float): Edge weight for ``protein_scale`` only.

    Returns:
        pd.DataFrame: One row per input sequence with columns for each requested metric.

    Example:
        Clean a specified peptide list, then compute features (no ``pAnnData`` required):
            ```python
            from scpviz import utils as scutils

            raw_peptides = [
                "[K].KPVVDCVVSVPCFYTDAER.[R]",
                "[R].MQHNLEQQIQAR.[N] MOD:1xOxidation [M1]",
            ]
            cleaned = [scutils.strip_peptide_sequence(p) for p in raw_peptides]
            props = scutils.compute_peptide_properties(cleaned)
            props
            #    gravy  molecular_weight  isoelectric_point
            # 0  ...              ...                ...
            # 1 -1.392         1495.663              6.502
            ```

        Single already-clean sequence (e.g. DIA-NN ``Stripped.Sequence``):
            ```python
            scutils.compute_peptide_properties("MQHNLEQQIQAR")
            #    gravy  molecular_weight  isoelectric_point
            # 0 -1.392         1495.663              6.502
            ```

        Mix local and Biopython metrics:
            ```python
            cleaned = scutils.strip_peptide_sequence("[R].MQHNLEQQIQAR.[N]")
            scutils.compute_peptide_properties(
                cleaned,
                properties=["length", "gravy", "aromaticity"],
            )
            ```

    !!! note "Supported ``properties``"
        Pass any mix of the names below in ``properties``.

        **Local (scpviz, not Biopython):**

        - ``length`` → ``length`` — amino-acid count of the cleaned sequence
          (``len(sequence)``)

        **Biopython** (``Bio.SeqUtils.ProtParam.ProteinAnalysis`` method names). See
        [Bio.SeqUtils.ProtParam](https://biopython.org/docs/latest/api/Bio.SeqUtils.ProtParam.html):

        Scalar (one column each):

        - ``gravy`` → ``gravy``
        - ``molecular_weight`` → ``molecular_weight``
        - ``aromaticity`` → ``aromaticity``
        - ``instability_index`` → ``instability_index``
        - ``isoelectric_point`` → ``isoelectric_point``
        - ``charge_at_pH`` → ``charge_at_pH_<pH>`` (requires ``charge_at_pH`` kwarg)

        Tuple-valued (split into multiple columns):

        - ``secondary_structure_fraction`` → ``ss_helix``, ``ss_turn``, ``ss_sheet``
        - ``molar_extinction_coefficient`` → ``extinction_reduced``, ``extinction_oxidized``

        Dict- or list-valued (object columns):

        - ``count_amino_acids`` → ``aa_counts`` (dict)
        - ``get_amino_acids_percent`` → ``aa_percent`` (dict)
        - ``flexibility`` → ``flexibility`` (list)
        - ``protein_scale`` → ``protein_scale`` (list; requires ``protein_scale``,
          ``protein_scale_window``, and ``protein_scale_edge`` kwargs)

        Default ``properties``: ``gravy``, ``molecular_weight``, ``isoelectric_point``.

    Note:
        Biopython accepts only standard amino-acid letters. Modifications are ignored
        after stripping PD/DIA-NN notation.

    Related Functions:
        - strip_peptide_sequence: Clean individual peptide strings.
        - get_peptide_properties: Compute and annotate peptides in a ``pAnnData`` object.
    """
    props = _validate_properties(properties)
    if "protein_scale" not in props:
        protein_scale = None

    if isinstance(sequences, str):
        seq_list = [sequences]
        index = [0]
    elif isinstance(sequences, pd.Series):
        seq_list = sequences.astype(str).tolist()
        index = sequences.index
    else:
        seq_list = [str(s) for s in sequences]
        index = range(len(seq_list))

    columns = _property_columns(props, charge_at_pH=charge_at_pH)
    rows: list[dict[str, Any]] = []
    for seq in seq_list:
        raw = str(seq).strip().upper()
        clean = raw if _STANDARD_AA.fullmatch(raw) else strip_peptide_sequence(raw)
        if clean is None:
            rows.append({col: pd.NA for col in columns})
            continue
        rows.append(
            _compute_one(
                clean,
                props,
                monoisotopic=monoisotopic,
                charge_at_pH=charge_at_pH,
                protein_scale=protein_scale,
                protein_scale_window=protein_scale_window,
                protein_scale_edge=protein_scale_edge,
            )
        )

    df = pd.DataFrame(rows, index=index)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def _build_peptide_rows(
    pdata: pAnnData,
    *,
    accessions: list[str] | None,
    sequence_from: str | None,
) -> pd.DataFrame:
    if accessions is not None:
        pep_df = get_peptides_for_accessions(
            pdata,
            accessions,
            sequence_from=sequence_from if sequence_from is not None else "index",
        )
        if pep_df.empty:
            return pep_df
        rows = []
        for _, row in pep_df.iterrows():
            peptide_id = str(row["peptide_id"])
            peptide_sequence = resolve_peptide_sequence(
                pdata, peptide_id, sequence_from=sequence_from
            )
            rows.append(
                {
                    "accession": row["accession"],
                    "peptide_id": peptide_id,
                    "peptide_sequence": peptide_sequence,
                }
            )
        return pd.DataFrame(rows)

    pep_names = pdata.pep.var_names.astype(str)
    sequences = [
        resolve_peptide_sequence(pdata, pid, sequence_from=sequence_from)
        for pid in pep_names
    ]
    return pd.DataFrame(
        {
            "peptide_id": pep_names,
            "peptide_sequence": sequences,
        }
    )


def get_peptide_properties(
    pdata: pAnnData,
    properties: Sequence[str] = DEFAULT_PEPTIDE_PROPERTIES,
    *,
    accessions: list[str] | None = None,
    sequence_from: str | None = None,
    force: bool = False,
    return_copy: bool = False,
    monoisotopic: bool = False,
    charge_at_pH: float | None = None,
    protein_scale: dict[str, float] | None = None,
    protein_scale_window: int = 9,
    protein_scale_edge: float = 1.0,
) -> pd.DataFrame:
    """
    Compute peptide biophysical properties for peptides in a ``pAnnData`` object.

    Supports local metrics (e.g. ``length``) and Biopython ProtParam metrics.
    By default annotates ``pdata.pep.var`` (``return_copy=False``) and returns a
    DataFrame. Scalar metrics are stored as columns; dict/list metrics (e.g.
    ``aa_counts``, ``flexibility``) are stored as object columns.

    Args:
        pdata (pAnnData): Object with ``.pep`` (and ``.rs`` when ``accessions`` is set).
        properties (list of str): Metric names to compute (local and/or Biopython; see
            Supported properties below). Default: ``gravy``, ``molecular_weight``,
            ``isoelectric_point``.
        accessions (list of str, optional): Protein accessions or gene names. When
            ``None``, all peptides are considered (requires ``force=True``).
        sequence_from (str, optional): ``.pep.var`` column for sequence resolution,
            or ``None`` to auto-detect.
        force (bool): Required when ``accessions=None`` to compute all peptides.
        return_copy (bool): If ``True``, return the DataFrame without modifying
            ``pdata.pep.var``.
        monoisotopic (bool): Use monoisotopic masses for ``molecular_weight``.
        charge_at_pH (float, optional): pH for ``charge_at_pH`` (required when requested).
        protein_scale (dict, optional): Scale dict for ``protein_scale`` (required when requested).
        protein_scale_window (int): Window size for ``protein_scale`` only.
        protein_scale_edge (float): Edge weight for ``protein_scale`` only.

    Returns:
        pd.DataFrame: Columns ``peptide_id``, ``peptide_sequence``, requested property
            columns, and ``accession`` when ``accessions`` is provided.

    Raises:
        ValueError: If ``.pep`` is missing, or if ``accessions=None`` without ``force=True``.

    Example:
        Peptide sequences only (no ``pAnnData``): clean, then compute:
            ```python
            from scpviz import utils as scutils

            raw_peptides = [
                "[K].KPVVDCVVSVPCFYTDAER.[R]",
                "[R].MQHNLEQQIQAR.[N] MOD:1xOxidation [M1]",
            ]
            cleaned = [scutils.strip_peptide_sequence(p) for p in raw_peptides]
            props = scutils.compute_peptide_properties(cleaned)
            ```

        Peptides for a selected protein accession list (also writes to ``.pep.var``):
            ```python
            from scpviz import utils as scutils

            protein_list = ["P34932", "Q9CZW5"]
            df = scutils.get_peptide_properties(pdata, accessions=protein_list)
            df.head()
            #   accession  peptide_id  peptide_sequence  gravy  molecular_weight  isoelectric_point
            # 0   P34932    ...         ...               ...    ...               ...
            ```

        One accession or gene name:
            ```python
            df = scutils.get_peptide_properties(pdata, accessions=["HSPA4"])
            ```

        Mix local and Biopython metrics:
            ```python
            df = scutils.get_peptide_properties(
                pdata,
                properties=["length", "gravy", "molecular_weight"],
                accessions=protein_list,
            )
            ```

        Return a DataFrame without annotating ``.pep.var``:
            ```python
            df = scutils.get_peptide_properties(
                pdata, accessions=protein_list, return_copy=True
            )
            ```

        All peptides in the object (slow on large datasets; requires ``force=True``):
            ```python
            df = scutils.get_peptide_properties(pdata, force=True)
            ```

        Equivalent via the ``pAnnData`` method:
            ```python
            df = pdata.get_peptide_properties(accessions=protein_list)
            ```

    !!! note "Supported ``properties``"
        Pass any mix of the names below in ``properties``.

        **Local (scpviz, not Biopython):**

        - ``length`` → ``length`` — amino-acid count of the cleaned sequence
          (``len(sequence)``)

        **Biopython** (``Bio.SeqUtils.ProtParam.ProteinAnalysis`` method names). See
        [Bio.SeqUtils.ProtParam](https://biopython.org/docs/latest/api/Bio.SeqUtils.ProtParam.html):

        Scalar (one ``.pep.var`` / DataFrame column each):

        - ``gravy`` → ``gravy``
        - ``molecular_weight`` → ``molecular_weight``
        - ``aromaticity`` → ``aromaticity``
        - ``instability_index`` → ``instability_index``
        - ``isoelectric_point`` → ``isoelectric_point``
        - ``charge_at_pH`` → ``charge_at_pH_<pH>`` (requires ``charge_at_pH`` kwarg)

        Tuple-valued (split into multiple columns):

        - ``secondary_structure_fraction`` → ``ss_helix``, ``ss_turn``, ``ss_sheet``
        - ``molar_extinction_coefficient`` → ``extinction_reduced``, ``extinction_oxidized``

        Dict- or list-valued (object columns):

        - ``count_amino_acids`` → ``aa_counts`` (dict)
        - ``get_amino_acids_percent`` → ``aa_percent`` (dict)
        - ``flexibility`` → ``flexibility`` (list)
        - ``protein_scale`` → ``protein_scale`` (list; requires ``protein_scale``,
          ``protein_scale_window``, and ``protein_scale_edge`` kwargs)

        Default ``properties``: ``gravy``, ``molecular_weight``, ``isoelectric_point``.

    Note:
        For peptide lists outside ``pAnnData``, use ``strip_peptide_sequence`` then
        ``compute_peptide_properties`` (see first example above). With
        ``accessions=None`` and ``force=False``, a warning is printed and computation
        is refused.

    Related Functions:
        - strip_peptide_sequence: Clean peptide strings without ``pdata``.
        - compute_peptide_properties: Sequence-only property computation.
        - get_peptides_for_accessions: Map accessions to observed peptide IDs.
        - pAnnData.get_peptide_properties: Same workflow as a method on ``pdata``.
    """
    if pdata.pep is None:
        raise ValueError("get_peptide_properties requires peptide data (.pep).")

    props = _validate_properties(properties)
    if "protein_scale" not in props:
        protein_scale = None

    n_peptides = pdata.pep.n_vars
    if accessions is None and not force:
        print(
            f"{format_log_prefix('warn')} get_peptide_properties: {n_peptides:,} peptides "
            "would be computed for the full dataset, which may take a long time. "
            "Pass accessions= [...] to restrict the calculation, or set force=True "
            "to compute properties for all peptides."
        )
        raise ValueError(
            f"Refusing to compute peptide properties for all {n_peptides:,} peptides "
            "without force=True."
        )

    base_rows = _build_peptide_rows(
        pdata, accessions=accessions, sequence_from=sequence_from
    )
    if base_rows.empty:
        return base_rows

    unique = (
        base_rows.drop_duplicates(subset=["peptide_id"])
        .set_index("peptide_id", drop=False)
        .sort_index(kind="mergesort")
    )
    prop_cols = _property_columns(props, charge_at_pH=charge_at_pH)

    computed = compute_peptide_properties(
        unique["peptide_sequence"].fillna("").tolist(),
        props,
        monoisotopic=monoisotopic,
        charge_at_pH=charge_at_pH,
        protein_scale=protein_scale,
        protein_scale_window=protein_scale_window,
        protein_scale_edge=protein_scale_edge,
    )
    computed.index = unique.index
    prop_by_peptide = computed

    out = base_rows.merge(
        prop_by_peptide,
        left_on="peptide_id",
        right_index=True,
        how="left",
    )
    col_order = ["peptide_id", "peptide_sequence"]
    if "accession" in out.columns:
        col_order = ["accession", *col_order]
    out = out[col_order + [c for c in prop_cols if c in out.columns]]

    if not return_copy:
        if pdata.pep.is_view:
            pdata.pep = pdata.pep.copy()
        pep_index = pdata.pep.var_names.astype(str)
        aligned = prop_by_peptide.reindex(pep_index)
        for col in aligned.columns:
            pdata.pep.var[col] = aligned[col].values

    return out.reset_index(drop=True)
