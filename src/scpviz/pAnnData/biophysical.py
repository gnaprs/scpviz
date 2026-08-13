"""Biophysical peptide/protein sequence characterization for pAnnData."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from scpviz.utils.peptide_param import (
    DEFAULT_PEPTIDE_PROPERTIES,
    get_peptide_properties as _get_peptide_properties,
)


class BiophysicalMixin:
    """
    Biophysical characterization of peptide (and future protein) sequences.

    Functions:
        get_peptide_properties: Compute peptide biophysical properties and annotate ``.pep.var``.
    """

    def get_peptide_properties(
        self,
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
        Compute peptide biophysical properties for peptides in this object.

        Supports local metrics (e.g. ``length``) and Biopython ProtParam metrics.
        Scalar metrics (e.g. ``gravy``, ``molecular_weight``, ``isoelectric_point``)
        are written to ``.pep.var`` unless ``return_copy=True``. Dict- or list-valued
        metrics are stored in object columns (e.g. ``aa_counts``, ``flexibility``).

        Args:
            properties (list of str): Metric names to compute (local and/or Biopython; see
                Supported properties below). Default: ``gravy``, ``molecular_weight``,
                ``isoelectric_point``.
            accessions (list of str, optional): Protein accessions or gene names. When
                ``None``, all peptides are computed only if ``force=True``.
            sequence_from (str, optional): Column in ``.pep.var`` for sequence resolution,
                or ``None`` to auto-detect (``Stripped.Sequence``, then ``Annotated Sequence``).
            force (bool): Required to compute properties for every peptide in the object.
            return_copy (bool): If ``True``, return the DataFrame without modifying ``.pep.var``.
            monoisotopic (bool): Use monoisotopic mass for ``molecular_weight``.
            charge_at_pH (float, optional): pH for ``charge_at_pH`` when that property is requested.
            protein_scale (dict, optional): Amino-acid scale dict when ``protein_scale`` is requested.
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

            Peptides linked to a selected protein list (default: annotates ``.pep.var``):
                ```python
                protein_list = ["P34932", "Q9CZW5"]
                df = pdata.get_peptide_properties(accessions=protein_list)
                df.head()
                #   accession  peptide_id  peptide_sequence  gravy  molecular_weight  isoelectric_point
                ```

            Single accession or gene name:
                ```python
                df = pdata.get_peptide_properties(accessions=["HSPA4"])
                ```

            Mix local and Biopython metrics:
                ```python
                df = pdata.get_peptide_properties(
                    properties=["length", "gravy", "molecular_weight"],
                    accessions=protein_list,
                )
                ```

            Compute without modifying ``.pep.var``:
                ```python
                df = pdata.get_peptide_properties(
                    accessions=protein_list, return_copy=True
                )
                ```

            All peptides in the object (may be slow; requires ``force=True``):
                ```python
                df = pdata.get_peptide_properties(force=True)
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
            Sequence parsing auto-detects DIA-NN (``Stripped.Sequence``) and Proteome
            Discoverer (``Annotated Sequence``) formats.

        Related Functions:
            - scpviz.utils.strip_peptide_sequence: Clean peptide strings without ``pdata``.
            - scpviz.utils.compute_peptide_properties: Sequence-only computation.
            - scpviz.utils.get_peptide_properties: Same API as a standalone function.
            - scpviz.utils.get_peptides_for_accessions: Map accessions to peptide IDs.
        """
        kwargs: dict = {
            "accessions": accessions,
            "sequence_from": sequence_from,
            "force": force,
            "return_copy": return_copy,
            "monoisotopic": monoisotopic,
            "charge_at_pH": charge_at_pH,
        }
        if "protein_scale" in properties:
            kwargs["protein_scale"] = protein_scale
            kwargs["protein_scale_window"] = protein_scale_window
            kwargs["protein_scale_edge"] = protein_scale_edge

        return _get_peptide_properties(self, properties, **kwargs)
