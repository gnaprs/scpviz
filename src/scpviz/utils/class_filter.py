"""Sample filtering and UpSet helpers for scpviz."""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import warnings

import numpy as np
import anndata as ad
import pandas as pd

if TYPE_CHECKING:
    from scpviz.pAnnData.pAnnData import pAnnData


def format_class_filter(
    classes: str | list[str],
    class_value: str | list[str] | list[list[str]],
    exact_cases: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Convert legacy-style filter input into dictionary-style format.

    This function standardizes `(classes, class_value)` input into the dictionary
    format expected by `pAnnData.filter_sample_values()`. It supports both loose
    OR-style filtering and exact case matching across multiple metadata fields.

    Args:
        classes (str or list of str): Metadata field(s) to filter on.
            Example: `"treatment"` or `["cellline", "treatment"]`.
        class_value (str, list of str, or list of list): Values to filter by.
            - str: May be underscore-joined (e.g. `"kd_AS"`).
            - list of str: Multiple values, interpreted as OR (if `exact_cases=False`)
              or split into combinations (if `exact_cases=True`).
            - list of list: Each inner list defines a full set of values across classes.
        exact_cases (bool): If True, return a list of dictionaries representing
            exact combinations across fields. If False, return a dictionary with
            OR logic applied.

    Returns:
        formatted (dict or list of dict): Dictionary-style filter input compatible
        with `.filter_sample_values()`.

    Raises:
        ValueError: If input shapes are inconsistent with the number of classes,
            or if `class_value` entries are not valid strings/lists.

    Example:
        Single class with OR logic:
            ```python
            format_class_filter("treatment", ["kd", "sc"])
            ```
            ```
            {'treatment': ['kd', 'sc']}
            ```

        Multiple classes with loose matching:
            ```python
            format_class_filter(["cellline", "treatment"], ["AS", "kd"])
            ```
            ```
            {'cellline': 'AS', 'treatment': 'kd'}
            ```

        Multiple classes with exact cases (underscore-joined strings):
            ```python
            format_class_filter(
                ["cellline", "treatment"],
                ["AS_kd", "BE_sc"],
                exact_cases=True
            )
            ```
            ```
            [{'cellline': 'AS', 'treatment': 'kd'},
             {'cellline': 'BE', 'treatment': 'sc'}]
            ```

        Multiple classes with exact cases (list of lists):
            ```python 
            format_class_filter(
                ["cellline", "treatment"],
                [["AS", "kd"], ["BE", "sc"]],
                exact_cases=True
            )
            ```
            ```
            # [{'cellline': 'AS', 'treatment': 'kd'},
             {'cellline': 'BE', 'treatment': 'sc'}]
            ```

    !!! warning "Note"

        This function is primarily used internally by `utils.filter()` and
        `pAnnData.filter_sample_values()`. End users should generally call
        `.filter_sample_values()` directly on `pAnnData` objects instead of
        using this helper.
    """

    if isinstance(classes, str):
        # Simple case: one class
        if isinstance(class_value, list) and exact_cases:
            return [{classes: val} for val in class_value]
        else:
            return {classes: class_value}

    elif isinstance(classes, list):
        if exact_cases:
            if isinstance(class_value, str):
                class_value = [class_value]

            formatted = []
            for entry in class_value:
                if isinstance(entry, str):
                    values = entry.split('_')
                elif isinstance(entry, list):
                    values = entry
                else:
                    raise ValueError("Each class_value entry must be a string or a list.")

                if len(values) != len(classes):
                    raise ValueError("Each class_value entry must match the number of classes. Check that group/class labels did not contain unintentional underscores ('_').")
                formatted.append({cls: val for cls, val in zip(classes, values)})

            return formatted

        else:
            # loose match — OR within each class
            if isinstance(class_value, str):
                values = class_value.split('_')
            else:
                values = class_value
            if len(values) != len(classes):
                raise ValueError("class_value must align with the number of classes. Check that group/class labels did not contain unintentional underscores ('_').")
            return {cls: val for cls, val in zip(classes, values)}

    else:
        raise ValueError("Invalid input: `classes` should be a string or list of strings.")
    
def filter(
    pdata: pAnnData | ad.AnnData,
    class_type: str | list[str],
    values: dict[str, Any] | list[Any] | str,
    exact_cases: bool = False,
    debug: bool = False,
) -> pAnnData | ad.AnnData:
    """
    Legacy-style filtering of samples in pAnnData or AnnData objects.

    This function filters samples based on metadata values using the older
    `(class_type, values)` interface. For pAnnData objects, it automatically
    delegates to `.filter_sample_values()` after converting the input into the
    recommended dictionary-style format.

    !!! warning

        For pAnnData users, prefer `.filter_sample_values()` with dictionary-style
        input, as it is more flexible and consistent. The `filter()` utility is
        retained primarily for backward compatibility and direct AnnData usage.


    Args:
        pdata (pAnnData or AnnData): Input data object to filter.
        class_type (str or list of str): Metadata field(s) in `.obs` to filter on.
            Example: `"treatment"`, or `["cell_type", "treatment"]`.
        values (list, dict, or list of dict): Metadata values to match.
            - If `exact_cases=False`: Provide a dictionary or list-of-values per class.
            - If `exact_cases=True`: Provide a list of dictionaries specifying
              exact combinations across fields.
        exact_cases (bool): Whether to interpret `values` as exact combinations (AND logic).
            Defaults to False, which applies OR logic within each class type.
        debug (bool): If True, print the query string used for filtering.

    Returns:
        filtered (pAnnData or AnnData): A filtered object of the same type as `pdata`.


    Raises:
        ValueError: If input types are invalid, if fields are missing in `.obs`,
            or if `values` format does not match `exact_cases`.

    Example:
        Filter samples by a single metadata field:
            ```python
            samples = utils.filter(pdata, class_type="treatment", values="kd")
            ```

        Filter by multiple fields with OR logic: 
            ```python
            samples = utils.filter(
                    adata,
                    class_type=["cell_type", "treatment"],
                    values=[["wt", "kd"], ["control", "treatment"]]
                ) 
            # returns samples where cell_type is either 'wt' or 'kd' and treatment is either 'control' or 'treatment'
            ```

        Filter by exact case combinations:
            ```python 
            samples = utils.filter(
                    adata,
                    class_type=["cell_type", "treatment"],
                    values=[{"cell_type": "wt", "treatment": "control"},
                            {"cell_type": "kd", "treatment": "treatment"}],
                    exact_cases=True
                )
            # returns samples where cell_type is 'wt' and treatment is 'kd', or cell_type is 'control' and treatment is 'treatment'
            ```
    """
    
    if hasattr(pdata, "filter_sample_values"):
        warnings.warn(
            "You passed a pAnnData object to `filter()`. "
            "It is recommended to use `pdata.filter_sample_values()` directly.",
            UserWarning)
        
        print("UserWarning: It is recommended to use the class method `.filter_sample_values()` with dictionary-style input for cleaner and more consistent filtering.")

    formatted_values = format_class_filter(class_type, values, exact_cases)
    
    # pAnnData input
    if hasattr(pdata, "filter_sample_values"):
        return pdata.filter_sample_values(
            values=formatted_values,
            exact_cases=exact_cases,
            debug=debug,
            return_copy=True
        )

    # plain AnnData input
    elif isinstance(pdata, ad.AnnData):
        adata = pdata
        obs_keys = adata.obs.columns

        if exact_cases:
            if not isinstance(formatted_values, list) or not all(isinstance(v, dict) for v in formatted_values):
                raise ValueError("When exact_cases=True, `values` must be a list of dictionaries.")

            for case in formatted_values:
                if not case:
                    raise ValueError("Empty dictionary found in values.")
                for key in case:
                    if key not in obs_keys:
                        raise ValueError(f"Field '{key}' not found in adata.obs.")

            query = " | ".join([
                " & ".join([
                    f"(adata.obs['{k}'] == '{v}')" for k, v in case.items()
                ])
                for case in formatted_values
            ])

        else:
            if not isinstance(formatted_values, dict):
                raise ValueError("When exact_cases=False, `values` must be a dictionary.")

            for key in formatted_values:
                if key not in obs_keys:
                    raise ValueError(f"Field '{key}' not found in adata.obs.")

            query_parts = []
            for k, v in formatted_values.items():
                v_list = v if isinstance(v, list) else [v]
                part = " | ".join([f"(adata.obs['{k}'] == '{val}')" for val in v_list])
                query_parts.append(f"({part})")
            query = " & ".join(query_parts)

        if debug:
            print(f"Filter query: {query}")

        return adata[eval(query)]

    else:
        raise ValueError("Input must be a pAnnData or AnnData object.")
        
def resolve_class_filter(
    adata: pAnnData | ad.AnnData,
    classes: str | list[str],
    class_value: str | list[str],
    debug: bool = False,
    *,
    filter_func: Callable[..., pAnnData | ad.AnnData] | None = None,
) -> pAnnData | ad.AnnData:
    """
    Resolve `(classes, class_value)` inputs and apply filtering.

    This helper standardizes class/value pairs into dictionary-style filters
    and applies them to an AnnData or pAnnData object. It is primarily used
    internally by plotting and analysis functions.

    Args:
        adata (AnnData or pAnnData): Input data object to filter.
        classes (str or list of str): Metadata field(s) used for filtering.
        class_value (str or list of str): Values corresponding to `classes`.
        debug (bool): If True, print resolved class/value pairs.
        filter_func (callable, optional): Filtering function to apply.
            Defaults to :func:`filter`.

    Returns:
        filtered (AnnData or pAnnData): Subset of the input object, same type as `adata`.

    !!! warning
        This is an internal helper for use inside functions such as
        `plot_rankquant` and `plot_raincloud`. End users should call
        `pAnnData.filter_sample_values()` instead.

    Related Functions:
        - filter: Legacy utility for sample filtering.
        - format_class_filter: Standardizes filter inputs.
        - pAnnData.filter_sample_values: Recommended user-facing filter method.
    """

    if isinstance(classes, str):
        values = class_value
    else:
        values = class_value.split('_')

    if debug:
        print(f"Classes: {classes}, Values: {values}")

    if filter_func is None:
        filter_func = filter

    return filter_func(adata, classes, values, debug=debug)

def get_upset_contents(
    pdata: pAnnData,
    classes: str | list[str],
    on: str = "protein",
    upsetForm: bool = True,
    debug: bool = False,
) -> pd.DataFrame | dict[str, list[str]]:
    """
    Construct contents for an UpSet plot from a pAnnData object.

    This function extracts feature sets (proteins or peptides) present in
    specified sample classes and returns them either as a dictionary or
    in an `upsetplot`-compatible format.

    Args:
        pdata (pAnnData): The pAnnData object containing `.prot` and `.pep`.
        classes (str or list of str): Metadata column(s) in `.obs` to define sample groups.
            Example: `"cell_type"`, or `["cell_type", "treatment"]`.
        on (str): Data level to use. Options are `"protein"` (default) or `"peptide"`.
        upsetForm (bool): If True, return an `UpSet`-compatible DataFrame via
            `upsetplot.from_contents`. If False, return a raw dictionary.
        debug (bool): If True, print filtering steps and class resolution details.

    Returns:
        upset_data (pandas.DataFrame): Binary presence/absence DataFrame for use with
            `upsetplot.UpSet`, if `upsetForm=True`.
        upset_dict (dict): Mapping of class → list of present features,
            if `upsetForm=False`.

    Raises:
        ValueError: If `on` is not `"protein"` or `"peptide"`.

    Example:
        Get contents for an UpSet plot of sample classes:
            ```python
            upset_data = get_upset_contents(pdata, classes="treatment")
            from upsetplot import UpSet
            UpSet(upset_data, subset_size="count").plot()
            ```

        Retrieve raw dictionary of sets instead:
            ```python
            upset_dict = get_upset_contents(pdata, classes="treatment", upsetForm=False)
            ```

        Query proteins from a set and highlight them in a plot:
            ```python
            upset_data = scutils.get_upset_contents(pdata, classes="condition")
            prot_df = scutils.get_upset_query(
                upset_data, present=["treated"], absent=["control"], fetch_uniprot=False, pdata=pdata
            )
            scplt.plot_rankquant(ax, pdata, classes="condition", cmap=cmaps, color=colors)
            scplt.mark_rankquant(ax, pdata, mark_df=prot_df, class_values=["treated"], color="black")
            ```

    Related Functions:
        - plot_upset: Plot UpSet diagrams directly.
        - plot_venn: Plot Venn diagrams for up to 3 sets.
        - get_upset_query: Query an intersection and build a mark_df for highlighting.
    """
    import scpviz.utils as _u

    if on == 'protein':
        adata = pdata.prot
    elif on == 'peptide':
        adata = pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

    # Common error: if classes is a list with only one element, unpack it
    if isinstance(classes, list) and len(classes) == 1:
        classes = classes[0]

    classes_list = _u.get_classlist(adata, classes)
    upset_dict = {}

    for j, class_value in enumerate(classes_list):
        data_filtered = _u.resolve_class_filter(adata, classes, class_value, debug=True)

        # get proteins that are present in the filtered data (at least one value is not NaN, not 0)
        X = data_filtered.X.toarray()
        mask_present = (~np.isnan(X)) & (X != 0)
        prot_present = data_filtered.var_names[mask_present.sum(axis=0) > 0]
        upset_dict[class_value] = prot_present.tolist()

    if upsetForm:
        upset_data = _u.upsetplot.from_contents(upset_dict)
        return upset_data

    else:
        return upset_dict

_GENE_VAR_PRECEDENCE = (
    "gene_primary",
    "Gene Names",
    "Genes",
    "gene_names",
    "Gene",
)

def get_upset_query(
    upset_content: pd.DataFrame,
    present: list[str],
    absent: list[str],
    *,
    fetch_uniprot: bool,
    pdata: pAnnData | None = None,
    on: str = "protein",
) -> pd.DataFrame:
    """
    Query features from UpSet contents given inclusion and exclusion criteria.

    This function extracts the set of features (proteins or peptides) that are
    present in all specified groups and absent in others. You decide whether to
    query UniProt metadata for the resulting accessions via ``fetch_uniprot``
    (set ``False`` for large intersections to avoid slow API calls). The result
    is returned as a ``mark_df``-compatible DataFrame for use with
    ``mark_rankquant`` or ``mark_raincloud``.

    Args:
        upset_content (pandas.DataFrame): Output from `get_upset_contents` with
            ``upsetForm=True`` (the default).
        present (list of str): List of groups in which the features must be present.
        absent (list of str): List of groups in which the features must be absent.
        fetch_uniprot (bool): If True, query UniProt for full metadata via
            `get_uniprot_fields`. If False, build a lightweight DataFrame from
            ``pdata`` using accessions and any available gene names in ``.var``
            (recommended for large intersections, e.g. 1000+ proteins).
        pdata (pAnnData, optional): Required when ``fetch_uniprot=False`` so gene
            names can be read from ``.var``. Ignored when ``fetch_uniprot=True``.
        on (str): Data level for gene lookup when ``fetch_uniprot=False``.
            Options are ``"protein"`` (default) or ``"peptide"``.

    Returns:
        prot_query_df (pandas.DataFrame): Features matching the query. When
        ``fetch_uniprot=True``, includes UniProt metadata. When
        ``fetch_uniprot=False``, includes at least an ``accession`` column and,
        when available, ``gene_primary``.

    Example:
        Small intersection - Query proteins unique to one group and highlight them in a plot, also fetches UniProt metadata for gene labels:
            ```python
            upset_data = scutils.get_upset_contents(pdata, classes="condition")
            prot_df = scutils.get_upset_query(
                upset_data, present=["treated"], absent=["control"], fetch_uniprot=True
            )
            scplt.plot_rankquant(ax, pdata, classes="condition", cmap=cmaps, color=colors)
            scplt.mark_rankquant(ax, pdata, mark_df=prot_df, class_values=["treated"], color="black")
            ```

        Large intersection - skip UniProt and use local gene names:
            ```python
            upset_data = scutils.get_upset_contents(pdata, classes="condition")
            prot_df = scutils.get_upset_query(
                upset_data, present=["treated"], absent=["control"],
                fetch_uniprot=False, pdata=pdata,
            )
            ```

    Related Functions:
        - get_upset_contents: Generate presence/absence sets for UpSet analysis.
        - plot_upset: Plot UpSet diagrams from class-based sets.
        - plot_venn: Plot Venn diagrams for 2 to 3 sets.
    """
    import scpviz.utils as _u
    from scpviz.utils.formatting import format_log_prefix

    prot_query = (
        _u.upsetplot.query(upset_content, present=present, absent=absent)
        .data["id"]
        .tolist()
    )

    if not prot_query:
        return pd.DataFrame()

    if fetch_uniprot:
        return _u.get_uniprot_fields(prot_query, verbose=False)

    if pdata is None:
        raise ValueError(
            "pdata is required when fetch_uniprot=False so gene names can be "
            "read from .var. Pass pdata=pdata, or set fetch_uniprot=True to "
            "query UniProt instead."
        )

    adata = _u.get_adata(pdata, on)
    result = pd.DataFrame({"accession": prot_query})

    gene_var_col = next((c for c in _GENE_VAR_PRECEDENCE if c in adata.var.columns), None)
    if gene_var_col is None:
        warnings.warn(
            f"{format_log_prefix('warn')} No gene name column found in .{on}.var "
            f"(tried: {', '.join(_GENE_VAR_PRECEDENCE)}). Returning accessions only.",
            stacklevel=2,
        )
        return result

    genes = pd.Series(
        [
            adata.var.at[acc, gene_var_col] if acc in adata.var_names else pd.NA
            for acc in prot_query
        ],
        index=prot_query,
        dtype=object,
    )
    result["gene_primary"] = genes.values

    missing_mask = genes.isna() | (genes.astype(str).str.strip() == "")
    n_missing = int(missing_mask.sum())
    if n_missing:
        warnings.warn(
            f"{format_log_prefix('warn')} {n_missing}/{len(prot_query)} feature(s) "
            f"have missing gene names in .{on}.var['{gene_var_col}'].",
            stacklevel=2,
        )

    not_in_var = [acc for acc in prot_query if acc not in adata.var_names]
    if not_in_var:
        warnings.warn(
            f"{format_log_prefix('warn')} {len(not_in_var)}/{len(prot_query)} "
            f"feature(s) from the UpSet query are not in .{on}.var_names.",
            stacklevel=2,
        )

    return result
