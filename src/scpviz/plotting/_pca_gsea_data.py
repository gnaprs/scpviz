"""PCA–GSEA: table construction, pathway selection, and shared helpers."""
import re
import numpy as np
import pandas as pd
from scpviz import utils

def _ensure_pca_gsea_payload(
    pdata,
    on="protein",
    key_added="pca_gsea",
    requested_pcs=None,
    force=False,
    gsea_kwargs=None,
):
    """
    Ensure ``adata.uns[key_added]`` holds PCA-GSEA results, running ``pca_gsea`` if missing or forced.

    Returns:
        tuple: ``(adata, payload)`` after validating ``adata.uns[key_added]``.
    """

    def _get_pca_gsea_payload(pdata, on="protein", key_added="pca_gsea"):
        """Return adata + validated pca_gsea payload."""
        adata = utils.get_adata(pdata, on)
        if key_added not in adata.uns:
            raise KeyError(
                f"{utils.format_log_prefix('error',2)} PCA GSEA key '{key_added}' not found in .uns."
            )
        payload = adata.uns[key_added]
        if "results" not in payload:
            raise KeyError(
                f"{utils.format_log_prefix('error',2)} Missing 'results' in .uns['{key_added}']."
            )
        return adata, payload

    adata = utils.get_adata(pdata, on)
    needs_run = force or key_added not in adata.uns
    if needs_run:
        if requested_pcs is None:
            raise ValueError("`requested_pcs` must be provided when auto-running pca_gsea.")
        if force:
            print(f"{utils.format_log_prefix('warn')} force=True: re-running pca_gsea on requested PCs {requested_pcs}.")
        else:
            print(f"{utils.format_log_prefix('info')} pca_gsea results not found; running pca_gsea on requested PCs {requested_pcs}.")
        kwargs = dict(gsea_kwargs or {})
        pdata.pca_gsea(on=on, pcs=[int(pc) for pc in requested_pcs], key_added=key_added, **kwargs)
    return _get_pca_gsea_payload(pdata, on, key_added)

def _build_pca_gsea_tables(payload, pcs=None):
    """
    Build long-format and pivot tables from a PCA-GSEA ``results`` dict.

    Returns:
        tuple: ``(long_df, matrix_df, fdr_df, missing_pc_keys)``. Pathway identifiers align with
        ``pathway_raw`` / ``pathway`` / ``library`` columns in ``long_df``.
    """
    rows = []
    results_dict = payload.get("results", {})
    available_pc_keys = sorted(results_dict.keys(), key=lambda x: int(str(x).replace("PC", "")))
    requested_pc_keys = available_pc_keys.copy()
    if pcs is not None:
        requested_pc_keys = [f"PC{int(pc)}" for pc in pcs]
    available_selected = [pc for pc in available_pc_keys if pc in requested_pc_keys]
    missing_pc_keys = [pc for pc in requested_pc_keys if pc not in available_pc_keys]
    if len(available_selected) == 0:
        raise ValueError("No matching PCs were found in pca_gsea results.")

    for pc_key in available_selected:
        df = results_dict[pc_key].copy()
        if "Term" in df.columns:
            pathway_raw = df["Term"].astype(str)
        elif df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            pathway_raw = pd.Index(df.index).astype(str)
        else:
            raise KeyError("Could not resolve pathway names (expected 'Term' column or pathway index).")

        nes_col = "NES" if "NES" in df.columns else ("nes" if "nes" in df.columns else None)
        if nes_col is None:
            raise KeyError(f"Missing NES column in pca_gsea result for {pc_key}.")
        fdr_col = "FDR q-val" if "FDR q-val" in df.columns else ("FDR" if "FDR" in df.columns else None)

        if "pathway" in df.columns and "library" in df.columns:
            pathway_short = df["pathway"].astype(str)
            pathway_lib = df["library"].astype(str)
        else:
            pathway_short = pathway_raw.map(lambda x: str(x).split("__", 1)[1] if "__" in str(x) else str(x))
            pathway_lib = pathway_raw.map(lambda x: str(x).split("__", 1)[0] if "__" in str(x) else "")
        tmp = pd.DataFrame({
            "pathway_raw": pathway_raw.values,
            "pathway": pathway_short.values,
            "library": pathway_lib.values,
            "pc": pc_key,
            "NES": pd.to_numeric(df[nes_col], errors="coerce").values,
        })
        if fdr_col is not None:
            tmp["FDR q-val"] = pd.to_numeric(df[fdr_col], errors="coerce").values
        else:
            tmp["FDR q-val"] = np.nan
        rows.append(tmp)

    long_df = pd.concat(rows, axis=0, ignore_index=True).dropna(subset=["pathway_raw", "NES"])
    matrix_df = long_df.pivot_table(index="pathway_raw", columns="pc", values="NES", aggfunc="first")
    fdr_df = long_df.pivot_table(index="pathway_raw", columns="pc", values="FDR q-val", aggfunc="first")
    matrix_df = matrix_df.reindex(columns=requested_pc_keys)
    fdr_df = fdr_df.reindex(index=matrix_df.index, columns=requested_pc_keys)
    return long_df, matrix_df, fdr_df, missing_pc_keys

def _format_pathway_label(label):
    """Convert enrichment labels like `DNA_REPLICATION` to `Dna Replication`."""
    raw = str(label).strip()
    text = raw.split("__", 1)[1] if "__" in raw else raw
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.title()

def _resolve_pca_gsea_namelist_pathways(matrix_df, long_df, namelist):
    """
    Map ``namelist`` strings to ``pathway_raw`` indices using **pathway raw (Term)** or **short pathway**
    only (not library). Case-insensitive. Preserves ``matrix_df`` index order among matches.

    Raises:
        ValueError: If any namelist entry matches nothing, or no rows remain in ``matrix_df``.
    """
    meta = long_df[["pathway_raw", "pathway"]].drop_duplicates("pathway_raw")
    dedup = []
    seen_q = set()
    for x in namelist:
        s = str(x)
        if s not in seen_q:
            seen_q.add(s)
            dedup.append(s)
    pr_lower = meta["pathway_raw"].astype(str).str.lower()
    pw_lower = meta["pathway"].astype(str).str.lower()
    missing = []
    matched_set = set()
    for query in dedup:
        ql = query.lower()
        hits = meta.loc[(pr_lower == ql) | (pw_lower == ql), "pathway_raw"].astype(str).unique().tolist()
        if not hits:
            missing.append(query)
        else:
            matched_set.update(hits)
    if missing:
        raise ValueError(
            "No pathways matched these `namelist` entries (matches `Term` / pathway_raw or short pathway "
            f"name only, not library): {missing!r}. Check exact strings, e.g. "
            "`utils.get_adata(pdata, on).uns[key_added]['results']['PC1']` (use your `on`, `key_added`, and PC key)."
        )
    ordered = [r for r in matrix_df.index if str(r) in matched_set]
    if not ordered:
        raise ValueError(
            "No pathways from `namelist` remain after `exclude_pathways`. "
            "Inspect `adata.uns[key_added]['results'][...]` for valid `Term` values."
        )
    return ordered

def _apply_pathway_name_filters(long_df, matrix_df, fdr_df, include_pathways=None, exclude_pathways=None):
    """Filter pathway rows by user include/exclude names (raw, short, or library)."""
    if include_pathways is None and exclude_pathways is None:
        return long_df, matrix_df, fdr_df

    meta = long_df[["pathway_raw", "pathway", "library"]].drop_duplicates("pathway_raw")
    meta = meta.set_index("pathway_raw")

    def _to_set(x):
        if x is None:
            return None
        if isinstance(x, str):
            return {x}
        return {str(v) for v in x}

    include_set = _to_set(include_pathways)
    exclude_set = _to_set(exclude_pathways)

    selected = pd.Series(True, index=meta.index)
    if include_set:
        selected &= (
            meta.index.to_series().isin(include_set)
            | meta["pathway"].astype(str).isin(include_set)
            | meta["library"].astype(str).isin(include_set)
        )
    if exclude_set:
        selected &= ~(
            meta.index.to_series().isin(exclude_set)
            | meta["pathway"].astype(str).isin(exclude_set)
            | meta["library"].astype(str).isin(exclude_set)
        )

    keep = meta.index[selected].tolist()
    long_df = long_df[long_df["pathway_raw"].isin(keep)].copy()
    matrix_df = matrix_df.loc[matrix_df.index.intersection(keep)]
    fdr_df = fdr_df.reindex(index=matrix_df.index, columns=matrix_df.columns)
    return long_df, matrix_df, fdr_df

def _compute_pc_score_df(matrix_df, fdr_df, fdr_cutoff=0.1):
    """
    Per-PC ranking score: ``|NES| * -log10(FDR)``, optionally gated by ``fdr_cutoff``.

    If ``fdr_cutoff`` is not ``None``, scores are zeroed on PCs where FDR exceeds the cutoff. ``None`` uses
    all FDR values with no gate. This function does not drop rows; callers may filter pathways by FDR before
    scoring.
    """
    fdr_safe = fdr_df.clip(lower=1e-300)
    score_df = matrix_df.abs().mul(-np.log10(fdr_safe))
    if fdr_cutoff is not None:
        score_df = score_df.where(fdr_df <= float(fdr_cutoff), 0.0)
    return score_df.fillna(0.0)

def _validate_plot_top_n(top_n, *, what="items"):
    """
    Require a positive integer ``top_n`` for PCA-GSEA and protein vector plots (no ``None``).
    """
    if top_n is None:
        raise ValueError(
            f"`top_n` must be a positive integer; got None. Pass e.g. 12 or 50 to cap {what}, "
            "or use `namelist` / `include_pathways` (bubble/heatmap) to restrict to a small set."
        )
    try:
        n = int(top_n)
    except (TypeError, ValueError) as err:
        raise ValueError(f"`top_n` must be a positive integer; got {top_n!r}.") from err
    if n < 1:
        raise ValueError(f"`top_n` must be at least 1; got {n}.")
    return n

def _select_top_pathways(score_df, top_n, top_n_mode="balanced"):
    """
    Select top pathways from a PC-score table.
    - balanced: split top_n approximately equally per PC (with dedupe).
    - max_score: global ranking by max score across PCs.

    ``top_n`` must be an integer >= 1 (validated at the public plot API).
    """
    n = int(top_n)
    if n < 1:
        raise ValueError("`top_n` must be at least 1.")

    if top_n_mode == "max_score":
        return score_df.max(axis=1).sort_values(ascending=False).head(n).index.tolist()

    if top_n_mode != "balanced":
        raise ValueError("`top_n_mode` must be either 'balanced' or 'max_score'.")

    pcs = list(score_df.columns)
    k = max(len(pcs), 1)
    base = n // k
    rem = n % k
    quotas = {pc: base + (1 if i < rem else 0) for i, pc in enumerate(pcs)}

    selected = []
    count_by_pc = {pc: 0 for pc in pcs}
    per_pc_rank = {pc: score_df[pc].sort_values(ascending=False).index.tolist() for pc in pcs}
    for pc in pcs:
        for pathway in per_pc_rank[pc]:
            if pathway in selected:
                continue
            selected.append(pathway)
            count_by_pc[pc] += 1
            if count_by_pc[pc] >= quotas[pc]:
                break

    if len(selected) < n:
        global_rank = score_df.max(axis=1).sort_values(ascending=False).index.tolist()
        for pathway in global_rank:
            if pathway in selected:
                continue
            selected.append(pathway)
            if len(selected) >= n:
                break

    return selected[:n]

# Use as default for ``n_vectors`` so "namelist only" does not also apply top-N unless the user passes ``n_vectors``.
N_VECTORS_UNSET = object()

def _resolve_protein_namelist_genes(matrix_df, namelist):
    """
    Map ``namelist`` to matrix row indices in list order (deduped). Match is exact on ``str(index)``.

    Returns:
        tuple: (ordered gene indices, set of those indices for excluding from top-N remainder).

    Raises:
        ValueError: If any namelist entry matches no row.
    """
    dedup = []
    seen_q = set()
    for x in namelist:
        t = str(x)
        if t not in seen_q:
            seen_q.add(t)
            dedup.append(t)
    missing = []
    ordered = []
    resolver_set = set()
    for name in dedup:
        hits = [i for i in matrix_df.index if str(i) == name]
        if not hits:
            missing.append(name)
        else:
            g = hits[0]
            if g not in resolver_set:
                ordered.append(g)
                resolver_set.add(g)
    if missing:
        raise ValueError(
            f"No proteins matched these `namelist` entries: {missing!r}. "
            "Check gene labels in the PCA loading table after `exclude_genes`."
        )
    return ordered, resolver_set

def _validate_plot_n_vectors(n_vectors, *, what="proteins"):
    """
    Validate ``n_vectors`` for protein PCA vectors: a positive int or a length-2 sequence of positive ints.

    Returns:
        tuple: ``("single", n)`` or ``("split", (nx, ny))``.
    """
    if isinstance(n_vectors, (list, tuple)):
        if len(n_vectors) != 2:
            raise ValueError(
                "`n_vectors` must be a positive int or a length-2 sequence of positive ints; "
                f"got a sequence of length {len(n_vectors)}."
            )
        out = []
        for i, v in enumerate(n_vectors):
            try:
                nv = int(v)
            except (TypeError, ValueError) as err:
                raise ValueError(f"`n_vectors[{i}]` must be a positive integer; got {v!r}.") from err
            if nv < 1:
                raise ValueError(f"`n_vectors[{i}]` must be at least 1; got {nv}.")
            out.append(nv)
        return ("split", tuple(out))
    if n_vectors is None:
        raise ValueError(
            f"`n_vectors` must be a positive int or a length-2 sequence; got None. "
            f"Pass e.g. 20 to cap {what}, use `namelist`, or pass both."
        )
    try:
        n = int(n_vectors)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"`n_vectors` must be a positive int or a length-2 sequence; got {n_vectors!r}."
        ) from err
    if n < 1:
        raise ValueError(f"`n_vectors` must be at least 1; got {n}.")
    return ("single", n)

def _select_pca_protein_vectors_split(score_df, pcx, pcy, nx, ny):
    """
    Top ``nx`` genes by score on ``pcx``, top ``ny`` on ``pcy``, union with X order first.
    """
    rank_x = score_df[pcx].sort_values(ascending=False).index.tolist()
    rank_y = score_df[pcy].sort_values(ascending=False).index.tolist()
    top_x = rank_x[:nx]
    selected = list(top_x)
    seen = set(top_x)
    for g in rank_y[:ny]:
        if g not in seen:
            selected.append(g)
            seen.add(g)
    return selected

def _vector_color_from_cmap(cmap, raw_label, formatted_label):
    """Resolve arrow/text color: exact raw, exact formatted, then case-insensitive key match."""
    if not cmap:
        return "black"
    if raw_label in cmap:
        return cmap[raw_label]
    if formatted_label in cmap:
        return cmap[formatted_label]
    gl = str(raw_label).lower()
    ll = str(formatted_label).lower()
    ci = {str(k).lower(): v for k, v in cmap.items()}
    if gl in ci:
        return ci[gl]
    if ll in ci:
        return ci[ll]
    return "black"
