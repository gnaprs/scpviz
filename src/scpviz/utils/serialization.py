"""
HDF5 serialization for pAnnData objects (``.pdata`` files) and h5ad export helpers.

Values are written recursively into h5py groups. Each node carries a ``scpviz_type``
attribute so it can be rebuilt with its original Python type (dict / list / tuple / set,
DataFrame with MultiIndex columns and ``attrs``, Categorical, nullable arrays, ...).
Anything that cannot be stored natively is pickled **per value** and reported back to
the caller, so one unusual entry never prevents a save.

This module has no pAnnData dependency; ``IOMixin.save`` / ``load`` / ``export_h5ad``
are the user-facing wrappers.

Functions:
    write_value: Recursively write one Python value into an h5py group.
    read_value: Rebuild a value written by ``write_value``.
    write_pdata_file: Write named top-level parts plus root attributes to a ``.pdata`` file.
    read_pdata_file: Read all parts and root attributes back from a ``.pdata`` file.
    to_h5ad_safe: Convert a value into an h5ad-writable form, recording lossy conversions.
"""

from __future__ import annotations

import datetime
import pickle
from pathlib import Path, PurePath

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata.io import read_elem, write_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem, write_elem

from scpviz.utils.formatting import format_log_prefix

SCHEMA_VERSION = 1
FILE_FORMAT = "scpviz.pdata"

_TYPE = "scpviz_type"
_ADATA_SLOTS = ("X", "layers", "obs", "var", "obsm", "varm", "obsp", "varp", "uns")
_PY_SCALARS = {"bool": bool, "int": int, "float": float, "complex": complex}
_MASKED_ARRAYS = {
    "IntegerArray": pd.arrays.IntegerArray,
    "FloatingArray": pd.arrays.FloatingArray,
    "BooleanArray": pd.arrays.BooleanArray,
}
# Missing-value codes for object string arrays (0 = present).
_MISSING = {1: np.nan, 2: None, 3: pd.NA}


def _is_valid_h5_key(key) -> bool:
    return type(key) is str and key not in ("", ".") and "/" not in key


def _write_strings(group, key, strings, kw):
    """Store strings as fixed-width UTF-8 bytes: unlike HDF5 variable-length strings, these compress."""
    data = np.char.encode(np.asarray(strings, dtype=str), "utf-8")
    if data.dtype.itemsize == 0:
        data = data.astype("S1")
    return group.create_dataset(key, data=data, **kw)


def _read_strings(ds):
    return np.char.decode(ds[()], "utf-8").astype(object)


def _write_array(group, key, arr, compression, path=None, pickled=None):
    """Write a numpy / pandas extension array; raise TypeError if unsupported."""
    kw = {"compression": compression} if compression and arr.ndim > 0 and arr.size > 0 else {}

    if isinstance(arr, pd.Categorical):
        g = group.create_group(key)
        g.attrs[_TYPE] = "categorical"
        g.attrs["ordered"] = bool(arr.ordered)
        _write_array(g, "codes", np.asarray(arr.codes), compression)
        write_value(g, "categories", arr.categories, compression=compression)
        return

    if isinstance(arr, tuple(_MASKED_ARRAYS.values())):
        g = group.create_group(key)
        g.attrs[_TYPE] = "masked"
        g.attrs["cls"] = type(arr).__name__
        g.attrs["dtype"] = str(arr.dtype.numpy_dtype)
        mask = np.asarray(arr.isna())
        fill = False if arr.dtype.numpy_dtype.kind == "b" else 0
        _write_array(g, "values", arr.to_numpy(dtype=arr.dtype.numpy_dtype, na_value=fill), compression)
        _write_array(g, "mask", mask, compression)
        return

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"unsupported array type {type(arr).__name__}")

    kind = arr.dtype.kind
    if kind in "biufcS":
        ds = group.create_dataset(key, data=arr, **kw)
    elif kind == "U":
        ds = _write_strings(group, key, arr, kw)
    elif kind in "Mm":
        ds = group.create_dataset(key, data=arr.view(np.int64), **kw)
    elif kind == "O":
        # Strings with optional missing values (NaN / None / pd.NA), e.g. PD/DIA-NN .var columns.
        flat = arr.ravel()
        codes = np.zeros(flat.shape, dtype=np.int8)
        mixed = False
        for i, v in enumerate(flat):
            if type(v) is str:
                continue
            if v is None:
                codes[i] = 2
            elif v is pd.NA:
                codes[i] = 3
            elif type(v) is float and np.isnan(v):
                codes[i] = 1
            else:
                mixed = True
                break
        if mixed:
            # Mixed / nested cells (e.g. STRING 'inputGenes' lists): one child per element.
            if arr.ndim != 1:
                raise TypeError("multi-dimensional object array with non-string elements")
            g = group.create_group(key)
            g.attrs[_TYPE] = "objarray"
            g.attrs["length"] = len(arr)
            for i, x in enumerate(arr):
                write_value(g, str(i), x, path=f"{path}[{i}]", pickled=pickled, compression=compression)
            return
        g = group.create_group(key)
        g.attrs[_TYPE] = "strarray"
        g.attrs["shape"] = arr.shape
        _write_strings(g, "data", np.where(codes == 0, flat, ""), kw)
        if codes.any():
            g.create_dataset("missing", data=codes, **kw)
        return
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    ds.attrs[_TYPE] = "ndarray"
    ds.attrs["dtype"] = str(arr.dtype)


def _read_array_dataset(ds):
    dtype = ds.attrs["dtype"]
    kind = np.dtype(dtype).kind
    if kind == "U":
        return _read_strings(ds).astype(dtype)
    if kind in "Mm":
        return ds[()].view(dtype)
    return ds[()]


def _write_index(group, key, index, compression, path, pickled):
    g = group.create_group(key)
    if isinstance(index, pd.RangeIndex):
        g.attrs[_TYPE] = "rangeindex"
        g.attrs["start"], g.attrs["stop"], g.attrs["step"] = index.start, index.stop, index.step
        write_value(g, "name", index.name, path=f"{path}.name", pickled=pickled)
    elif isinstance(index, pd.MultiIndex):
        g.attrs[_TYPE] = "multiindex"
        write_value(g, "levels", list(index.levels), path=f"{path}.levels", pickled=pickled, compression=compression)
        write_value(g, "codes", [np.asarray(c) for c in index.codes], path=f"{path}.codes", pickled=pickled, compression=compression)
        write_value(g, "names", list(index.names), path=f"{path}.names", pickled=pickled)
    else:
        g.attrs[_TYPE] = "index"
        values = index.array if isinstance(index, pd.CategoricalIndex) else index.to_numpy()
        _write_array(g, "values", values, compression)
        write_value(g, "name", index.name, path=f"{path}.name", pickled=pickled)


def _write_attrs(g, value, compression, path, pickled):
    if value.attrs:
        write_value(g, "attrs", dict(value.attrs), path=f"{path}.attrs", pickled=pickled, compression=compression)


def _column_array(series):
    return series.array if pd.api.types.is_extension_array_dtype(series.dtype) else series.to_numpy()


def write_value(group, key, value, *, path=None, pickled=None, compression=None):
    """
    Recursively write one Python value into an h5py group.

    Containers and pandas objects are rebuilt with their original types by
    :func:`read_value`. Any value (or sub-value) that cannot be stored natively is
    pickled on its own and its path is appended to ``pickled``.

    Args:
        group (h5py.Group): Parent group.
        key (str): Child name to create in ``group``.
        value: Object to write.
        path (str, optional): Human-readable location used in ``pickled`` reports.
        pickled (list of str, optional): Collects paths of values stored as pickle.
        compression (str, optional): h5py compression for array datasets (e.g. ``"gzip"``).

    Returns:
        list of str: The ``pickled`` list (created if not given).
    """
    path = key if path is None else path
    pickled = [] if pickled is None else pickled
    try:
        _write_native(group, key, value, path, pickled, compression)
    except Exception:
        if key in group:
            del group[key]
        ds = group.create_dataset(key, data=np.void(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)))
        ds.attrs[_TYPE] = "pickle"
        pickled.append(path)
    return pickled


def _write_native(group, key, value, path, pickled, compression):
    t = type(value)
    if value is None:
        group.create_group(key).attrs[_TYPE] = "none"
    elif value is pd.NA:
        group.create_group(key).attrs[_TYPE] = "pdna"
    elif t is str:
        ds = group.create_dataset(key, data=value, dtype=h5py.string_dtype())
        ds.attrs[_TYPE] = "str"
    elif t in (bool, int, float, complex):
        ds = group.create_dataset(key, data=value)
        ds.attrs[_TYPE] = t.__name__
    elif isinstance(value, np.generic):
        ds = group.create_dataset(key, data=value)
        ds.attrs[_TYPE] = "npscalar"
    elif t is dict:
        if all(_is_valid_h5_key(k) for k in value):
            g = group.create_group(key, track_order=True)
            g.attrs[_TYPE] = "dict"
            for k, v in value.items():
                write_value(g, k, v, path=f"{path}[{k!r}]", pickled=pickled, compression=compression)
        else:
            g = group.create_group(key)
            g.attrs[_TYPE] = "dict_keyed"
            write_value(g, "keys", list(value.keys()), path=f"{path}.keys()", pickled=pickled)
            vals = g.create_group("values")
            for i, (k, v) in enumerate(value.items()):
                write_value(vals, str(i), v, path=f"{path}[{k!r}]", pickled=pickled, compression=compression)
    elif t in (list, tuple, set, frozenset):
        items = list(value)
        elem = None
        if items:
            first = type(items[0])
            if all(type(x) is first for x in items):
                if first is str:
                    elem = "str"
                elif first in (bool, int, float):
                    elem = first.__name__
                elif issubclass(first, np.generic) and first is not np.str_:
                    elem = "numpy"
        if elem is not None:
            kw = {"compression": compression} if compression else {}
            if elem == "str":
                ds = _write_strings(group, key, items, kw)
            else:
                ds = group.create_dataset(key, data=np.array(items), **kw)
            ds.attrs[_TYPE] = t.__name__
            ds.attrs["elem"] = elem
        else:
            g = group.create_group(key)
            g.attrs[_TYPE] = t.__name__
            g.attrs["length"] = len(items)
            for i, v in enumerate(items):
                write_value(g, str(i), v, path=f"{path}[{i}]", pickled=pickled, compression=compression)
    elif t is pd.DataFrame:
        g = group.create_group(key)
        g.attrs[_TYPE] = "dataframe"
        g.attrs["n_cols"] = value.shape[1]
        _write_index(g, "index", value.index, compression, f"{path}.index", pickled)
        _write_index(g, "columns", value.columns, compression, f"{path}.columns", pickled)
        data = g.create_group("data")
        for i in range(value.shape[1]):
            col = value.iloc[:, i]
            write_value(data, str(i), _column_array(col), path=f"{path}[{value.columns[i]!r}]",
                        pickled=pickled, compression=compression)
        _write_attrs(g, value, compression, path, pickled)
    elif t is pd.Series:
        g = group.create_group(key)
        g.attrs[_TYPE] = "series"
        write_value(g, "values", _column_array(value), path=f"{path}.values", pickled=pickled, compression=compression)
        _write_index(g, "index", value.index, compression, f"{path}.index", pickled)
        write_value(g, "name", value.name, path=f"{path}.name", pickled=pickled)
        _write_attrs(g, value, compression, path, pickled)
    elif isinstance(value, pd.Index):
        _write_index(group, key, value, compression, path, pickled)
    elif isinstance(value, (np.ndarray, pd.api.extensions.ExtensionArray)):
        _write_array(group, key, value, compression, path=path, pickled=pickled)
    elif sparse.issparse(value):
        write_elem(group, key, value, dataset_kwargs={"compression": compression} if compression else {})
        group[key].attrs[_TYPE] = "sparse"
        group[key].attrs["is_array"] = isinstance(value, sparse.sparray)
    elif t is pd.Timestamp or t is datetime.datetime or t is datetime.date:
        ds = group.create_dataset(key, data=value.isoformat(), dtype=h5py.string_dtype())
        ds.attrs[_TYPE] = t.__name__
    elif isinstance(value, PurePath):
        ds = group.create_dataset(key, data=str(value), dtype=h5py.string_dtype())
        ds.attrs[_TYPE] = "path"
    elif t is ad.AnnData:
        _write_adata(group, key, value, path, pickled, compression)
    else:
        raise TypeError(f"no native encoding for {t.__name__}")


def _write_adata(group, key, adata, path, pickled, compression):
    g = group.create_group(key)
    g.attrs[_TYPE] = "anndata"
    for slot in _ADATA_SLOTS:
        value = getattr(adata, slot)
        if slot in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            value = dict(value)
        if slot in ("obs", "var"):
            value = pd.DataFrame(value)
        write_value(g, slot, value, path=f"{path}.{slot}", pickled=pickled, compression=compression)
    if adata.raw is not None:
        write_value(g, "raw", adata.raw.to_adata(), path=f"{path}.raw", pickled=pickled, compression=compression)


def _read_index(g):
    tag = g.attrs[_TYPE]
    if tag == "rangeindex":
        return pd.RangeIndex(int(g.attrs["start"]), int(g.attrs["stop"]), int(g.attrs["step"]), name=read_value(g["name"]))
    if tag == "multiindex":
        return pd.MultiIndex(levels=read_value(g["levels"]), codes=read_value(g["codes"]), names=read_value(g["names"]))
    return pd.Index(read_value(g["values"]), name=read_value(g["name"]))


def _read_attrs(g, obj):
    if "attrs" in g:
        obj.attrs.update(read_value(g["attrs"]))
    return obj


def read_value(elem):
    """
    Rebuild a value written by :func:`write_value`.

    Args:
        elem (h5py.Group or h5py.Dataset): Node to read.

    Returns:
        The reconstructed Python object.
    """
    tag = elem.attrs.get(_TYPE)
    if tag is None:
        return read_elem(elem)
    if tag == "pickle":
        return pickle.loads(elem[()].tobytes())
    if tag == "none":
        return None
    if tag == "pdna":
        return pd.NA
    if tag in ("str", "path", "Timestamp", "datetime", "date"):
        s = elem.asstr()[()]
        return {"str": str, "path": Path, "Timestamp": pd.Timestamp,
                "datetime": datetime.datetime.fromisoformat, "date": datetime.date.fromisoformat}[tag](s)
    if tag in _PY_SCALARS:
        return _PY_SCALARS[tag](elem[()])
    if tag == "npscalar":
        return elem[()]
    if tag == "dict":
        return {k: read_value(elem[k]) for k in elem.keys()}
    if tag == "dict_keyed":
        keys = read_value(elem["keys"])
        return {k: read_value(elem["values"][str(i)]) for i, k in enumerate(keys)}
    if tag in ("list", "tuple", "set", "frozenset"):
        cls = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag]
        if isinstance(elem, h5py.Group):
            return cls(read_value(elem[str(i)]) for i in range(int(elem.attrs["length"])))
        e = elem.attrs["elem"]
        if e == "str":
            return cls(_read_strings(elem).tolist())
        if e == "numpy":
            return cls(elem[()])
        return cls(elem[()].tolist())
    if tag == "ndarray":
        return _read_array_dataset(elem)
    if tag == "objarray":
        arr = np.empty(int(elem.attrs["length"]), dtype=object)
        for i in range(len(arr)):
            arr[i] = read_value(elem[str(i)])
        return arr
    if tag == "strarray":
        arr = _read_strings(elem["data"])
        if "missing" in elem:
            for i in np.flatnonzero(elem["missing"][()]):
                arr[i] = _MISSING[int(elem["missing"][i])]
        return arr.reshape(tuple(elem.attrs["shape"]))
    if tag == "categorical":
        cats = read_value(elem["categories"])
        return pd.Categorical.from_codes(elem["codes"][()], categories=cats, ordered=bool(elem.attrs["ordered"]))
    if tag == "masked":
        values = read_value(elem["values"]).astype(elem.attrs["dtype"])
        return _MASKED_ARRAYS[elem.attrs["cls"]](values, read_value(elem["mask"]))
    if tag in ("rangeindex", "multiindex", "index"):
        return _read_index(elem)
    if tag == "dataframe":
        index = _read_index(elem["index"])
        columns = _read_index(elem["columns"])
        n = int(elem.attrs["n_cols"])
        # Build positionally, then attach labels (index/columns may contain duplicates).
        df = pd.DataFrame({i: read_value(elem["data"][str(i)]) for i in range(n)}, index=pd.RangeIndex(len(index)))
        df.index = index
        df.columns = columns
        return _read_attrs(elem, df)
    if tag == "series":
        s = pd.Series(read_value(elem["values"]), index=_read_index(elem["index"]), name=read_value(elem["name"]))
        return _read_attrs(elem, s)
    if tag == "sparse":
        m = read_elem(elem)
        if bool(elem.attrs["is_array"]) and not isinstance(m, sparse.sparray):
            m = sparse.csr_array(m) if m.format == "csr" else sparse.csc_array(m)
        elif not bool(elem.attrs["is_array"]) and isinstance(m, sparse.sparray):
            m = sparse.csr_matrix(m) if m.format == "csr" else sparse.csc_matrix(m)
        return m
    if tag == "anndata":
        parts = {slot: read_value(elem[slot]) for slot in _ADATA_SLOTS}
        adata = ad.AnnData(**parts)
        if "raw" in elem:
            adata.raw = read_value(elem["raw"])
        return adata
    raise ValueError(f"unknown {_TYPE} {tag!r} at {elem.name}")


def write_pdata_file(filepath, parts, attrs, compression=None):
    """
    Write named top-level parts and root attributes to a ``.pdata`` (HDF5) file.

    Args:
        filepath (str or Path): Destination file (overwritten if it exists).
        parts (dict): Mapping of part name (e.g. ``"prot"``, ``"stats"``) to value.
        attrs (dict): Root attributes (scalars / strings), e.g. schema version.
        compression (str, optional): h5py compression for array datasets.

    Returns:
        list of str: Paths of values that were stored as pickle.
    """
    pickled = []
    with h5py.File(filepath, "w") as f:
        f.attrs.update(attrs)
        for name, value in parts.items():
            write_value(f, name, value, path=name, pickled=pickled, compression=compression)
    return pickled


def read_pdata_file(filepath):
    """
    Read all parts and root attributes from a ``.pdata`` file.

    Args:
        filepath (str or Path): File written by :func:`write_pdata_file`.

    Returns:
        tuple(dict, dict): ``(parts, attrs)``.

    Raises:
        ValueError: If the file is not a ``.pdata`` file, or was written with a newer schema.
    """
    with h5py.File(filepath, "r") as f:
        attrs = dict(f.attrs)
        if attrs.get("format") != FILE_FORMAT:
            raise ValueError(f"{format_log_prefix('error')} {filepath} is not a scpviz .pdata file.")
        if int(attrs.get("schema_version", 0)) > SCHEMA_VERSION:
            raise ValueError(
                f"{format_log_prefix('error')} {filepath} uses .pdata schema v{attrs['schema_version']}, "
                f"newer than this scpviz supports (v{SCHEMA_VERSION}). Upgrade scpviz to load it."
            )
        parts = {name: read_value(f[name]) for name in f.keys()}
    return parts, attrs


def to_h5ad_safe(value, path, notes):
    """
    Convert a value into a form ``AnnData.write_h5ad`` accepts, recording lossy changes.

    Tuples/sets become lists, non-string or ``/``-containing keys become strings,
    MultiIndex columns are joined with ``"|"``, ``DataFrame.attrs`` are dropped,
    mixed object columns become strings, and unsupported objects become their ``repr``.

    Args:
        value: Object to convert.
        path (str): Location of ``value``, used in ``notes``.
        notes (list of str): Collects one message per lossy conversion.

    Returns:
        The converted value.
    """
    t = type(value)
    if value is None or t in (str, bool, int, float) or isinstance(value, (np.generic, np.ndarray)) or sparse.issparse(value):
        if isinstance(value, np.ndarray) and value.dtype.kind == "O":
            if not all(type(x) is str for x in value.ravel()):
                notes.append(f"{path}: object array → str")
                return value.astype(str)
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            new_k = str(k).replace("/", "|")
            if new_k != k:
                notes.append(f"{path}: key {k!r} → {new_k!r}")
            out[new_k] = to_h5ad_safe(v, f"{path}[{new_k!r}]", notes)
        return out
    if t in (list, tuple, set, frozenset):
        if t is not list:
            notes.append(f"{path}: {t.__name__} → list")
        items = [to_h5ad_safe(v, f"{path}[{i}]", notes) for i, v in enumerate(value)]
        if any(isinstance(x, (dict, list, pd.DataFrame)) for x in items):
            # anndata can only write lists of scalars (e.g. metadata['identifier_map_history'] is a list of dicts)
            notes.append(f"{path}: list of containers → dict keyed by position")
            return {str(i): x for i, x in enumerate(items)}
        if items and len({type(x) for x in items}) > 1 and all(np.isscalar(x) or x is None for x in items):
            notes.append(f"{path}: mixed-type list → str")
            items = [str(x) for x in items]
        return items
    if t is pd.Series:
        notes.append(f"{path}: Series → DataFrame")
        value = value.to_frame()
        t = pd.DataFrame
    if isinstance(value, pd.DataFrame):
        if value.attrs:
            notes.append(f"{path}: DataFrame.attrs dropped")
        df = pd.DataFrame(value).copy()
        df.attrs = {}
        if isinstance(df.columns, pd.MultiIndex):
            notes.append(f"{path}: MultiIndex columns joined with '|'")
            df.columns = ["|".join(map(str, c)) for c in df.columns]
        elif not all(type(c) is str for c in df.columns):
            df.columns = [str(c) for c in df.columns]
        if isinstance(df.index, pd.MultiIndex):
            notes.append(f"{path}: MultiIndex index joined with '|'")
            df.index = ["|".join(map(str, i)) for i in df.index]
        for c in df.columns:
            col = df[c]
            if col.dtype == object and not col.map(lambda x: type(x) is str).all():
                if col.dropna().map(lambda x: type(x) is str).all():
                    # Same conversion anndata applies to obs/var on write; values are preserved, so no note.
                    df[c] = col.astype("category")
                else:
                    notes.append(f"{path}[{c!r}]: mixed object column → str")
                    df[c] = col.astype(str)
            elif pd.api.types.is_datetime64_any_dtype(col.dtype):
                notes.append(f"{path}[{c!r}]: datetime column → str")
                df[c] = col.astype(str)
        return df
    if t in (pd.Timestamp, datetime.datetime, datetime.date):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    notes.append(f"{path}: {t.__name__} stored as repr")
    return repr(value)
