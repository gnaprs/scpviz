# Installation

`scpviz` is distributed as a Python package and can be installed with `pip`.

```bash
pip install scpviz
```

This will install all required dependencies, including `scanpy`, `anndata`, `pandas`, and common plotting libraries.

## Optional extras

For single-cell proteomics workflows, install the `sc` extra to enable directLFQ normalization, Leiden clustering, Harmony batch correction, and imputation:

```bash
pip install scpviz[sc]
```

This adds: `directlfq`, `pimms-learn`, `harmonypy`, `leidenalg`, `igraph`, `scikit-misc`.

!!! note
    The `sc` extras are required for the [Single-cell tutorial](single_cell.md). The core installation is sufficient for bulk proteomics workflows described in the [Quickstart](quickstart.md).

## Development installation

To install the latest development version directly from GitHub:

```bash
git clone https://github.com/gnaprs/scpviz.git
cd scpviz
pip install -e .
```

This installs `scpviz` in **editable mode**, so any changes to the code will be reflected immediately.

If you would like to contribute to `scpviz`, please see the [Contributing Guide](../dev/contributing.md)

## Dependencies

- Python ≥ 3.10
- Core scientific stack: `numpy`, `scipy`, `pandas`  
- Data structures: `anndata`, `scanpy`  
- Plotting: `matplotlib`, `seaborn`, `upsetplot`  
- Network and enrichment: `requests` (for UniProt/STRING API access)  

**Optional (`sc` extra):** `directlfq`, `pimms-learn`, `harmonypy`, `leidenalg`, `igraph`, `scikit-misc`
