# scpviz
<img src="https://raw.githubusercontent.com/gnaprs/scpviz/refs/heads/main/docs/assets/300ppi/logo_white_label@300x.png"
 align="right" width="200"/>
 [![DOI](https://zenodo.org/badge/762480088.svg)](https://doi.org/10.5281/zenodo.17362532)

**Build & Tests:**  
[![codecov](https://codecov.io/gh/gnaprs/scpviz/branch/main/graph/badge.svg)](https://codecov.io/gh/gnaprs/scpviz)

**CI Matrix Status:**
| Python | Ubuntu | macOS | Windows |
|--------|--------|--------|----------|
| **3.11** | ![Ubuntu-3.11](https://github.com/gnaprs/scpviz/actions/workflows/ubuntu-3.11.yml/badge.svg) | ![macOS-3.11](https://github.com/gnaprs/scpviz/actions/workflows/macos-3.11.yml/badge.svg) | ![Windows-3.11](https://github.com/gnaprs/scpviz/actions/workflows/windows-3.11.yml/badge.svg) |
| **3.8** | ![Ubuntu-3.8](https://github.com/gnaprs/scpviz/actions/workflows/ubuntu-3.8.yml/badge.svg) | Unsupported | ![Windows-3.8](https://github.com/gnaprs/scpviz/actions/workflows/windows-3.8.yml/badge.svg) |

*macOS + Python 3.8 is unsupported due to issues with `scikit-misc` installation.*

**Documentation:**  
[![Docs CI](https://github.com/gnaprs/scpviz/actions/workflows/docs.yml/badge.svg)](https://github.com/gnaprs/scpviz/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-v0.5.2a-brightgreen.svg)](https://gnaprs.github.io/scpviz)

## Overview
**scpviz** is a Python package for single-cell and spatial proteomics data analysis, built around a custom `pAnnData` object.  
It extends the [AnnData](https://anndata.readthedocs.io/) ecosystem with proteomics-specific functionality, enabling seamless integration of proteins, peptides, and relational data.

* **Documentation**: https://gnaprs.github.io/scpviz/
* **Python Package Index (PyPI)**: https://pypi.org/project/scpviz/

## Getting started
### Installation

`scpviz` requires Python 3.8 or later. It is distributed as a Python package and can be installed with `pip`.

    python3 -m pip install scpviz

This will install all required dependencies, including `scanpy`, `anndata`, `pandas`, and common plotting libraries.

For the most up-to-date version of scpviz, clone the repository and
install the package using pip:

    conda create -n scpviz python=3.8 numpy pandas pip
    conda activate scpviz
    pip install git+https://github.com/gnaprs/scpviz.git@development

### Quickstart

Check out the [quickstart](https://gnaprs.github.io/scpviz/tutorials/quickstart/) guide for a run through import, basic preprocessing and quick visualization

### In-depth Tutorials
For more in-depth guides on importing, filtering, plotting, and running enrichment, see the [tutorials](https://gnaprs.github.io/scpviz/tutorials/).

### API Reference

Full function documentation for the `pAnnData` class and utility modules can be found on our [documentation page](https://gnaprs.github.io/scpviz/reference/).

## Dash Web App (MVP)

This repository includes a local Dash app that wraps the main `scpviz` workflow (import, QC, preprocessing, embeddings, DE, and STRING enrichment).

### Run the app

```bash
python -m pip install -e .
python -m dash_app.app
```

Then open the local URL shown in the terminal (typically `http://127.0.0.1:8050`).

### Supported upload modes

- **Proteome Discoverer (PD)**: upload protein file (required) and peptide file (optional).
- **DIA-NN**: upload report file.

### Notes

- This MVP targets local, single-user usage with in-memory session state.
- Existing `scpviz` plotting functions are rendered in the app via image conversion where needed, and Plotly-native rendering is used for volcano visualization.

## Contributing

If you'll like to contribute to `scpviz`, please see the [contributing guidelines](https://gnaprs.github.io/scpviz/dev/contributing/). We welcome contributions from the community to help improve, expand, and document the functionality of scpviz.

## License
`scpviz` was created by Marion Pang. It is licensed under the terms of the MIT license.
