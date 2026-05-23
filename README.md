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

### Deploy behind Cloudflare

Use the dedicated guide:

- [DEPLOY_CLOUDFLARE.md](DEPLOY_CLOUDFLARE.md)

### Standard Operating Protocol (SOP)

Use this checklist for a complete analysis run:

1. **Start app**
   - App is now on the render service: https://scpviz-webapp.onrender.com/
   - Note: The service may take a few minutes to load if left offline for too long.
2. **Import (Tab 1)**
   - Choose `Proteome Discoverer (PD)` or `DIA-NN`
   - Upload required file(s)
   - Click `Import dataset`
   - Confirm import log + summary table
3. **QC (Tab 2)**
   - Set grouping/metric as needed
   - Click `Refresh QC plots`
   - Optionally apply min-protein filter
4. **Preprocess (Tab 3)**
   - Choose normalize/impute settings
   - Click `Run preprocessing`
5. **Embeddings (Tab 4)**
   - Set class columns + abundance genes/proteins
   - Click `Compute embeddings + plots`
   - Confirm PCA/UMAP/abundance figures
6. **Differential expression (Tab 5)**
   - Configure group filters and method
   - Click `Run DE + volcano`
   - Confirm DE table + volcano plot
7. **Volcano styling and labels (Tab 5)**
   - Adjust colors, font, thresholds
   - Add labels from selection/click or by list/cutoff
   - Use exact/substring matching toggle for list labeling
   - Use label manager for update/delete/snap
8. **STRING enrichment (Tab 6)**
   - Refresh/select DE key
   - Run enrichment
   - Load STRING SVG/network
9. **Plot editor (Tab 7)**
   - Open from DE/enrichment or load directly in tab
   - Edit SVG, save, and optionally download edited SVG
10. **Export**
   - Download per-tab CSVs as needed
   - Use `Download all tables + plots (ZIP)` for full bundle

### Troubleshooting

- **DE fails with default JSON keys**
  - If your dataset uses `treatment` instead of `condition`, update group JSON accordingly (for example `{"cellline":"BE","treatment":"kd"}` vs `{"cellline":"BE","treatment":"sc"}`).
- **Volcano labels are not added**
  - Ensure DE has run and points are selected/clicked first.
  - If using list-based labels, verify exact vs substring toggle and list tokens.
- **Too many labels reduce readability**
  - Lower `Max labels`, use stricter p-value/log2FC cutoffs, and use snap/dodge tools.
- **STRING enrichment outputs missing**
  - Re-run enrichment, refresh keys, and ensure network access to STRING API.
- **Bundle ZIP missing files**
  - ZIP includes only artifacts generated in the current session; run the corresponding tab steps first.

## Contributing

If you'll like to contribute to `scpviz`, please see the [contributing guidelines](https://gnaprs.github.io/scpviz/dev/contributing/). We welcome contributions from the community to help improve, expand, and document the functionality of scpviz.

## License
`scpviz` was created by Marion Pang. It is licensed under the terms of the MIT license.
