# `pAnnData` Overview

The `pAnnData` object is the central data container in **scpviz**, extending the [AnnData](https://anndata.readthedocs.io/) structure for single-cell and bulk proteomics.  
It integrates matched **protein-level** and **peptide-level** matrices, along with metadata, summaries, and a protein–peptide relational structure (RS matrix).

This page introduces the core design of `pAnnData` and shows how to import data from supported formats.

!!! info "Key Features"
    - Matched `AnnData` objects for proteins (`.prot`) and peptides (`.pep`)
    - Support for multiple layers (raw, normalized, imputed, etc.)
    - Integrated metadata (`.metadata`) and summary tables (`.summary`)
    - Tracking of filtering, normalization, and analysis history
    - Compatible with all scpviz modules (plotting, enrichment, filtering, etc.)

<!-- ```mermaid
flowchart TB
    subgraph sources["Data sources"]
        PD["Proteome Discoverer\n(prot_file, pep_file)"]
        DIA["DIA-NN\n(report_file)"]
    end

    subgraph import["Import (scpviz.pAnnData.io)"]
        import_data["import_data(source_type, ...)"]
        parse["Parse matrices, obs, var,\npeptide→protein mapping"]
        build_rs["_build_rs_matrix()\n(protein × peptide)"]
        create["_create_pAnnData_from_parts()"]
    end

    subgraph container["pAnnData container"]
        pdata["pAnnData"]
        prot["prot: AnnData\n(samples × proteins)\n.X, .layers, .obs, .var, .obsm"]
        pep["pep: AnnData\n(samples × peptides)\n.X, .layers, .obs, .var, .obsm"]
        rs["rs: sparse matrix\n(proteins × peptides)"]
        summary["summary: TrackedDataFrame\n(merged sample-level metrics)"]
    end

    subgraph ops["Downstream operations (on .prot / .pep)"]
        filter["FilterMixin\nfilter_sample, filter_prot*, filter_rs"]
        edit["EditingMixin\nset_X, layers, export"]
        analysis["AnalysisMixin\nnormalize, impute, de, rank"]
        dimred["PCA, neighbor, UMAP, Leiden\n(via scanpy on .prot or .pep)"]
        enrich["EnrichmentMixin\nenrichment_*"]
        id_maps["IdentifierMixin\ngene↔accession, peptide↔protein"]
    end

    subgraph scanpy_integration["Scanpy integration"]
        direct["Use scanpy directly on\npdata.prot or pdata.pep"]
        sc_pp["sc.pp.neighbors\nsc.tl.pca\nsc.tl.umap\nsc.tl.leiden"]
        harmony["sc.external.pp.harmony_integrate"]
    end

    PD --> import_data
    DIA --> import_data
    import_data --> parse
    parse --> build_rs
    parse --> create
    build_rs --> create
    create --> pdata

    pdata --> prot
    pdata --> pep
    pdata --> rs
    prot --> summary
    pep --> summary
    pdata --> summary

    pdata --> filter
    pdata --> edit
    pdata --> analysis
    pdata --> dimred
    filter --> prot
    filter --> pep
    edit --> prot
    edit --> pep
    analysis --> prot
    analysis --> pep
    dimred --> prot
    dimred --> pep

    prot --> direct
    pep --> direct
    direct --> sc_pp
    direct --> harmony
    dimred -.->|"uses"| sc_pp
``` -->

```mermaid
flowchart LR
    subgraph pAnnData["`pAnnData`"]
        P["`.prot  
        (protein AnnData)`"]
        Q["`.pep  
        (peptide AnnData)`"]
        S["`.summary  
        (sample-level table)`"]
        M["`.metadata  
        (dict of metadata)`"]
        R["`RS matrix  
        (protein × peptide mapping)`"]
    end

    P -- "proteins ↔ peptides" --> R
    Q -- "peptides ↔ proteins" --> R
    P -. "linked by sample IDs" .-> S
    Q -. "linked by sample IDs" .-> S
    M --> P
    M --> Q
    M --> S
```

## Importing Data

Data can be imported into `pAnnData` directly from DIA-NN, Proteome Discoverer, or other supported formats:

```python
from scpviz import pAnnData as pAnnData
from scpviz import plotting as scplt
from scpviz import utils as scutils

# From DIA-NN report
pdata = pAnnData.import_data(source_type='diann', report_file ="report.tsv")

# From Proteome Discoverer output
pdata = pAnnData.import_data(source_type='pd', prot_file ="proteomediscoverer_prot.txt", pept_file ="proteomediscoverer_pep.txt")
```

See the [importing tutorial](../tutorials/importing.md) for more information.

Once imported, the `pAnnData` object serves as the entry point for downstream workflows:
filtering, normalization, imputation, visualization, and enrichment analysis.

## Workflow Pipeline

The `pAnnData` object enables a modular analysis pipeline for single-cell and bulk proteomics.  
Each step builds on the previous one, but you can skip or repeat steps depending on your dataset and analysis goals.

```mermaid
graph TB
    A["`Import data  
    (DIA-NN / PD)`"] --> B["`Parse metadata  
    (.obs from filenames)`"]
    B --> C["`Filter proteins/peptides  
    (≥2 unique peptides, sample queries)`"]
    C --> D["`Normalize  
    (global, reference feature, directLFQ)`"]
    D --> E["`Impute missing values  
    (KNN / group-wise)`"]
    E --> F["`Visualize data  
    (abundance, PCA/UMAP, clustermap, raincloud, volcano)`"]
    F --> G["`Differential Expression  
    (mean, pairwise, peptide-level)`"]
    G --> H["`Enrichment (STRING)  
    (GSEA, GO, PPI)`"]
    B --> I["`Export results`"]

    %% Optional side paths
    B -. "QC summaries" .-> F
    C -. "RS matrix checks" .-> F
    G -. "ranked/unranked lists" .-> H
    D .-> I
    F .-> I
    G .-> I
```

!!! tip "Tutorials"
    Each step of the pipeline is explained in detail in the [tutorials](../tutorials/index.md):

    - [Importing and Exporting Data](../tutorials/importing.md)
    - [Filtering ](../tutorials/filtering.md)  
    - [Imputation + Normalization](../tutorials/imputation.md)  
    - [Plotting](../tutorials/plotting.md)
    - [Differential Expression](../tutorials/de.md)  
    - [Enrichment + Networks](../tutorials/enrichment.md)  

    For a guided introduction, start with the [Quickstart](../tutorials/quickstart.md).  

---

## Modules  

In addition to the core `pAnnData` class, **scpviz** provides two standalone modules that support analysis and visualization:  

- **`scpviz.utils`**  
  A collection of helper functions for data processing, filtering, formatting, and interacting with external resources such as UniProt.  
  These functions are primarily used internally but can also be useful for advanced users who need finer control over their workflows.  

- **`scpviz.plotting`**  
  A set of high-level visualization utilities designed to work seamlessly with `pAnnData`.  
  Functions include abundance plots, rank plots, raincloud plots, clustermaps, UMAP/PCA projections, and UpSet diagrams.  
  Each plotting function is designed to accept a `matplotlib.axes.Axes` object for flexible integration into custom figure layouts.  

---

## Developer Utilities  

**scpviz** also includes a small set of developer-focused tools that help maintain consistency and internal state:  

- **`TrackedDataFrame`**  
  A subclass of `pandas.DataFrame` that marks its parent `pAnnData` object as “stale” whenever it is modified directly.  
  This ensures that summary tables and metadata stay consistent with the main data layers.  

- **Hidden functions**  
  Internal helpers that are not part of the standard API.  
  These are documented for completeness but should generally not be used directly in analysis workflows.  

!!! warning "Developer utilities"  
    These tools are included for package maintainers and power users. Most end-users will not need to interact with them directly.  
 
---
