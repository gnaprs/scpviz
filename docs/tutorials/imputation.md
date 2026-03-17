*This tutorial is still under construction*

# Tutorial 3: Imputation and Normalization
## Imputation
Missing values are common in proteomics. scpviz provides several imputation methods.

!!! note
    Pre-processing functions like normalize() and impute() act directly on the pAnnData object rather than returning a copy.
    By default, the processed data are written to the active .X layer unless a new layer name is specified.

---

### KNN Imputation
here: include a figure showing what KNN imputation looks like

```python
pdata.impute(method="knn", n_neighbors=5)
```

---

### Group-wise Imputation
scpviz also supports group-wise imputation. we implement it like this (maybe show diagram?)
e.g. for median method, with groupby condition:
can just show the example we have in dev actually!

```python
pdata.impute(method="group_mean", groupby="condition")
```

---
#
## Checking Imputation Stats

```python
pdata.stats["imputation"]
```

📊 *This dictionary stores how many values were imputed per group or overall.*

## Normalization
Brief explanation of normalization - to try to remove the effects of small variation in sample prep/sample loading, or mass spec performance. The assumption is that all samples (if global) or all samples within the group (if use `groupby`) should have a similar distribution of abundances.

scpviz implements several options:
(put as table?)

method      how it works                            recommended use
mean        makes mean of all samples match
median      makes median of all samples match       typically use in bulk proteomics
sum         makes sum of all samples match          typically use in single cell rnaseq, but not as appropriate for scp?
directlfq   (look at paper)                         used for single cell proteomics (see paper link)

## Normalization

```python
# Normalize intensities by global median
pdata.normalize(method="median")

# Normalize to reference proteins
pdata.normalize(method="reference_feature", reference_columns=["ACTB", "GAPDH"])
```

💡 *Note: Normalization choices can affect downstream DE and enrichment.*
