*This tutorial is still under construction*

# Tutorial 4: Plotting

Generate publication-ready plots with scpviz — abundance panels, PCA/UMAP, clustermaps, raincloud, volcano plots, and more.

Most plotting functions accept a `matplotlib.axes.Axes` object as the first argument, allowing seamless integration into multi-panel figures.

---

## Abundance Plots

`plot_abundance()` draws violin or bar plots (with strip points) for selected proteins or peptides. It automatically chooses barplots when groups have few replicates:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
pdata.plot_abundance(
    ax,
    namelist=["ACTB", "GAPDH"],
    classes="condition",
    order=["control", "treated"]
    )
plt.show()
```

---

## Abundance Boxgrid Panels

`plot_abundance_boxgrid()` builds a one-row panel of compact box, bar, violin, or mean-line plots — one subplot per gene — with optional grouping on `.obs` columns.

### Basic panel

```python
fig, axes = pdata.plot_abundance_boxgrid(
    namelist=["GAPDH", "TUBB", "ACTB"],
    classes=["cellline", "condition"],
    plot_type="box",
    figsize=(2, 2.5),
)
plt.show()
```

### Significance brackets

Pass `sig_pairs` to run pairwise tests and draw significance bars (same group-spec format as `plot_volcano` / `de()`). Use `return_df=True` to also receive the abundance table and a `stats_df` of p-values.

**Per cell line** — compare sc vs kd within BE and within AS:

```python
fig, axes, df, stats = pdata.plot_abundance_boxgrid(
    namelist=["GAPDH", "TUBB", "ACTB"],
    classes=["cellline", "condition"],
    sig_pairs=[
        ({"cellline": "BE", "condition": "sc"}, {"cellline": "BE", "condition": "kd"}),
        ({"cellline": "AS", "condition": "sc"}, {"cellline": "AS", "condition": "kd"}),
    ],
    sig_kwargs={"fontsize": 8},
    return_df=True,
)
plt.show()
```

<figure markdown="span">
![Boxgrid significance — per cell line](../assets/plots/plot_abundance_boxgrid_significance.png)
</figure>

**Shared group across pairs** — the same group can appear in multiple comparisons (brackets stack vertically):

```python
fig, axes = pdata.plot_abundance_boxgrid(
    namelist=["GAPDH", "TUBB", "ACTB"],
    classes=["cellline", "condition"],
    sig_pairs=[
        ({"cellline": "BE", "condition": "sc"}, {"cellline": "BE", "condition": "kd"}),
        ({"cellline": "BE", "condition": "kd"}, {"cellline": "AS", "condition": "kd"}),
    ],
    sig_kwargs={"fontsize": 8},
)
plt.show()
```

<figure markdown="span">
![Boxgrid significance — shared group in multiple pairs](../assets/plots/plot_abundance_boxgrid_significance_multi.png)
</figure>

**Two groups only** — when a single `classes` column has exactly two levels, use `sig_pairs=True`:

```python
fig, axes = pdata.plot_abundance_boxgrid(
    namelist=["GAPDH"],
    classes="treatment",
    sig_pairs=True,
)
plt.show()
```

`sig_kwargs` defaults include `sig_test` (`"ttest"`, `"mannwhitneyu"`, or `"wilcoxon"`) and `sig_equal_var`; remaining keys are passed to `plot_significance`. Groups with no detectable abundance are labeled ND and skipped for testing. See the API reference for `plot_abundance_boxgrid` and `annotate_abundance_boxgrid_significance`.

---

## PCA and UMAP

```python
pdata.plot_pca(classes="celltype")
pdata.plot_umap(classes="condition")
```

---

## Clustermap

```python
pdata.plot_clustermap(namelist=["TP53", "VIM", "MAPT"], classes="condition")
```

Colors automatically follow sample classes, but you can customize palettes.
