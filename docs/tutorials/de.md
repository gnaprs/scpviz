*This tutorial is still under construction*
# Tutorial 5: Differential Expression (DE)

Run DE analysis at the protein or peptide level - including standard two-group tests and **donor-blocked** mixed-model / pseudobulk DE when biological replicates are shared across conditions.
---

## Protein-Level DE

```python
de_results = pdata.de(
    values=[
        {"condition": "treated"},
        {"condition": "control"},
    ],
    on="protein",
)
de_results.head()
```

`values=[group1, group2]` matches the volcano convention: **log2fc = group1 − group2**, and upregulated proteins (positive log2fc) annotate toward **group1** on the right of the plot.

---

## Fold Change Strategies

```python
# Mean-based fold change (default)
pdata.de(values=comparison_values, fold_change_mode="mean")

# Pairwise protein-level median
pdata.de(values=comparison_values, fold_change_mode="pairwise_median")

# Peptide-level pairwise median (via RS matrix)
pdata.de(values=comparison_values, fold_change_mode="pep_pairwise_median")
```

*Different strategies may be useful depending on noise and sample size.*

---

## Test Methods

```python
pdata.log_transform(base=2)  # limma needs a log-transformed layer

# Moderated t-test (limma-style empirical Bayes variance shrinkage)
de_results = pdata.de(values=comparison_values, method="limma", correct_fdr=True)
de_results.attrs["limma"]  # shared prior: s2_prior, df_prior, estimator
```

*Placeholder: limma shrinks each protein's variance toward a shared prior, which helps most with few replicates. Filter sparse proteins first (e.g. `filter_prot`), since the prior is fitted on whatever proteins are passed in. Other methods: `"ttest"` (default), `"mannwhitneyu"`, `"wilcoxon"`. Details to come.*

---

## Joint multi-group DE (`joint_de`)

`de()` compares two groups. When you have several conditions (e.g. vehicle plus three doses, or two detergents with and without drug), `joint_de()` fits one linear model per protein across **all** groups and tests any number of contrasts. Every contrast uses the residual variance pooled over all groups, so it has more degrees of freedom than separate two-group tests (assuming a similar within-group variance in every group).

```python
pdata.log_transform(base=2)

# Several doses against one vehicle (limma-moderated by default)
res = pdata.joint_de(
    groupby="condition", reference="vehicle", comparisons=["low", "mid", "high"],
)
res["contrasts"]     # {label: DataFrame} with the same columns as de() (+ se, n_<group>)
res["diagnostics"]   # per protein: group counts, df_residual, s2, s2_post, df_total

# Plain pooled-variance t-tests instead of limma moderation
pdata.joint_de(groupby="condition", reference="vehicle", comparisons=["low"], method="lm")
```

**Different controls per detergent.** Pass named `(test, ref)` pairs; with several columns the group label is the values joined by `_`:

```python
pdata.joint_de(
    groupby=["detergent", "treatment"],
    contrasts={
        "NP40 drug vs vehicle": ("NP40_drug", "NP40_vehicle"),
        "DDM drug vs vehicle": ("DDM_drug", "DDM_vehicle"),
    },
)
```

**Does the drug act differently in the two detergents?** Give a weight for each group. Groups with `+1` are added, groups with `-1` are subtracted, and the weights must sum to zero. This one computes "drug change in NP40" minus "drug change in DDM" in a single test:

```python
pdata.joint_de(
    groupby=["detergent", "treatment"],
    contrasts={"drug effect: NP40 minus DDM": {
        "NP40_drug": +1, "NP40_vehicle": -1, "DDM_drug": -1, "DDM_vehicle": +1,
    }},
)
```

`log2fc` is then (change in NP40) − (change in DDM): near 0 means the drug acts the same in both detergents. A protein that is significant in one detergent but not the other does *not* show the detergents differ; only this contrast tests that.

Plot one contrast with `plot_volcano(ax, de_data=res["contrasts"][label], correct_fdr=True)`.

*Notes: samples are treated as independent (use `mixed_de` for paired or repeated measurements); restrict the samples first if some groups are unrelated or much noisier, since they all share one variance estimate; filter sparse proteins upstream (`filter_prot`). `method="limma"` is the no-trend, non-robust moderation, checked against R limma (`lmFit(~0 + group)`, `contrasts.fit`, `eBayes`).*

---

## Donor-blocked mixed DE (`mixed_de`)

Standard `de()` treats every cell/sample as independent. In many proteomics designs (especially **single-cell** or multi-sample assays) that overstates evidence:

- Cells from the **same donor / animal / case / culture batch** share genetics, age, dissection, and prep technicalities.
- A strong donor baseline can look like a condition effect if one condition is over-represented in a donor (or vice versa).
- Ignoring that correlation can produce **false positives** that vanish when you compare *within* donors.

`pdata.mixed_de()` accounts for that **biological replicate structure** by blocking on a donor column (`donor_col`). Depending on settings it fits a cell-level mixed model (`expr ~ group + (1 | donor)`) or averages to **donor × group** pseudobulk and tests at the donor level.

!!! tip "When to use `mixed_de` vs `de`"
    Prefer **`mixed_de`** when the same donors contribute observations to both (or several) contrast levels e.g. region (Cortex vs SNpc) within animal, Agg+ vs Agg- within donor, or treatment within case.
    Prefer ordinary **`de()`** for simple unpaired cohort designs with no shared biological replicate IDs.

### Region comparison with donor blocking

`contrast=(test, ref)` uses the same order as `values=[group1, group2]` and DESeq2-style numerator/denominator: **log2fc = test − ref**.

```python
# Explicit contrast: log2fc = Cortex − SNpc
volcano_df = pdata.mixed_de(
    group_col="region",
    contrast=("Cortex", "SNpc"),  # (test, ref)
    donor_col="animal",
)
stats_key = "mixed: [{'region': 'Cortex'}] vs [{'region': 'SNpc'}] | donor=animal"

# Same comparison via values sugar (identical order)
volcano_df = pdata.mixed_de(
    values=[{"region": "Cortex"}, {"region": "SNpc"}],
    donor_col="animal",
)
```

Plot from the stored stats key (or the returned table):

```python
import matplotlib.pyplot as plt
from scpviz import plotting as scplt

fig, ax = plt.subplots(figsize=(4, 4))
scplt.plot_volcano(ax, pdata, stats_key=stats_key, correct_fdr=True)
plt.show()
```

### Subset to a cell type, covariates, and values sugar

```python
# Explicit contrast=(test, ref): log2fc = Agg+ − Agg−; subset stored as filter: [{...}]
df = pdata.mixed_de(
    group_col="condition",
    contrast=("Agg+", "Agg-"),
    donor_col="donor",
    fixed_covariates=["batch"],
    subset={"cell_type": "Astrocyte"},
)

# Equivalent values sugar: [group1, group2] → log2fc = group1 − group2
df = pdata.mixed_de(
    values=[{"condition": "Agg+"}, {"condition": "Agg-"}],
    donor_col="donor",
    subset={"cell_type": "Astrocyte"},
)
```

### Panel over cell types

```python
for ct in ["Astrocyte", "Microglia"]:
    pdata.mixed_de(
        group_col="condition",
        contrast=("Agg+", "Agg-"),
        donor_col="donor",
        subset={"cell_type": ct},
    )
```

### Random intercept + slope

```python
df = pdata.mixed_de(
    group_col="Cortex",
    contrast=("pSyn", "NeuN"),  # (test, ref)
    donor_col="Case",
    random_effects="intercept_slope",
)
```

### Interaction contrast (effect *at* a layer, without subsetting)

Use `contrast_at` when the formula has an interaction and you want the condition effect evaluated at a specific interacting level:

```python
df = pdata.mixed_de(
    formula="expr ~ condition * layer",
    contrast_term="condition",
    contrast=("disease", "control"),  # (test, ref)
    contrast_at={"layer": "L5"},
    donor_col="donor",
    subset={"cell_type": "Neuron"},
)
```

If the population of interest is already “L5 neurons only”, prefer `subset` instead:

```python
df = pdata.mixed_de(
    group_col="condition",
    contrast=("disease", "control"),
    donor_col="donor",
    subset={"cell_type": "Neuron", "layer": "L5"},
)
```

!!! note "Runtime"
    Cell-level mixed models (`observation_level='cells'`, `method='auto'` / `'mixedlm'`) fit **one model per protein** and can take **5+ minutes** on larger matrices. With few paired donors, try `observation_level='pseudobulk'` for a much faster donor-averaged test. The run header prints a warning when the slow path is selected.

!!! tip "Layers should be log2-scale"
    Typical workflow: normalize on a linear scale, then `pdata.log_transform(layer='X', base=2)` and pass `layer='X_log2'` (or rely on `set_X=True` so default `layer='X'` resolves via `.uns['current_X_layer']`).
