# single cell

Handling single cell data is slightly different from bulk data. More sparse, missing values, so need to handle normalization and imputation more carefully. Here, we go through some typical workflows for single cell proteomics (field is still being established!)

## preprocessing
### filter proteins
we suggest doing a 40% filter followed by minimum value imputation

### normalization
typically bulk proteomics is median normalization, but this only works if missing values are not too much and the distribution of protein abundance looks similar

there's also directlfq algorithm that we can use (cite paper), and [insert brief explanation here on how it works]. directlfq normalization can implemented by doing
```
pdata.normalize(method='directlfq')
```

!!! note
    this algorithm will create files in the workspace, and also might take awhile

### imputation

For imputation, we recommend the PIMMS algorith (cite paper)

## visualize data
typically done with umap

can also use tsne plot

## using scanpy
need to first use cleanup function to make data clean for scanpy (expects 0, not NaNs, will throw error otherwise)
```
pdata.clean_X()
```

scanpy expects AnnData objects, so we send in the .prot objects (after we have done filtering with eg pep)

```
pdata.filter_rs(unique_pep=2)
prot = pdata.prot
sc.tsne(prot)
```