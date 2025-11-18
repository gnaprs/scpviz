*This tutorial is still under construction*

# Tutorial 1: Importing Data

This tutorial shows how to import DIA-NN or Proteome Discoverer (PD) outputs into a `pAnnData` object.

`scpviz` currently allows import from Proteome Discoverer (tested on Versions 2.5, 3.2) and DIA-NN (tested on versions 1.8.1, 2.0 and 2.1)

`pAnnData` objects are designed with the integration of both protein and peptide level data. 
* For DIA-NN inputs, the report contains all the information required. 
* For Proteome Discoverer files, peptide level data is optional, but recommended for enabling the peptide-level filtering and analysis functions.

## Encoding metadata

It is important to encode metadata about our samples to enable classification - e.g. knockdown vs a scrambled control, or treatment vs a control group.

(do one of those tab things here, to show how diff softwares encode the metadata)
*** DIA-NN Encoding
In DIA-NN, data is split by file name, so this  should encoded in raw file names. For example, a typical name could be:
`20251106_Caltech-Marion_Astral_25min_Aur25cm_KD-01.raw`
which when split by '_' consists of 
date - <20251106>
user - <Caltech-Marion>
mass spectrometer - <Astral>
gradient length - <25min>
column - <Aur25cm>
sample condition, replicate - <KD-01>
(The .raw is automatically dropped from metadata)

!!! note
    `scpviz` automatically detects and assisgns a delimiter based on the character that occurs most often. You can also specify your own delimiter during import, if this detection fails.

*** PD encoding
In PD, the sample type has to be set up in the study. Typically this involves creating a 'categorical variable' (e.g. `sample_condition`) in the study page, adding possible values to the varialbe (e.g. `control`, `kd`), and then assigning samples to the respective variables in the 'Samples' tab. 
TODO:show image here!!



## Loading DIA-NN Reports

For DIA-NN, the `report.parquet` file is all we need, since this file contains peptide level detail (per row) per file, and we can assemble the protein-peptide matrix as is. We can import it directly:

```python
from scpviz import pAnnData as pAnnData

# Load DIA-NN report
pdata = scv.pAnnData.from_file("example_diann_report.txt", source="diann")

pdata.describe()
```

---

## Loading Proteome Discoverer (PD) Reports

For Proteome Discoverer, we recommend making changes to the default layout to include the sample abundances (typically called "Abundances"), since default layout is scaled abundnance, not raw abundance. For ease of use, here is a download link to a pd layout file that you can just apply under - Layout -> open and select -> apply


Proteome Discoverer allows export of data tabs. We recommend exporting as text (a tab delimited file) - excel is supported, but will be much much slower as file opening is not optimized for excel format.
(image here for what to export)[docs/assets/pd_export_1.png]

Make sure to export AT MINIMUM the "protein" and "peptide sequence groups" tabs - select the check boxes as so:
(image for what to select)[docs/assets/pd_export_2.png]

Now we can import - we need to set the `prot_file` parameter at minimum. If `pep_file` is also given, then the protein-peptide matrix will be made, and this enables downstream peptide-based filtering or analysis on the protein level.

```python
from scpviz import pAnnData as pAnnData

prot_file_path = '../assets/pd32_Proteins.txt'
pep_file_path = '../assets/pd32_PeptideSequenceGroups.txt'

# Load PD report
pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file_path , pep_file=pep_file_path)
```
result is (need to format nicely in result markdown)
```text
🧭 [USER] Importing data of type [pd]
      Auto-detecting ',' as delimiter from first filename.
ℹ️ Filenames are uniform. Using `suggest_obs_columns()` to recommend obs_columns...

From filename: Sample, AS, RA, kd, d7
Suggested .obs columns:
  unknown??                 : Sample
  unknown??                 : AS
  unknown??                 : RA
  condition                 : kd
  unknown??                 : d7
Unrecognized token(s): ['Sample', 'AS', 'RA', 'd7']
Please manually label these.

ℹ️ Suggested obs:
obs_columns = ['<Sample?>', '<AS?>', '<RA?>', 'condition', '<d7?>']
     ⚠️ [WARN] Please review the suggested `obs_columns` above.
   → If acceptable, rerun `import_data(..., obs_columns=...)` with this list.
```

note that for PD, there is only global FDR data unlike for DIA-NN

### before version 3.2 i.e. PD 2.5
can also import, using the same format
---

## Metadata Parsing
— extract `.obs` columns from filenames or reports.  
Sample metadata (columns in `.obs`) can be inferred directly from filenames:

```python
pdata.obs.head()
```

- any updates to .summary will be automatically pushed to `.prot.obs` and `.pep.obs` (if available). User will be prompted when necessary to run `pdata.update_summary()`, typically if I (author of package) can't tell if its intentional/

*If filenames follow different formats, scpviz will suggest possible `.obs` columns or default to generic labels.*

In this case, I recommend making a parsing function - e.g. 
`parse_filenames` and reassigning to `.summary`. 

```py
def parse_filename_index(df):
    """
    Parses the index of a DataFrame assumed to be filenames into structured metadata columns.

    Expected filename format (delimited by "_"):
        [0] date
        [1] gradient
        [2] sample_id
        [3] size
        [4] confirmation
        [5] thickness
        [6] sample
        [7] organism
        [8] region
        [9] well_position

    Args:
        df (pd.DataFrame): DataFrame with index containing delimited filenames.

    Returns:
        pd.DataFrame: Original DataFrame with added metadata columns.
    """
    colnames = [
        'date',
        'gradient',
        'sample_id',
        'size',
        'confirmation',
        'thickness',
        'sample',
        'organism',
        'region',
        'well_position'
    ]

    parts = df.index.to_series().str.split('_', expand=True)
    if parts.shape[1] != len(colnames):
        raise ValueError(f"Expected {len(colnames)} parts after splitting index, got {parts.shape[1]}")

    df_parsed = df.copy()
    for i, col in enumerate(colnames):
        df_parsed[col] = parts.iloc[:, i]
    return df_parsed

```

---

## Export results
— save processed datasets, DE tables, or plots.

...
---
➡️ Next: [Filtering and Normalization](filtering.md)
