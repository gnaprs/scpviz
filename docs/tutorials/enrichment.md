*This tutorial is still under construction*
# Tutorial 6: Enrichment and Networks

Use STRING integration for enrichment and protein–protein interaction (PPI) analysis.

---

## Functional Enrichment (GSEA / GO)

```python
pdata.enrichment_functional(from_de=True, de_key="...")  # key from pdata.list_enrichments()
```

---

## PPI Enrichment

```python
pdata.enrichment_ppi(genes=["P55072", "NPLOC4", "UFD1", "STX5A", "NSFL1C"])
```

---

## Viewing Results

```python
pdata.list_enrichments()
```

🔗 *Each enrichment run will provide a STRING link for interactive exploration.*
