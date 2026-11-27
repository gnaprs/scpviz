"""Build the small Spectronaut test fixtures from real Spectronaut wide ("Run Pivot") reports.

Run once:  python dev/build_spectronaut_fixture.py <Protein_Report.tsv> <Peptide_Report.tsv>
(the reports must contain at least 40 runs, and run 40 must be empty; keep them out of git).
Writes tests/test_spectronaut_prot.tsv and tests/test_spectronaut_pep.tsv (runs 1-7 plus the empty run 40,
~150 protein groups present in both files, all peptides of each kept group).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scpviz.pAnnData.io import _parse_spectronaut_columns

TESTS = Path(__file__).resolve().parent.parent / "tests"
KEEP_RUNS = {1, 2, 3, 4, 5, 6, 7, 40}
N_GROUPS = 150
MAX_PEPTIDES = 40  # keep the files small

def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def run_index(col):
    m = re.match(r"^\[(\d+)\] ", col)
    return int(m.group(1)) if m else None

if len(sys.argv) != 3:
    sys.exit(__doc__)
prot, pep = read(sys.argv[1]), read(sys.argv[2])

def keep_cols(df):
    run_cols = {c for c, _, _ in _parse_spectronaut_columns(df.columns)}
    return [c for c in df.columns if c not in run_cols or run_index(c) in KEEP_RUNS]

prot, pep = prot[keep_cols(prot)], pep[keep_cols(pep)]
pq = [c for c in prot.columns if c.endswith("PG.QValue (Run-Wise)")]
px = [c for c in prot.columns if c.endswith("PG.Quantity")]
ex = [c for c in pep.columns if "EG.TotalQuantity" in c]

# candidate groups: in both files, few peptides
n_pep = pep.groupby("PG.ProteinGroups").size()
cands = [g for g in prot["PG.ProteinGroups"] if g in n_pep.index and n_pep[g] <= MAX_PEPTIDES]
cands = list(dict.fromkeys(cands))
prot_c = prot.set_index("PG.ProteinGroups", drop=False)
n_members = pd.Series({g: len(g.split(";")) for g in cands})

# required groups (searched among candidates)
chosen = []
def pick(mask_groups, label):
    g = next((g for g in mask_groups if g not in chosen), None)
    assert g is not None, f"no candidate group for: {label}"
    chosen.append(g)

pick([g for g in cands if n_members[g] >= 3], ">=3 members")
pick([g for g in cands if n_members[g] == 2], "2 members")
pick([g for g in cands if prot_c.loc[g, "PG.Genes"] == ""], "blank PG.Genes")
pick([g for g in cands if any(prot_c.loc[g, px[i]] in ("", "NaN") and float(prot_c.loc[g, pq[i]] or 1) < 0.01
                              for i in range(len(px)))], "NaN quantity with q < 0.01")
cand_set = set(cands)
for mod in ("Carbamidomethyl (C)", "Oxidation (M)", "Acetyl (Protein N-term)"):
    has_mod = pep.loc[pep["EG.PrecursorId"].str.contains(mod, regex=False), "PG.ProteinGroups"]
    pick([g for g in has_mod.unique() if g in cand_set], mod)
has_filtered = pep.loc[(pep[ex] == "Filtered").any(axis=1), "PG.ProteinGroups"]
pick([g for g in has_filtered.unique() if g in cand_set], "Filtered cell")

rng = np.random.default_rng(0)
rest = [g for g in cands if g not in chosen]
chosen += list(rng.choice(rest, size=N_GROUPS - len(chosen), replace=False))

prot_out = prot[prot["PG.ProteinGroups"].isin(chosen)]
pep_out = pep[pep["PG.ProteinGroups"].isin(chosen)]

# assert the properties the tests rely on
assert prot_out["PG.ProteinGroups"].nunique() == pep_out["PG.ProteinGroups"].nunique() == N_GROUPS
assert (prot_out["PG.ProteinGroups"].str.count(";") >= 2).any() and (prot_out["PG.ProteinGroups"].str.count(";") == 1).any()
assert (prot_out["PG.Genes"] == "").any()
assert (pep_out[ex] == "Filtered").to_numpy().any()
for mod in ("Carbamidomethyl (C)", "Oxidation (M)", "Acetyl (Protein N-term)"):
    assert pep_out["EG.PrecursorId"].str.contains(mod, regex=False).any(), mod
empty_x = [c for c in px if run_index(c) == 40][0]
assert (prot_out[empty_x].isin(["", "NaN"])).all()
assert any((prot_out[px[i]].isin(["", "NaN"]) & (pd.to_numeric(prot_out[pq[i]], errors="coerce") < 0.01)).any() for i in range(len(px)))

assert {run_index(c) for c in prot_out.columns if run_index(c)} == KEEP_RUNS
prot_out.to_csv(TESTS / "test_spectronaut_prot.tsv", sep="\t", index=False, lineterminator="\n")
pep_out.to_csv(TESTS / "test_spectronaut_pep.tsv", sep="\t", index=False, lineterminator="\n")
for f in ("test_spectronaut_prot.tsv", "test_spectronaut_pep.tsv"):
    size = (TESTS / f).stat().st_size
    assert size < 1_000_000, f"{f} is {size // 1024} KB; reduce N_GROUPS or MAX_PEPTIDES"
    print(f, size // 1024, "KB")
print("groups:", len(prot_out), "peptides:", len(pep_out))
