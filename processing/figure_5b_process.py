"""
Figure 5b processing: PWLF breakpoint frequencies for helix vs sheet.

Input   (place in data/secondary_structure_analysis/):
  pwlf_breakpoint_data.pickle

Output  (written to data/processed/):
  figure_5b.csv

CSV schema (long format): structure_type | breakpoint_A
  (filtered to 3 Å < breakpoint < 6 Å)
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
SS_DIR   = ROOT_DIR / "data" / "secondary_structure_analysis"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_CUTOFF  = 6   # Angstroms  (exclude > 6 Å)
HIGH_CUTOFF = 3   # Angstroms  (exclude < 3 Å)

print("Processing Figure 5b ...")

with open(SS_DIR / "pwlf_breakpoint_data.pickle", "rb") as f:
    pwlf_data = pickle.load(f)


def get_breakpoints(key):
    bps = np.array(pwlf_data[key]["breakpoints"])[:, 1]  # second breakpoint
    mask = (bps < LOW_CUTOFF) & (bps > HIGH_CUTOFF)
    return bps[mask]


helix_bp = get_breakpoints("helix")
sheet_bp = get_breakpoints("sheet")

rows = (
    [{"structure_type": "helix", "breakpoint_A": v} for v in helix_bp]
    + [{"structure_type": "sheet", "breakpoint_A": v} for v in sheet_bp]
)
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "figure_5b.csv", index=False)

print(f"  helix N={len(helix_bp)}, sheet N={len(sheet_bp)}")
print("  -> data/processed/figure_5b.csv")
print("Done.")
