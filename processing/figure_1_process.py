"""
Figure 1 processing: radial profiles from uniform B-factor model maps.

Inputs  (place in data/):
  data/full_side_chain/uniform_bfactor_{B}_0p5.mrc      B in [0,50,100,150,200,250,300]
  data/backbone_only/backbone_uniform_bfactor_{B}_0p5.mrc

Outputs (written to data/processed/):
  figure_1a.csv  – radial profiles for full side-chain maps
  figure_1b.csv  – radial profiles for backbone-only maps

CSV schema: freq | bfactor_0 | bfactor_50 | ... | bfactor_300
"""

from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
import os 
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Root directory: {ROOT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUT_DIR}")
APIX          = 0.5
BFACTOR_RANGE = [0, 50, 100, 150, 200, 250, 300]

print("Processing Figure 1 (a) and (b) ...")

freq        = None
profiles_sc = {}   # side-chain
profiles_bb = {}   # backbone-only

for b in BFACTOR_RANGE:
    sc_path = DATA_DIR / "full_side_chain"  / f"uniform_bfactor_{b}_0p5.mrc"
    bb_path = DATA_DIR / "backbone_only"    / f"backbone_uniform_bfactor_{b}_0p5.mrc"

    with mrcfile.open(str(sc_path)) as mrc:
        rp_sc = compute_radial_profile(mrc.data.copy())

    with mrcfile.open(str(bb_path)) as mrc:
        rp_bb = compute_radial_profile(mrc.data.copy())

    if freq is None:
        freq = frequency_array(rp_sc, apix=APIX)

    profiles_sc[b] = rp_sc
    profiles_bb[b] = rp_bb

# ── Figure 1a: side-chain profiles ───────────────────────────────────────────
df_1a = pd.DataFrame({"freq": freq})
for b in BFACTOR_RANGE:
    df_1a[f"bfactor_{b}"] = profiles_sc[b]
df_1a.to_csv(OUT_DIR / "figure_1a.csv", index=False)

# ── Figure 1b: backbone-only profiles ────────────────────────────────────────
df_1b = pd.DataFrame({"freq": freq})
for b in BFACTOR_RANGE:
    df_1b[f"bfactor_{b}"] = profiles_bb[b]
df_1b.to_csv(OUT_DIR / "figure_1b.csv", index=False)

print("  -> data/processed/figure_1a.csv")
print("  -> data/processed/figure_1b.csv")
print("Done.")
