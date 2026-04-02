"""
Figure 4b processing: Wilson B-factor distributions at multiple RMSD perturbation levels.

Inputs  (place in data/):
  data/model_maps/pdb6y5a_modelmap_no_overlap_rmsd_{N}.mrc  N in [0,100,200,500,1000,1500,2000]
  data/pdb6y5a_additional_refined.pdb

Outputs (written to data/processed/):
  figure_4b_bfactors.csv   – B-factor arrays (1000 samples) per RMSD level
  figure_4b_trend.csv      – Pearson r vs RMSD magnitude

figure_4b_bfactors.csv schema:
  bfactor_0pm | bfactor_100pm | bfactor_200pm | bfactor_500pm |
  bfactor_1000pm | bfactor_1500pm | bfactor_2000pm

figure_4b_trend.csv schema:
  rmsd_pm | rmsd_A | pearson_r
"""

from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from locscale.include.emmer.ndimage.map_tools import (
    get_bfactor_distribution_multiple,
    get_atomic_model_mask,
)

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RMSD_PM  = [0, 100, 200, 500, 1000, 1500, 2000]
RMSD_A   = [0,   1,   2,   5,   10,   15,   20]
PDB_PATH = DATA_DIR / "pdb6y5a_additional_refined.pdb"
FSC_RES  = 2.8

print("Processing Figure 4b (this may take several minutes) ...")

name_tpl   = "pdb6y5a_modelmap_no_overlap_rmsd_{}.mrc"
map_paths  = [str(DATA_DIR / "model_maps" / name_tpl.format(r)) for r in RMSD_PM]

mask_path  = get_atomic_model_mask(map_paths[0], str(PDB_PATH))
bfactors_d = get_bfactor_distribution_multiple(
    map_paths, mask_path, FSC_RES, num_centers=1000
)

bfactor_arrays = {}
for path, rmsd_pm in zip(map_paths, RMSD_PM):
    name = name_tpl.format(rmsd_pm)
    bfactor_arrays[rmsd_pm] = np.array([v[0] for v in bfactors_d[name].values()])

# ── CSV 1: B-factor sample arrays ───────────────────────────────────────────
df_bf = pd.DataFrame({f"bfactor_{r}pm": bfactor_arrays[r] for r in RMSD_PM})
df_bf.to_csv(OUT_DIR / "figure_4b_bfactors.csv", index=False)

# ── CSV 2: Pearson correlation trend ────────────────────────────────────────
corrs = [pearsonr(bfactor_arrays[0], bfactor_arrays[r])[0] for r in RMSD_PM]
df_trend = pd.DataFrame({
    "rmsd_pm":   RMSD_PM,
    "rmsd_A":    RMSD_A,
    "pearson_r": corrs,
})
df_trend.to_csv(OUT_DIR / "figure_4b_trend.csv", index=False)

print("  -> data/processed/figure_4b_bfactors.csv")
print("  -> data/processed/figure_4b_trend.csv")
print("Done.")
