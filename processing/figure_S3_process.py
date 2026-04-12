"""
Figure S3 processing: FSC curves between LocScale maps at varying RMSD perturbation.

Inputs  (place in data/):
  data/emd_10692_additional_1.map
  data/pdb6y5a_additional_refined.pdb
  data/refined/locscale_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{N}_A.mrc
  data/blur200/locscale_additional_refined_perturbed_strict_masking_blurred200_no_overlap_rmsd_{N}_A.mrc
    N in [0, 1, 2, 5, 10, 15, 20]

Outputs (written to data/processed/):
  figure_S3_normal.csv   – FSC curves, refined maps
  figure_S3_blurred.csv  – FSC curves, blurred (B=200) maps

CSV schema: freq | rmsd_0A | rmsd_1A | rmsd_2A | rmsd_5A | rmsd_10A | rmsd_15A | rmsd_20A
"""

from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
from locscale.include.emmer.ndimage.profile_tools import frequency_array
from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RMSD_A     = [0, 1, 2, 5, 10, 15, 20]
EMMAP_PATH = DATA_DIR / "emd_10692_additional_1.map"
PDB_PATH   = DATA_DIR / "pdb6y5a_additional_refined.pdb"

PREFIX_NORMAL  = "locscale_additional_refined_perturbed_strict_masking_no_overlap_rmsd_{}_A.mrc"
PREFIX_BLURRED = "locscale_additional_refined_perturbed_strict_masking_blurred200_no_overlap_rmsd_{}_A.mrc"

print("Processing Figure S3 ...")

with mrcfile.open(str(EMMAP_PATH)) as mrc:
    apix = float(mrc.voxel_size.x)

mask_path = get_atomic_model_mask(str(EMMAP_PATH), str(PDB_PATH), dilation_radius=3, softening_parameter=5)
mask = mrcfile.open(mask_path).data

def load_map(subfolder, template, rmsd):
    return mrcfile.open(str(DATA_DIR / subfolder / template.format(rmsd))).data

normal_maps  = {r: load_map("refined",  PREFIX_NORMAL,  r) for r in RMSD_A}
blurred_maps = {r: load_map("blur200",  PREFIX_BLURRED, r) for r in RMSD_A}

freq       = None
fsc_normal  = {}
fsc_blurred = {}

for r in tqdm(RMSD_A, desc="Computing FSC"):
    fn = calculate_fsc_maps(normal_maps[0]  * mask, normal_maps[r]  * mask)
    fb = calculate_fsc_maps(blurred_maps[0] * mask, blurred_maps[r] * mask)
    if freq is None:
        freq = frequency_array(fn, apix=apix)
    fsc_normal[r]  = fn
    fsc_blurred[r] = fb

pd.DataFrame({
    "freq": freq,
    **{f"rmsd_{r}A": fsc_normal[r]  for r in RMSD_A},
}).to_csv(OUT_DIR / "figure_S3_normal.csv",  index=False)

pd.DataFrame({
    "freq": freq,
    **{f"rmsd_{r}A": fsc_blurred[r] for r in RMSD_A},
}).to_csv(OUT_DIR / "figure_S3_blurred.csv", index=False)

print("  -> data/processed/figure_S3_normal.csv")
print("  -> data/processed/figure_S3_blurred.csv")
print("Done.")
