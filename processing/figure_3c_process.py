"""
Figure 3c processing: radial profiles of perturbed model maps.

Inputs  (place in data/model_maps/):
  pdb6y5a_modelmap_no_overlap_rmsd_{N}.mrc   N in [0,100,200,500,1000,1500,2000]  (pm)

Output  (written to data/processed/):
  figure_3c.csv

CSV schema: freq | rmsd_0A | rmsd_1A | rmsd_2A | rmsd_5A | rmsd_10A | rmsd_15A | rmsd_20A
            (each profile is max-normalised)
"""

from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd

from locscale.include.emmer.ndimage.map_tools import compute_radial_profile_simple
from locscale.include.emmer.ndimage.profile_tools import frequency_array

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSD: picometres for filenames, Angstroms for column labels
RMSD_PM = [0, 100, 200, 500, 1000, 1500, 2000]
RMSD_A  = [0,   1,   2,   5,   10,   15,   20]

print("Processing Figure 3c ...")

freq     = None
profiles = {}

for rmsd_pm, rmsd_a in zip(RMSD_PM, RMSD_A):
    mrc_path = DATA_DIR / "model_maps" / f"pdb6y5a_modelmap_no_overlap_rmsd_{rmsd_pm}.mrc"
    with mrcfile.open(str(mrc_path)) as mrc:
        data = mrc.data.copy()
        apix = float(mrc.voxel_size.x)

    rp = compute_radial_profile_simple(data)
    rp_norm = rp / rp.max()

    if freq is None:
        freq = frequency_array(rp, apix)

    profiles[f"rmsd_{rmsd_a}A"] = rp_norm

df = pd.DataFrame({"freq": freq, **profiles})
df.to_csv(OUT_DIR / "figure_3c.csv", index=False)

print("  -> data/processed/figure_3c.csv")
print("Done.")
