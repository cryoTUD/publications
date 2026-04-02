"""
Figure 2f processing: local atomic B-factor vs Wilson B-factor correlation.

Inputs  (place in data/):
  data/pdb6y5a_additional_refined.pdb
  data/emd_10692_additional_1.map

Output  (written to data/processed/):
  figure_2f.csv

CSV schema: atomic_bfactor | wilson_bfactor   (1000 sampled atoms, unfiltered)
"""

from pathlib import Path
import mrcfile
import gemmi
import numpy as np
import pandas as pd
import random
random.seed(42)  # for reproducibility

from locscale.include.emmer.pdb.pdb_tools import get_all_atomic_positions, get_atomic_bfactor_window
from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position
from locscale.include.emmer.ndimage.map_tools import get_local_bfactor_emmap
from tqdm import tqdm
import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_PATH    = DATA_DIR / "pdb6y5a_additional_refined.pdb"
EMMAP_PATH  = DATA_DIR / "emd_10692_additional_1.map"
WINDOW_A    = 25
SAMPLE_SIZE = 1000
FSC_RES     = 2.8

print("Processing Figure 2f ...")

with mrcfile.open(str(EMMAP_PATH)) as mrc:
    apix = float(mrc.voxel_size.x)

window_pix = int(round(WINDOW_A / apix))
st = gemmi.read_structure(str(PDB_PATH))
all_positions = list(get_all_atomic_positions(gemmi_structure=st, as_dictionary=False))
pdb_centers  = random.sample(all_positions, SAMPLE_SIZE)
mrc_centers  = convert_pdb_to_mrc_position(pdb_centers, apix=apix)

atomic_bfactors = []
wilson_bfactors = []

for i, pdb_center in enumerate(tqdm(pdb_centers, desc="Sampling B-factors")):
    mrc_center = mrc_centers[i]
    ab = get_atomic_bfactor_window(st, pdb_center, WINDOW_A)
    wb = get_local_bfactor_emmap(
        str(EMMAP_PATH), mrc_center,
        fsc_resolution=FSC_RES, boxsize=window_pix,
        wilson_cutoff="traditional"
    )[0]
    atomic_bfactors.append(ab)
    wilson_bfactors.append(wb)

df = pd.DataFrame({
    "atomic_bfactor": atomic_bfactors,
    "wilson_bfactor": wilson_bfactors,
})
df.to_csv(OUT_DIR / "figure_2f.csv", index=False)

print("  -> data/processed/figure_2f.csv")
print("Done.")
