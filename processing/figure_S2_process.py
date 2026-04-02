"""
Figure S2 processing: Wilson B-factor fit quality at specific atom positions.

Inputs  (place in data/):
  data/emd_10692_additional_1.map
  data/pdb6y5a_additional_refined.pdb

Output  (written to data/processed/):
  figure_S2.csv

CSV schema:
  freq | rp_C67_CZ | exp_C67_CZ | qfit_C67_CZ (scalar repeated) |
         rp_A303_CA | exp_A303_CA | qfit_A303_CA (scalar repeated)
"""

from pathlib import Path
import mrcfile
import gemmi
import numpy as np
import pandas as pd

from locscale.include.emmer.ndimage.profile_tools import (
    compute_radial_profile, frequency_array, estimate_bfactor_standard,
)
from locscale.include.emmer.ndimage.map_utils import convert_pdb_to_mrc_position, extract_window

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMMAP_PATH    = DATA_DIR / "emd_10692_additional_1.map"
PDB_PATH      = DATA_DIR / "pdb6y5a_additional_refined.pdb"
WINDOW_A      = 25
WILSON_CUTOFF = 10
FSC_RES       = 2.8

print("Processing Figure S2 ...")

with mrcfile.open(str(EMMAP_PATH)) as mrc:
    apix = float(mrc.voxel_size.x)
window_pix = int(round(WINDOW_A / apix))


def get_profile_fit(chain_name, res_seqid, atom_name):
    st    = gemmi.read_structure(str(PDB_PATH))
    emmap = mrcfile.open(str(EMMAP_PATH)).data
    pos   = None
    for res in st[0][chain_name]:
        if res.seqid.num == res_seqid:
            for atom in res:
                if atom.name == atom_name:
                    pos = atom.pos.tolist()
    mrc_pos = convert_pdb_to_mrc_position([pos], apix)[0]
    window  = extract_window(emmap, mrc_pos, size=window_pix)
    rp      = compute_radial_profile(window)
    freq    = frequency_array(rp, apix)
    bfactor, amp, qfit = estimate_bfactor_standard(
        freq, rp, wilson_cutoff=WILSON_CUTOFF, fsc_cutoff=FSC_RES,
        return_amplitude=True, return_fit_quality=True, standard_notation=True,
    )
    exp_fit = amp * np.exp(-0.25 * bfactor * freq ** 2)
    return freq, rp, exp_fit, round(qfit, 2)


freq, rp1, exp1, qfit1 = get_profile_fit("C", 67,  "CZ")
_,    rp2, exp2, qfit2 = get_profile_fit("A", 303, "CA")

n = len(freq)
df = pd.DataFrame({
    "freq":         freq,
    "rp_C67_CZ":    rp1,
    "exp_C67_CZ":   exp1,
    "qfit_C67_CZ":  [qfit1] * n,
    "rp_A303_CA":   rp2,
    "exp_A303_CA":  exp2,
    "qfit_A303_CA": [qfit2] * n,
})
df.to_csv(OUT_DIR / "figure_S2.csv", index=False)

print("  -> data/processed/figure_S2.csv")
print("Done.")
