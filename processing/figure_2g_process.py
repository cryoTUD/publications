"""
Figure 2g processing: local radial profiles at three atom positions.

Inputs  (place in data/):
  data/emd_10692_additional_1.map
  data/emd_10692_half_map_1_localResolutions.mrc
  data/pdb6y5a_additional_refined.pdb

Output  (written to data/processed/):
  figure_2g.csv

CSV schema:
  freq | rp_C67 | rp_B443 | rp_E332 |
  exp_C67 | exp_B443 | exp_E332 |
  fsc_res_C67 | fsc_res_B443 | fsc_res_E332   (scalar repeated)
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
DATA_DIR      = ROOT_DIR / "data"
OUT_DIR       = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMMAP_PATH    = DATA_DIR / "emd_10692_additional_1.map"
LOCAL_RES_PATH = DATA_DIR / "emd_10692_half_map_1_localResolutions.mrc"
PDB_PATH      = DATA_DIR / "pdb6y5a_additional_refined.pdb"
WINDOW_A      = 25
WILSON_CUTOFF = 10
FSC_RES       = 2.8

print("Processing Figure 2g ...")

with mrcfile.open(str(EMMAP_PATH)) as mrc:
    apix = float(mrc.voxel_size.x)
window_pix = int(round(WINDOW_A / apix))


def get_local_profile(chain_name, res_seqid):
    st         = gemmi.read_structure(str(PDB_PATH))
    emmap      = mrcfile.open(str(EMMAP_PATH)).data
    local_res  = mrcfile.open(str(LOCAL_RES_PATH)).data

    for res in st[0][chain_name]:
        if res.seqid.num == res_seqid:
            ca_pos = res.get_ca().pos.tolist()

    mrc_pos  = convert_pdb_to_mrc_position([ca_pos], apix)[0]
    window   = extract_window(emmap, mrc_pos, size=window_pix)
    fsc_val  = float(local_res[mrc_pos[0], mrc_pos[1], mrc_pos[2]])

    rp   = compute_radial_profile(window)
    freq = frequency_array(rp, apix)
    bfactor, amp, _ = estimate_bfactor_standard(
        freq, rp, wilson_cutoff=WILSON_CUTOFF, fsc_cutoff=FSC_RES,
        return_amplitude=True, return_fit_quality=True, standard_notation=True,
    )
    exp_fit = amp * np.exp(-0.25 * bfactor * freq ** 2)
    return freq, rp, exp_fit, round(fsc_val, 2)


freq, rp1, exp1, fsc1 = get_local_profile("C", 67)
_,    rp2, exp2, fsc2 = get_local_profile("B", 443)
_,    rp3, exp3, fsc3 = get_local_profile("E", 332)

n = len(freq)
df = pd.DataFrame({
    "freq":        freq,
    "rp_C67":      rp1,
    "rp_B443":     rp2,
    "rp_E332":     rp3,
    "exp_C67":     exp1,
    "exp_B443":    exp2,
    "exp_E332":    exp3,
    "fsc_res_C67":  [fsc1] * n,
    "fsc_res_B443": [fsc2] * n,
    "fsc_res_E332": [fsc3] * n,
})
df.to_csv(OUT_DIR / "figure_2g.csv", index=False)

print("  -> data/processed/figure_2g.csv")
print("Done.")
