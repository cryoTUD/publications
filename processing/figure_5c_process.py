"""
Figure 5c processing: apply theoretical secondary-structure profiles to repair maps.

Inputs  (place in data/apply_average_profile/):
  proper_sidechain_b300.mrc
  proper_sidechain_b50.mrc

Outputs (written to data/processed/):
  figure_5c_profiles.csv    – original vs repaired radial profiles
  figure_5c_theoretical.csv – helix / sheet / average theoretical profiles
  figure_S5b_fsc.csv        – FSC between repaired maps

figure_5c_profiles.csv schema:
  freq | rp_b300 | rp_repaired_helix | rp_repaired_sheet | rp_repaired_average

figure_5c_theoretical.csv schema:
  freq | helix_theoretical | sheet_theoretical | average_theoretical

figure_S5b_fsc.csv schema:
  freq | fsc_helix_vs_sheet | fsc_helix_vs_average
"""

from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd

from locscale.include.emmer.ndimage.profile_tools import (
    frequency_array, get_theoretical_profile, scale_profiles,
)
from locscale.include.emmer.ndimage.map_tools import (
    compute_radial_profile_simple, set_radial_profile_to_volume,
)
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
IN_DIR   = ROOT_DIR / "data" / "apply_average_profile"
OUT_DIR  = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Processing Figure 5c (and S5a, S5b) ...")

emmap_path = str(IN_DIR / "proper_sidechain_b300.mrc")
modmap_path = str(IN_DIR / "proper_sidechain_b50.mrc")

with mrcfile.open(emmap_path) as mrc:
    apix  = float(mrc.voxel_size.x)
    emmap = mrc.data.copy()
with mrcfile.open(modmap_path) as mrc:
    modmap = mrc.data.copy()

rp_emmap  = compute_radial_profile_simple(emmap)
rp_modmap = compute_radial_profile_simple(modmap)
freq = frequency_array(rp_emmap, apix)

helix_th  = get_theoretical_profile(len(rp_emmap), apix=apix, profile_type="helix")[1]
sheet_th  = get_theoretical_profile(len(rp_emmap), apix=apix, profile_type="sheet")[1]
avg_th    = (helix_th + sheet_th) / 2.0

scaled_helix = scale_profiles((freq, rp_modmap), (freq, helix_th), wilson_cutoff=10, fsc_cutoff=2)[1]
scaled_sheet = scale_profiles((freq, rp_modmap), (freq, sheet_th), wilson_cutoff=10, fsc_cutoff=2)[1]
scaled_avg   = (scaled_helix + scaled_sheet) / 2.0

rep_helix = set_radial_profile_to_volume(emmap, scaled_helix)
rep_sheet = set_radial_profile_to_volume(emmap, scaled_sheet)
rep_avg   = set_radial_profile_to_volume(emmap, scaled_avg)

rp_rep_h = compute_radial_profile_simple(rep_helix)
rp_rep_s = compute_radial_profile_simple(rep_sheet)
rp_rep_a = compute_radial_profile_simple(rep_avg)

fsc_hs = calculate_fsc_maps(input_map_1=rep_helix, input_map_2=rep_sheet)
fsc_ha = calculate_fsc_maps(input_map_1=rep_helix, input_map_2=rep_avg)
freq_fsc = frequency_array(fsc_hs, apix=apix)

# ── CSV 1: profiles ──────────────────────────────────────────────────────────
pd.DataFrame({
    "freq":               freq,
    "rp_b300":            rp_emmap,
    "rp_repaired_helix":  rp_rep_h,
    "rp_repaired_sheet":  rp_rep_s,
    "rp_repaired_average": rp_rep_a,
}).to_csv(OUT_DIR / "figure_5c_profiles.csv", index=False)

# ── CSV 2: theoretical profiles ──────────────────────────────────────────────
pd.DataFrame({
    "freq":               freq,
    "helix_theoretical":  helix_th,
    "sheet_theoretical":  sheet_th,
    "average_theoretical": avg_th,
}).to_csv(OUT_DIR / "figure_5c_theoretical.csv", index=False)

# ── CSV 3: FSC curves ────────────────────────────────────────────────────────
pd.DataFrame({
    "freq":                 freq_fsc,
    "fsc_helix_vs_sheet":   fsc_hs,
    "fsc_helix_vs_average": fsc_ha,
}).to_csv(OUT_DIR / "figure_S5b_fsc.csv", index=False)

print("  -> data/processed/figure_5c_profiles.csv")
print("  -> data/processed/figure_5c_theoretical.csv")
print("  -> data/processed/figure_S5b_fsc.csv")
print("Done.")
