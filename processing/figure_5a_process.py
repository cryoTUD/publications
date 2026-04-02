"""
Figure 5a processing: ensemble radial profiles from secondary-structure PDB database.

Inputs  (place in data/secondary_structure_analysis/):
  helix_profiles_process_0.pickle
  sheet_profiles_process_1.pickle
  dna_profiles_process_2.pickle
  rna_profiles_process_3.pickle
  selected_pdb.pickle

Output  (written to data/processed/):
  figure_5a.csv

CSV schema:
  freq |
  helix_mean | helix_std | helix_N |
  sheet_mean | sheet_std | sheet_N |
  dna_mean   | dna_std   | dna_N   |
  rna_mean   | rna_std   | rna_N
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from locscale.include.emmer.ndimage.profile_tools import frequency_array, estimate_bfactor_standard
from locscale.include.emmer.pdb.pdb_tools import find_wilson_cutoff

import os 

ROOT_DIR = Path(os.environ["ElectronScattering2022_ROOT"])
SS_DIR    = ROOT_DIR / "data" / "secondary_structure_analysis"
OUT_DIR   = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APIX         = 0.5
PROFILE_SIZE = 256

print("Processing Figure 5a ...")


def verify_profile(freq, amplitude):
    if np.isnan(amplitude).any() or np.any(amplitude <= 0):
        return False
    if len(amplitude) != PROFILE_SIZE:
        return False
    wilson_cutoff = find_wilson_cutoff(num_atoms=float(amplitude[0]))
    bfactor = estimate_bfactor_standard(
        freq, amplitude=amplitude,
        wilson_cutoff=wilson_cutoff, fsc_cutoff=1,
        standard_notation=True,
    )
    return bfactor <= 10


def clean_profiles(pickle_path, selected_pdbs):
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)
    freq = frequency_array(profile_size=PROFILE_SIZE, apix=APIX)
    cleaned = {}
    for name, entry in raw.items():
        amplitude = np.array(entry["amplitude"])
        # Derive PDB ID from filename
        parts = name.split("_")
        pdb_id = parts[1] if parts[0] == "pdb" else parts[0]
        if pdb_id not in selected_pdbs:
            continue
        if not (np.isfinite(amplitude).all() and not np.isnan(amplitude).any()):
            continue
        if not verify_profile(freq, amplitude):
            continue
        cleaned[name] = amplitude
    return cleaned


with open(SS_DIR / "selected_pdb.pickle", "rb") as f:
    selected_pdbs = set(pickle.load(f))

groups = {
    "helix": SS_DIR / "helix_profiles_process_0.pickle",
    "sheet": SS_DIR / "sheet_profiles_process_1.pickle",
    "dna":   SS_DIR / "dna_profiles_process_2.pickle",
    "rna":   SS_DIR / "rna_profiles_process_3.pickle",
}

freq = frequency_array(profile_size=PROFILE_SIZE, apix=APIX)
records = {"freq": freq}

for gname, pkl in groups.items():
    profiles = clean_profiles(pkl, selected_pdbs)
    arr = np.array(list(profiles.values()))   # shape (N, 256)
    records[f"{gname}_mean"] = arr.mean(axis=0)
    records[f"{gname}_std"]  = arr.std(axis=0)
    records[f"{gname}_N"]    = np.full(PROFILE_SIZE, len(arr))
    print(f"  {gname}: N={len(arr)}")

pd.DataFrame(records).to_csv(OUT_DIR / "figure_5a.csv", index=False)
print("  -> data/processed/figure_5a.csv")
print("Done.")
