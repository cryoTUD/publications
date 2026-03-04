# Script to perturb and refine a given structure

## IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])

import numpy as np

from locscale.include.emmer.pdb.pdb_utils import shake_pdb_within_mask
from genericpath import isfile
import os
import sys
import shutil
import subprocess
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import gemmi
import random 

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

perturbed_refinement_folder_main = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/processed/pdbs/figure_2/perturbation_study"

def copy_files_to_folder(file, folder):
    # if copied file already exists then ignore 
    test_copied_path = os.path.join(folder, os.path.basename(file))
    if os.path.exists(test_copied_path):
        print("File already exists: {}".format(test_copied_path))
        return test_copied_path
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    new_path = shutil.copy(file, folder)
    return new_path

def replace_atoms_with_pseudo_atoms(atomic_model_path):
    st = gemmi.read_structure(atomic_model_path)
    
    for cra in st[0].all():
        element_choice = np.random.choice(["C","O","N"], p=[0.63,0.2,0.17])
        cra.atom.name = element_choice
        cra.atom.element = gemmi.Element(element_choice)
        amino_acid_residues = ['TYR','THR','SER','PRO','PHE','MET','LEU','ILE','HIS','GLY','GLU','GLN','ASP','ASN','ALA','ARG','TRP','CYS']
        cra.residue.name = np.random.choice(amino_acid_residues)
    
    return st

input_unsharpened_map = sys.argv[1]
input_refined_pdb = sys.argv[2]
print(sys.argv)
print(input_unsharpened_map)
print(input_refined_pdb)
input_unsharpened_map = os.path.abspath(input_unsharpened_map)
input_refined_pdb = os.path.abspath(input_refined_pdb)

assert os.path.exists(input_unsharpened_map), "Unsharpened map does not exist: {}".format(input_unsharpened_map)
assert os.path.exists(input_refined_pdb), "Refined pdb does not exist: {}".format(input_refined_pdb)

emmap_basename = os.path.basename(input_unsharpened_map).split(".")[0]
JOB_ID = emmap_basename
perturbed_refinement_folder = os.path.join(perturbed_refinement_folder_main, JOB_ID)
if not os.path.exists(perturbed_refinement_folder):
    os.makedirs(perturbed_refinement_folder)

emmap_path_local = copy_files_to_folder(input_unsharpened_map, perturbed_refinement_folder)
refined_pdb_path_local = copy_files_to_folder(input_refined_pdb, perturbed_refinement_folder)
atomic_model_to_pseudomodel = replace_atoms_with_pseudo_atoms(refined_pdb_path_local)
atomic_model_to_pseudomodel_path = os.path.join(perturbed_refinement_folder, "atomic_model_to_pseudomodel.cif")
# atomic_model_to_pseudomodel.write_pdb(atomic_model_to_pseudomodel_path)
# write as mmCIF
atomic_model_to_pseudomodel.make_mmcif_document().write_file(atomic_model_to_pseudomodel_path)

# Get atomic model mask
from locscale.include.emmer.ndimage.map_tools import get_atomic_model_mask
atomic_model_mask_path = os.path.join(perturbed_refinement_folder, "atomic_model_mask.mrc")
atomic_model_mask_path = get_atomic_model_mask(emmap_path=emmap_path_local, pdb_path=refined_pdb_path_local, \
                                               output_filename = atomic_model_mask_path, save_files=True)

# %%
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.preprocessing.headers import run_servalcat_iterative
refinement_paths_input = {}
emmap, apix = load_map(emmap_path_local)
perturbation_magnitude = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]

# Measure map to model correlation
#%% 
from locscale.utils.map_quality import map_quality_pdb

metric = map_quality_pdb(emmap_path_local, atomic_model_mask_path, refined_pdb_path_local)
print(metric)
assert metric > 0.2, "Map to model correlation is too low: {}".format(metric)

for perturbation in perturbation_magnitude:
    perturbation_folder = os.path.join(perturbed_refinement_folder, "perturbation_{}_pm".format(int(perturbation*10)))
    if not os.path.exists(perturbation_folder):
        os.makedirs(perturbation_folder)
    perturbed_pdb_path = os.path.join(perturbation_folder, "perturbed_rmsd_{}_pm.pdb".format(int(perturbation*10)))
    perturbed_pdb = shake_pdb_within_mask(pdb_path = atomic_model_to_pseudomodel, mask_path = atomic_model_mask_path, \
                                          rmsd_magnitude = perturbation, use_pdb_mask = False)
    perturbed_pdb.write_pdb(perturbed_pdb_path)
    copied_emmap_path = copy_files_to_folder(emmap_path_local, perturbation_folder)

    refinement_paths_input[perturbation] = {
        "model_path": perturbed_pdb_path,
        "map_path": copied_emmap_path,
        "resolution": round(apix*2),
        "num_iter": 2, 
        "pseudomodel_refinement" : True, 
    }

# Atomic model refinement
atomic_model_directory = os.path.join(perturbed_refinement_folder, "atomic_model_refinement")
if not os.path.exists(atomic_model_directory):
    os.makedirs(atomic_model_directory)

copied_emmap_path = copy_files_to_folder(emmap_path_local, atomic_model_directory)
copied_pdb_path = copy_files_to_folder(refined_pdb_path_local, atomic_model_directory)

refinement_paths_input["atomic_model"] = {
    "model_path": copied_pdb_path,
    "map_path": copied_emmap_path,
    "resolution": round(apix*2),
    "num_iter": 2,
    "pseudomodel_refinement" : False,
}


# %%
from joblib import Parallel, delayed

refined_results = Parallel(n_jobs=9)(delayed(run_servalcat_iterative)(
    model_path = refinement_paths_input[key]["model_path"],            
    map_path = refinement_paths_input[key]["map_path"],
    resolution = refinement_paths_input[key]["resolution"],
    num_iter = refinement_paths_input[key]["num_iter"],
    pseudomodel_refinement = refinement_paths_input[key]["pseudomodel_refinement"],
)
for key in list(refinement_paths_input.keys()))

# %%







