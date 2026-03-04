## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import random
import gemmi
from tqdm import tqdm

from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist

# Set random seeds
np.random.seed(42)
random.seed(42)

# Function to count residues in a CIF model file
def get_num_residues(filepath):
    doc = gemmi.cif.read_file(filepath)
    block = doc.sole_block()
    residues = set()
    for asym_id, seq_id in zip(block.find_loop('_atom_site.label_asym_id'),
                               block.find_loop('_atom_site.label_seq_id')):
        if seq_id != '.' and asym_id != '.':
            residues.add((asym_id, seq_id))
    return len(residues)

def main():
    data_archive_path = setup_environment()

    input_folder = os.path.join(data_archive_path, "processed", "pdbs", "figure_7", "model_angelo_predictions_new_version")
    output_folder = os.path.join(data_archive_path, "structured_data", "figure_7")
    create_folders_if_they_do_not_exist(output_folder)

    output_path = os.path.join(output_folder, "model_angelo_residue_counts.pickle")

    model_angelo_results = {
        "num_residues_using_hybrid": {},
        "num_residues_using_hybrid_raw": {},
        "num_residues_using_unsharpened": {},
        "num_residues_using_unsharpened_raw": {},
    }

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    for entry in tqdm(os.listdir(input_folder)):
        entry_path = os.path.join(input_folder, entry)
        if not os.path.isdir(entry_path):
            continue

        emdb_id, pdb_id = entry.split('_')

        hybrid_folder = os.path.join(entry_path, f"emd_{emdb_id}_model_angelo_hybrid")
        unsharpened_folder = os.path.join(entry_path, f"emd_{emdb_id}_model_angelo_unsharpened")

        hybrid_path = os.path.join(hybrid_folder, f"emd_{emdb_id}_model_angelo_hybrid.cif")
        hybrid_raw_path = os.path.join(hybrid_folder, f"emd_{emdb_id}_model_angelo_hybrid_raw.cif")

        unsharpened_path = os.path.join(unsharpened_folder, f"emd_{emdb_id}_model_angelo_unsharpened.cif")
        unsharpened_raw_path = os.path.join(unsharpened_folder, f"emd_{emdb_id}_model_angelo_unsharpened_raw.cif")

        if os.path.exists(hybrid_path):
            model_angelo_results["num_residues_using_hybrid"][entry] = get_num_residues(hybrid_path)
        if os.path.exists(hybrid_raw_path):
            model_angelo_results["num_residues_using_hybrid_raw"][entry] = get_num_residues(hybrid_raw_path)
        if os.path.exists(unsharpened_path):
            model_angelo_results["num_residues_using_unsharpened"][entry] = get_num_residues(unsharpened_path)
        if os.path.exists(unsharpened_raw_path):
            model_angelo_results["num_residues_using_unsharpened_raw"][entry] = get_num_residues(unsharpened_raw_path)

    with open(output_path, "wb") as f:
        pickle.dump(model_angelo_results, f)

    print(f"Saved residue count results to {output_path}")

if __name__ == "__main__":
    main()
