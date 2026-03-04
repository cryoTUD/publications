## IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import numpy as np
import gemmi
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import random 
from tqdm import tqdm
from datetime import datetime
# Custom imports
from locscale.preprocessing.pseudomodel_classes import extract_model_from_mask
from locscale.include.emmer.ndimage.map_utils import load_map, measure_mask_parameters, save_as_mrc
from locscale.include.emmer.ndimage.filter import get_cosine_mask
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region, estimate_global_bfactor_map_standard

# Import helper functions 
from figure_2_functions import gradient_solver, create_modmap
from scripts.utils.general import setup_environment

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set global variables
SOLVE_PSEUDOMODEL = False
SIMULATE_MAPS = True
figure_number = 2
## SETUP 
def main():
    data_archive_path = setup_environment()

    # DEFINE THE PATHS 
    #input_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/pseudomodel_during_iterations/hybrid_pseudomodel_iterations"
    input_maps_folder = os.path.join(data_archive_path, "raw","maps", "figure_2")
    output_maps_folder = os.path.join(data_archive_path, "processed", "maps")
    #input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")
    output_pdbs_folder = os.path.join(data_archive_path, "processed", "pdbs")
    output_general_folder = os.path.join(data_archive_path, "processed", "general")

    output_pseudomodel_structure_folder = os.path.join(output_pdbs_folder, f"figure_{figure_number}", "pseudomodel_structures_iterations")
    os.makedirs(output_pseudomodel_structure_folder, exist_ok=True) 

    output_pseudomodel_map_folder = os.path.join(output_maps_folder, f"figure_{figure_number}", "pseudomodel_maps")
    os.makedirs(output_pseudomodel_map_folder, exist_ok=True)

    output_processing_folder = os.path.join(output_general_folder, f"figure_{figure_number}_processed")
    
    # DEFINE THE PATHS
    # emmap_path = os.path.join(input_maps_folder, "EMD_3061_unfiltered.mrc")
    # mask_path = os.path.join(input_maps_folder, "EMD_3061_unfiltered_confidenceMap.mrc")
    # pdb_path = os.path.join(input_maps_folder, "5a63.pdb")
    emmap_path = os.path.join(input_maps_folder, "figure_2_emd_8702_unsharpened_map.mrc")
    mask_path = os.path.join(input_maps_folder, "figure_2_emd_8702_FDR_confidence_final.map")
    pdb_path = os.path.join(input_maps_folder, "figure_2_cropped_model_pdb_5vkq.pdb")
    assert os.path.exists(emmap_path), f"Path does not exist: {emmap_path}"
    assert os.path.exists(mask_path), f"Path does not exist: {mask_path}"
    assert os.path.exists(pdb_path), f"Path does not exist: {pdb_path}"

    # DEFINE PARAMETERS 
    threshold = 0.99 
    n_jobs = 10

    # GET THE INITIAL PSEUDO MODEL
    global_bfactor_map = estimate_global_bfactor_map_standard(emmap_path=emmap_path, wilson_cutoff=9, fsc_cutoff=3.55)
    print("Global B-factor map estimated: ", global_bfactor_map)
    emmap, apix = load_map(emmap_path)
    
    difference_mask = find_unmodelled_mask_region(fdr_mask_path=mask_path, pdb_path=pdb_path, fdr_threshold=0.99, \
                                            atomic_mask_threshold=0.5, averaging_window_size=3, fsc_resolution=3.55)
    
    binarised_fdr_mask = difference_mask > 0.5
    num_atoms, _ = measure_mask_parameters(mask=binarised_fdr_mask, apix=apix, edge_threshold=0.5)

    pseudomodel = extract_model_from_mask(binarised_fdr_mask,num_atoms,threshold=threshold)

    # GET THE GRADIENT PARAMETERS
    emmap_shape = emmap.shape
    unitcell = gemmi.UnitCell(emmap_shape[0]*apix,emmap_shape[1]*apix,emmap_shape[2]*apix,90,90,90)

    outputlogfilepath = os.path.join(os.path.dirname(emmap_path),"pseudomodel_log.txt")
    output_file = open(outputlogfilepath,"w")

    gz,gy,gx = np.gradient(emmap)
    masked_grad_magnitude = binarised_fdr_mask * np.sqrt(gx**2 + gy**2 + gz**2)
    max_gradient = masked_grad_magnitude.max()

    g = round(100 / max_gradient)
    scale_lj = 1
    scale_map = 1
    friction = 10

    if SOLVE_PSEUDOMODEL:
        # RUN THE GRADIENT SOLVER
        arranged_points = gradient_solver(
            emmap, gx, gy, gz, pseudomodel, \
                save_file_folder=output_pseudomodel_structure_folder, \
                g=g,friction=friction,min_dist_in_angst=1.2,apix=apix,dt=0.1,\
                capmagnitude_lj=100,epsilon=1,scale_lj=scale_lj,capmagnitude_map=100,scale_map=scale_map,\
                total_iterations=50, compute_map=None,emmap_path=None,mask_path=None,\
                returnPointsOnly=True,integration='verlet',verbose=False, myoutput=output_file\
        )

    output_file.close()

    # SIMULATE THE MAPS
    pseudomodel_paths = [os.path.join(output_pseudomodel_structure_folder, f) \
                        for f in os.listdir(output_pseudomodel_structure_folder) if f.endswith(".mmcif")]
    

    if SIMULATE_MAPS:
        n_jobs = 10
        Parallel(n_jobs=n_jobs,verbose=10)(\
            delayed(create_modmap)(\
            input_pdb_path, apix, emmap_shape, output_pseudomodel_map_folder, global_bfactor_map, symmetry="C4") \
            for input_pdb_path in tqdm(pseudomodel_paths, desc="Simulating maps")
        )

if __name__ == "__main__":
    # Print the start time
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {processing_time}")
    print("="*80)


    




    

