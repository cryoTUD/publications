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
import matplotlib.pyplot as plt
import pickle

# Custom imports
from locscale.preprocessing.pseudomodel_classes import extract_model_from_mask
from locscale.include.emmer.ndimage.map_utils import load_map, measure_mask_parameters, save_as_mrc
from locscale.include.emmer.ndimage.filter import get_cosine_mask
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region, estimate_global_bfactor_map_standard
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps, calculate_amplitude_correlation_maps
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array

# Import helper functions  
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set global variables
SAVE_DATA = True
figure_number = 2 
## SETUP 
def main():
    data_archive_path = setup_environment()

    # DEFINE THE PATHS 
    #input_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/pseudomodel_during_iterations/hybrid_pseudomodel_iterations"
    input_maps_folder = os.path.join(data_archive_path, "raw","maps", "figure_2")
    output_maps_folder = os.path.join(data_archive_path, "processed", "maps")
    input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")
    output_pdbs_folder = os.path.join(data_archive_path, "processed", "pdbs")
    output_general_folder = os.path.join(data_archive_path, "processed", "general")
    output_structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data")
    suffix = "_3061"
    output_pseudomodel_structure_folder = os.path.join(output_pdbs_folder, f"figure_{figure_number}", f"pseudomodel_structures_iterations{suffix}")
    output_pseudomodel_map_folder = os.path.join(output_maps_folder, f"figure_{figure_number}", f"pseudomodel_maps{suffix}")
    output_folder_for_data = os.path.join(output_structured_data_folder, f"figure_{figure_number}")
    output_processing_folder = os.path.join(output_general_folder, f"figure_{figure_number}_processed")
    
    create_folders_if_they_do_not_exist(\
        output_pseudomodel_structure_folder,\
        output_pseudomodel_map_folder,\
        output_folder_for_data,\
        output_processing_folder\
    )

    # Output paths
    save_fig_path = os.path.join(output_processing_folder, f"cropped_map_slices{suffix}.png")
    path_to_store_fsc_average_pickle = os.path.join(output_folder_for_data, f"fsc_average_iterations{suffix}.pickle")
    path_to_store_radial_profile_pickle = os.path.join(output_folder_for_data, f"radial_profile_iterations{suffix}.pickle")

    # DEFINE THE PATHS
    # emmap_path = os.path.join(input_maps_folder, "figure_2_emd_8702_unsharpened_map.mrc")
    # mask_path = os.path.join(input_maps_folder, "figure_2_emd_8702_FDR_confidence_final.map")
    # pdb_path = os.path.join(input_pdbs_folder, "figure_2_cropped_model_pdb_5vkq.pdb")
    # simmap_path = os.path.join(input_maps_folder, "figure_2_pdb5vkq_uniform_bfactor.mrc")
    emmap_path = os.path.join(input_maps_folder, "EMD_3061_unfiltered.mrc")
    mask_path = os.path.join(input_maps_folder, "EMD_3061_unfiltered_confidenceMap.mrc")
    pdb_path = os.path.join(input_maps_folder, "5a63.pdb")
    simmap_path = os.path.join(input_maps_folder, "5a63_uniform_bfactor.mrc")

    if not os.path.exists(simmap_path):
        from locscale.include.emmer.pdb.pdb_to_map import pdb2map 
        from locscale.include.emmer.pdb.pdb_utils import set_atomic_bfactors
        import gemmi 
        emmap, apix = load_map(emmap_path)
        output_pdb_path = pdb_path[:-4] + "_uniform_bfactor.cif"
        set_atomic_bfactors(in_model_path=pdb_path, b_iso=50, out_file_path=output_pdb_path)
        assert os.path.exists(output_pdb_path), f"Path does not exist: {output_pdb_path}"
        simmap = pdb2map(output_pdb_path, apix=apix, size=emmap.shape)
        save_as_mrc(simmap, simmap_path, apix=apix, verbose=True)

#    output_pseudomodel_map_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/pseudomodel_during_iterations/hybrid_pseudomodel_iterations/pseudomodel_maps"

    pseudomodel_paths = [os.path.join(output_pseudomodel_structure_folder, f) for f in os.listdir(output_pseudomodel_structure_folder) if f.endswith(".mmcif")]
    pseudomodel_map_paths = [os.path.join(output_pseudomodel_map_folder, f) for f in os.listdir(output_pseudomodel_map_folder) if f.endswith(".mrc") and "pseudoatomic_model" in f]
    pseudomodel_map_paths_dict = {
        int(os.path.basename(f).split("_")[2].split(".")[0]) : f for f in pseudomodel_map_paths
    }

    # Sort the paths based on keys
    pseudomodel_map_paths_sorted_dict = dict(sorted(pseudomodel_map_paths_dict.items()))
    
    for path in pseudomodel_paths + pseudomodel_map_paths + \
        [emmap_path, mask_path, pdb_path, simmap_path]:
        assert os.path.exists(path), f"Path does not exist: {path}"

    
    # Load the data
    emmap, apix = load_map(emmap_path)
    simmap = load_map(simmap_path)[0]
    fdr_mask = load_map(mask_path)[0]

    # Compute the softmask
    
    filtered_fdr_mask = uniform_filter(fdr_mask,size=3)
    binarised_fdr_mask = (fdr_mask>0.99).astype(np.int_)
    softmask = get_cosine_mask(binarised_fdr_mask,5)

    emmap_masked = emmap * softmask
    simmap_masked = softmask * simmap

    # Extract cropping bounds
    # center_z = 150
    # center_y = 200
    # center_x = 200
    center_z = 90
    center_y = 90
    center_x = 90

    center = (center_z,center_y,center_x)
    # width = 150
    width = 60

    height = width
    depth = width
    start_index_z = center_z - depth//2
    end_index_z = center_z + depth//2
    start_index_y = center_y - height//2
    end_index_y = center_y + height//2
    start_index_x = center_x - width//2
    end_index_x = center_x + width//2

    reference_window = emmap_masked[\
        start_index_z:end_index_z,\
        start_index_y:end_index_y,\
        start_index_x:end_index_x\
    ]

    simmap_window = simmap_masked[\
        start_index_z:end_index_z,\
        start_index_y:end_index_y,\
        start_index_x:end_index_x\
    ]
    
    fig, ax = plt.subplots(1,3,figsize=(8,8))
    index = width//2

    ax[0].imshow(reference_window[index,:,:] , cmap='gray')
    ax[0].set_title("z-slice at index {}".format(index))
    ax[1].imshow(reference_window[:,index,:] , cmap='gray')
    ax[1].set_title("y-slice at index {}".format(index))
    ax[2].imshow(reference_window[:,:,index], cmap='gray')
    ax[2].set_title("x-slice at index {}".format(index))

    
    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()

    # Compute the radial profile 

    # reference_window = simmap_masked[\
    #     start_index_z:end_index_z,\
    #     start_index_y:end_index_y,\
    #     start_index_x:end_index_x\
    # ]

    radial_profile_iterations = {}
    fsc_average_iterations = {}
    for i in tqdm(range(len(pseudomodel_map_paths)), desc="Computing radial profile and FSC"):
        pseudomodel_map_path = pseudomodel_map_paths_sorted_dict[i]
        pseudomodel_map = load_map(pseudomodel_map_path)[0]
        pseudomodel_map_masked = pseudomodel_map * softmask
        pseudomodel_window = pseudomodel_map_masked[\
            start_index_z:end_index_z,\
            start_index_y:end_index_y,\
            start_index_x:end_index_x\
        ]

        fsc_window = calculate_fsc_maps(reference_window, pseudomodel_window)
        fsc_average_iterations[i] = np.mean(fsc_window)

        rp_pseudomodel = compute_radial_profile(pseudomodel_window)
        radial_profile_iterations[i] = rp_pseudomodel
    
    freq = frequency_array(rp_pseudomodel, apix)
    
    rp_simulated_window = compute_radial_profile(simmap_window)
    rp_simulated_window_sharpen = rp_simulated_window * np.exp(freq * 20 / 4)
    rp_simulated_window_sharpen[0] *= 0.55 
    rp_simulated_window_sharpen_normalized = rp_simulated_window_sharpen / rp_simulated_window_sharpen.max()
    radial_profile_iterations["other_info"] = {
        "reference" : rp_simulated_window,
        "reference_normalised" : rp_simulated_window_sharpen_normalized,
        "freq" : freq,
        "apix" : apix
    }

    if SAVE_DATA:
        # Pickle the data 
        with open(path_to_store_fsc_average_pickle, "wb") as f:
            pickle.dump(fsc_average_iterations, f)
        
        with open(path_to_store_radial_profile_pickle, "wb") as f:
            pickle.dump(radial_profile_iterations, f)


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


    




    

