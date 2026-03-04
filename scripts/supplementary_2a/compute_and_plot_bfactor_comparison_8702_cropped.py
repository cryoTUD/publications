## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import random
import pickle
from tqdm import tqdm
from scipy.stats import ks_2samp, invgamma
import matplotlib.pyplot as plt
import gemmi 
from locscale.include.emmer.pdb.pdb_utils import get_bfactors, shift_bfactors_by_probability
from locscale.include.emmer.ndimage.map_utils import load_map
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams, plot_correlations, get_2d_jointplot_with_text
from s2a_utils import get_atomic_bfactor_correlation

# Set random seed
np.random.seed(42)
random.seed(42)

def fit_inverse_gamma(bfactors):
    # Fit inverse gamma distribution
    shape, loc, scale = invgamma.fit(bfactors, floc=0)
    return shape, loc, scale

def main():
    import gemmi 
    data_archive_path = setup_environment()
    suffix = ""
    # Set input paths (as in notebook)
    output_folder = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "bfactor_refinement_all_using_halfmaps", "8702_5vkq")
    pseudo_path = os.path.join(output_folder, "model_free", "emd_8702_FDR_confidence_final_gradient_pseudomodel_proper_element_composition_shifted.mmcif")
    #pseudo_path = "/tudelft/abharadwaj1/staff-bulk/tnw/bn/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_model_free_C1/8702_5vkq/emd_8702_model_free_locscale_processing_C1/emd_8702_FDR_confidence_final_gradient_pseudomodel_proper_element_composition_shifted_bfactors.pdb"
    atomic_path = os.path.join(output_folder, "model_based", "PDB_5vkq_unrefined_shifted_servalcat_refined_servalcat_refined.cif")
    #atomic_path = "/home/abharadwaj1/papers/elife_paper/figure_information/data/maps/emd_8702/model_based/servalcat_0.4.105/refined.pdb"
    
    #mask_path = os.path.join(data_archive_path, "processed", "pdbs", "figure_2", "bfactor_comparison_pseudo_atomic", "emd_8702_difference_mask_not_micelle.mrc")
    pseudo_shifted_path = pseudo_path.replace(".cif", "_shifted.mmcif")
    atomic_shifted_path = atomic_path.replace(".mmcif", "_shifted.mmcif")
    output_folder = os.path.join(data_archive_path, "structured_data", "supplementary_2a")
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a", "temp")
    plot_output_path = os.path.join(plot_output_folder, "emd_8702_bfactor_correlation.pdf")
    plot_output_jointplot_path = os.path.join(plot_output_folder, "emd_8702_bfactor_correlation_jointplot.pdf")
    create_folders_if_they_do_not_exist(output_folder)

    # Output files
    json_output_path = os.path.join(output_folder, "emd_8702_bfactor_distribution_data.json")
    pickle_output_path = os.path.join(output_folder, "emd_8702_bfactor_distribution_data.pickle")

    assert_paths_exist(pseudo_path, atomic_path)

    pseudo_shifted_bfactor_st = shift_bfactors_by_probability(pseudo_path)[0]
    pseudo_shifted_bfactor_st.make_mmcif_document().write_file(pseudo_shifted_path)
    atomic_shifted_bfactor_st = shift_bfactors_by_probability(atomic_path)[0]
    atomic_shifted_bfactor_st.make_mmcif_document().write_file(atomic_shifted_path)

    # Get B-factors
    bfactor_comparison = get_atomic_bfactor_correlation(pseudo_shifted_path, atomic_shifted_path)
    # pseudo_bfactors = get_bfactors(pseudo_shifted_path)
    # atomic_bfactors = get_bfactors(atomic_shifted_path)
    pseudo_bfactors = np.array([x[1] for x in bfactor_comparison.values()]) 
    atomic_bfactors = np.array([x[0] for x in bfactor_comparison.values()]) 

    print("Minimum B-factor (pseudo):", np.min(pseudo_bfactors))
    print("Maximum B-factor (pseudo):", np.max(pseudo_bfactors))
    print("Range of B-factors (pseudo):", np.ptp(pseudo_bfactors))
    print("Minimum B-factor (atomic):", np.min(atomic_bfactors))
    print("Maximum B-factor (atomic):", np.max(atomic_bfactors))
    print("Range of B-factors (atomic):", np.ptp(atomic_bfactors))

    figsize_mm = (30, 20)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
        plot_correlations(pseudo_bfactors, atomic_bfactors, \
            "Pseudo-atomic model", "Hybrid model", fontscale=0.5, \
            xticks = None, yticks = None, \
            filepath=plot_output_path\
        )

        get_2d_jointplot_with_text(pseudo_bfactors, atomic_bfactors, \
            "Pseudo-atomic model", "Hybrid model", \
            save_path=plot_output_jointplot_path, \
            figsize_mm=figsize_mm, 
        )
    # Store data
    output_data = {
        "pseudo_bfactors": pseudo_bfactors,
        "atomic_bfactors": atomic_bfactors,
    }


    with open(pickle_output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved pickle to {pickle_output_path}")
    print(f"Saved plot to {plot_output_path}")
    print(f"Jointplot saved to {plot_output_jointplot_path}")

if __name__ == "__main__":
    main()
