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

from locscale.include.emmer.pdb.pdb_utils import get_bfactors
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
    data_archive_path = setup_environment()

    # Set input paths (as in notebook)
    pseudo_path = "/home/abharadwaj1/papers/elife_paper/figure_information/data/maps/emd_3061/emd_3061_pseudomodel_within_atomic_mask_tight.pdb"
    atomic_path = "/home/abharadwaj1/papers/elife_paper/figure_information/data/maps/emd_3061/model_based/processing_files/5a63_shifted_servalcat_refined.pdb"
    output_folder = os.path.join(data_archive_path, "structured_data", "supplementary_2a")
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a")
    plot_output_path = os.path.join(plot_output_folder, "emd_3061_bfactor_distribution_pseudo_and_atomic.pdf")
    plot_output_jointplot_path = os.path.join(plot_output_folder, "emd_3061_bfactor_distribution_pseudo_and_atomic_jointplot.pdf")
    create_folders_if_they_do_not_exist(output_folder)

    # Output files
    json_output_path = os.path.join(output_folder, "emd_3061_bfactor_distribution_data.json")
    pickle_output_path = os.path.join(output_folder, "emd_3061_bfactor_distribution_data.pickle")

    assert_paths_exist(pseudo_path, atomic_path)

    # Get B-factors
    bfactor_comparison = get_atomic_bfactor_correlation(pseudo_path, atomic_path)
    pseudo_bfactors = np.array([x[1] for x in bfactor_comparison.values()]) 
    atomic_bfactors = np.array([x[0] for x in bfactor_comparison.values()]) 

    figsize_mm = (30, 20)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
        plot_correlations(pseudo_bfactors, atomic_bfactors, \
            "Pseudo-atomic model", "Atomic model", scatter=True,\
            figsize_cm = figsize_cm, fontscale=0.5, \
            xticks = [75, 125, 175], yticks = [130, 160, 190], \
            filepath=plot_output_path\
        )

        # Plot joint distribution
        get_2d_jointplot_with_text(
            pseudo_bfactors, atomic_bfactors,
            x_label="Pseudo-atomic model",
            y_label="Atomic model",
            save_path=plot_output_jointplot_path,
            figsize_mm=figsize_mm,
            fontsize=fontsize,
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

if __name__ == "__main__":
    main()
