## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, pretty_plot_confidence_interval

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed", "structured_data", "figure_2")
    input_filename = os.path.join(data_input_folder_main, "fsc_curves_test.pkl")  # input pickle file
    # other input folder 

    figure_output_folder_main = os.path.join(data_archive_path, "outputs", "figure_2")
    # other output folder
    assert_paths_exist(input_filename) # for input files
    create_folders_if_they_do_not_exist(figure_output_folder_main) # for output files
    
    output_filename = os.path.join(figure_output_folder_main, "fsc_curves_with_fdr_mask_many_maps.pdf")  # output plot preferably in pdf format

    # Load the training data features
    with open(input_filename, 'rb') as f:
        fsc_curves = pickle.load(f)

    print(fsc_curves[0].keys())
    ## Do your processing here
    guinier_atomic_values = []
    guinier_hybrid_values = []
    guinier_pseudomodel_values = []

    wilson_atomic_values = []
    wilson_hybrid_values = []
    wilson_pseudomodel_values = []

    for fsc_curve in fsc_curves:
        guinier_atomic_values.append(fsc_curve["fsc_average_guinier_atomic"])
        guinier_hybrid_values.append(fsc_curve["fsc_average_guinier_hybrid"])
        guinier_pseudomodel_values.append(fsc_curve["fsc_average_guinier_pseudomodel"])
        
        wilson_atomic_values.append(fsc_curve["fsc_average_wilson_atomic"])
        wilson_hybrid_values.append(fsc_curve["fsc_average_wilson_hybrid"])
        wilson_pseudomodel_values.append(fsc_curve["fsc_average_wilson_pseudomodel"])
    
    freqs_resample = np.linspace(0, 0.5, 1000)

    fsc_curves_resample = {}
    for fsc_curve in fsc_curves:
        atomic_fsc_curve = fsc_curve["fsc_unsharpened_atomic"]
        hybrid_fsc_curve = fsc_curve["fsc_unsharpened_hybrid"]
        pseudomodel_fsc_curve = fsc_curve["fsc_unsharpened_pseudomodel"]
        
        resampled_atomic_fsc_curve = np.interp(freqs_resample, fsc_curve["freq"], atomic_fsc_curve)
        resampled_hybrid_fsc_curve = np.interp(freqs_resample, fsc_curve["freq"], hybrid_fsc_curve)
        resampled_pseudomodel_fsc_curve = np.interp(freqs_resample, fsc_curve["freq"], pseudomodel_fsc_curve)
        
        fsc_curves_resample[fsc_curve['emdb_pdb']] = fsc_curve.copy()
        fsc_curves_resample[fsc_curve['emdb_pdb']]["freqs_resample"] = freqs_resample
        fsc_curves_resample[fsc_curve['emdb_pdb']]["fsc_unsharpened_atomic_resampled"] = resampled_atomic_fsc_curve
        fsc_curves_resample[fsc_curve['emdb_pdb']]["fsc_unsharpened_hybrid_resampled"] = resampled_hybrid_fsc_curve
        fsc_curves_resample[fsc_curve['emdb_pdb']]["fsc_unsharpened_pseudomodel_resampled"] = resampled_pseudomodel_fsc_curve

    
    mean_atomic_fsc_curve = np.mean([fsc_curves_resample[fsc_curve]["fsc_unsharpened_atomic_resampled"] for fsc_curve in fsc_curves_resample], axis=0)
    mean_hybrid_fsc_curve = np.mean([fsc_curves_resample[fsc_curve]["fsc_unsharpened_hybrid_resampled"] for fsc_curve in fsc_curves_resample], axis=0)
    mean_pseudomodel_fsc_curve = np.mean([fsc_curves_resample[fsc_curve]["fsc_unsharpened_pseudomodel_resampled"] for fsc_curve in fsc_curves_resample], axis=0)

    atomic_fsc_curves_list = [fsc_curves_resample[fsc_curve]["fsc_unsharpened_atomic_resampled"] for fsc_curve in fsc_curves_resample]
    hybrid_fsc_curves_list = [fsc_curves_resample[fsc_curve]["fsc_unsharpened_hybrid_resampled"] for fsc_curve in fsc_curves_resample]
    pseudomodel_fsc_curves_list = [fsc_curves_resample[fsc_curve]["fsc_unsharpened_pseudomodel_resampled"] for fsc_curve in fsc_curves_resample]
    freq_resampled = fsc_curves_resample["0282_6huo"]["freqs_resample"]

    ## Plotting
        
    figsize_mm = (60, 60) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    rcparams['font.size'] = 8

    with temporary_rcparams(rcparams):
        # Plotting code here
        #sns.set_theme(context="paper")
        #sns.set_style("white")
        figsize_cm = (figsize_mm[0]/10, figsize_mm[1]/10)
        xaxis_label = r"Spatial Frequency, d$^{-1}$ ($\AA^{-1}$)"
        fig, ax1 = pretty_plot_confidence_interval(freq_resampled, atomic_fsc_curves_list, hybrid_fsc_curves_list, pseudomodel_fsc_curves_list, \
                                confidence_interval=95, figsize_cm=figsize_cm,linewidth=1, \
                                font="Helvetica",fontscale=2, alpha=0.2, xticks=None, \
                                num_xticks=3, yticks=None, ylims=None, xlims=None, labelsize=None, title=None, \
                                xlabel=xaxis_label, ylabel="FSC", showlegend=False)
        ax2 = ax1.twiny()
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())
        infinity_symbol = r"$\infty$"
        xtick_labels = [f"{1/x:.2f}" for x in ax1.get_xticks()]
        xtick_labels[0] = infinity_symbol
        ax2.set_xticklabels(xtick_labels)
        ax2.set_xlabel(r"Resolution, d ($\AA$)")
        
        fig.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

