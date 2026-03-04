## IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
from scripts.utils.plot_utils import *
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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Import helper functions 
from scripts.figure_2.figure_2_functions import plot_fsc_average, plot_fsc_curves_one_cycle, combine_fsc_plots
from scripts.utils.plot_utils import pretty_lineplot_XY, temporary_rcparams, configure_plot_scaling, pretty_plot_rainbow_series
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set global variables
SAVE_FIGURE = True
figure_number = 2
## SETUP 
def main():
    data_archive_path = setup_environment()

    # DEFINE THE PATHS 
    input_structured_data_folder = os.path.join(data_archive_path, "processed","structured_data", f"figure_{figure_number}")
    output_figures_folder = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}", "fsc_bfactor_refinement_animation")
    input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")

    # Input file paths 
    halfmap_1_fsc_average_pickle = os.path.join(input_structured_data_folder, f"fsc_average_halfmap_1_cycle50_masked.pickle")
    halfmap_2_fsc_average_pickle = os.path.join(input_structured_data_folder, f"fsc_average_halfmap_2_cycle50_masked.pickle")
    halfmap_1_path = os.path.join(data_archive_path, "processed", "pdbs", f"figure_{figure_number}", "overfitting_analysis","emd_8702_half_map_1.map")
    assert_paths_exist(
        halfmap_1_fsc_average_pickle, halfmap_2_fsc_average_pickle)
    
    # Create the output folders
    create_folders_if_they_do_not_exist(
        output_figures_folder
    )
    
    # Figure output paths
    fsc_curves_cycle_1_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_1.pdf")
    fsc_curves_cycle_1_figure_path_png = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_1.png")
    fsc_curves_cycle_5_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_5.pdf")
    fsc_curves_cycle_5_figure_path_png = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_5.png")
    fsc_curves_cycle_10_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_10.pdf")
    fsc_curves_cycle_10_figure_path_png = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_cycle_10.png")
    fsc_curves_1_5_10_combined_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_1_5_10_combined.pdf")
    fsc_curves_1_5_10_combined_figure_path_png = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_1_5_10_combined.png")
    fsc_curves_all_cycles_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_all_cycles.pdf")
    fsc_curves_all_cycles_figure_path_png = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_all_cycles.png")
    


    # Load the data
    with open(halfmap_1_fsc_average_pickle, "rb") as f:
        fsc_cycles_halfmap1_with_averaging = pickle.load(f)
    with open(halfmap_2_fsc_average_pickle, "rb") as f:
        fsc_cycles_halfmap2_with_averaging = pickle.load(f)
    
    # Plot the FSC curves for each cycle
    fsc_cycle_1_halfmap1 = fsc_cycles_halfmap1_with_averaging[1][2]
    fsc_cycle_1_halfmap2 = fsc_cycles_halfmap2_with_averaging[1][2]

    
    fsc_cycle_5_halfmap1 = fsc_cycles_halfmap1_with_averaging[5][2]
    fsc_cycle_5_halfmap2 = fsc_cycles_halfmap2_with_averaging[5][2]

    fsc_cycle_10_halfmap1 = fsc_cycles_halfmap1_with_averaging[10][2]
    fsc_cycle_10_halfmap2 = fsc_cycles_halfmap2_with_averaging[10][2]



    _, apix = load_map(halfmap_1_path)

    list_of_cycle_1_fsc = [fsc_cycle_1_halfmap1, fsc_cycle_1_halfmap2]
    list_of_cycle_5_fsc = [fsc_cycle_5_halfmap1, fsc_cycle_5_halfmap2]
    list_of_cycle_10_fsc = [fsc_cycle_10_halfmap1, fsc_cycle_10_halfmap2]
    freq = frequency_array(fsc_cycle_1_halfmap2, apix)
    print(f"Frequency array size: {len(freq)}")

    if SAVE_FIGURE:
        # combine_fsc_plots(
        #     freq, 
        #     list_of_cycle_1_fsc, list_of_cycle_5_fsc, list_of_cycle_10_fsc, 
        #     output_figures_folder, 
        #     legends=["Halfmap 1", "Halfmap 2"], 
        #     ylims=(-0.05, 1), 
        #     figsize_mm=(120, 40), 
        #     dpi=600, 
        #     font="Helvetica", 
        #     fontsize=10, 
        #     ticksize=8, 
        #     linewidth=2, 
        #     linestyle="--", 
        #     hline_color="k", 
        #     vline_colors=("r", "b")
        # )

        figsize_mm = (60, 60)
        rcParams_new = configure_plot_scaling(figsize_mm)
        rcParams_new['font.size'] = 8
        with temporary_rcparams(rcParams_new):
            list_of_y_array = [fsc_cycles_halfmap2_with_averaging[i][2] for i in fsc_cycles_halfmap2_with_averaging.keys()]
            num_cycles = len(list_of_y_array)
            print(f"Number of cycles: {num_cycles}")
            for cycle in range(num_cycles):
                fsc_curves_average_figure_path = os.path.join(output_figures_folder, f"figure_{figure_number}_fsc_curves_average_cycle_{cycle}.png")
                print(f"Cycle {cycle} of {num_cycles}")
                current_list_of_y_array = list_of_y_array[:cycle]

                fig_all_cycles = pretty_plot_rainbow_series(\
                    freq, current_list_of_y_array, \
                    figsize_mm=figsize_mm,\
                    xticks=[0, 0.2, 0.4],
                    yticks=[0, 0.5, 1],
                    xlabel=r'Spatial frequency ($\AA^{-1}$)',
                    ylabel='FSC',
                    )
                # Change figure title
                fig_all_cycles.suptitle(f"Cycle {cycle+1}")
                fig_all_cycles.tight_layout()
                fig_all_cycles.savefig(fsc_curves_average_figure_path, dpi=600, bbox_inches='tight')



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


    




    

