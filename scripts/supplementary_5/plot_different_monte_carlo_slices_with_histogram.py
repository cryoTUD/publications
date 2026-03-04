## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import json
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm


# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
from locscale.include.emmer.ndimage.map_utils import load_map

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
plot_sample_indices = [4, 8, 11, 15]
slice_index_to_plot = 8
voxel_to_highlight = (9, 26, slice_index_to_plot)
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "general", "supplementary_5")
    emmernet_model_path = os.path.join(data_input_folder_main, "EMmerNet_highContext.hdf5")
    cubes_input_folder = os.path.join(data_input_folder_main, "8069_cube_77")
    # figure_input_folder = /add/your/path/here
    

    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_5", "monte_carlo_slices")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(emmernet_model_path, cubes_input_folder)
    create_folders_if_they_do_not_exist(plot_output_folder)  # create output folders if they do not exist
    
    output_filename_mc_slices = os.path.join(plot_output_folder, "monte_carlo_slices.pdf")
    output_filename_histogram = os.path.join(plot_output_folder, "mc_histogram.pdf")
    histogram_data_json = os.path.join(plot_output_folder, "histogram_data.json")

    cube_file_paths = {i : os.path.join(cubes_input_folder, f"monte_carlo_sample_{i-1}.mrc") for i in plot_sample_indices}
    # load cubes 
    cubes = {i : load_map(cube_file_paths[i])[0] for i in plot_sample_indices} 
    slices_to_plot = {i : cubes[i][:, :, slice_index_to_plot] for i in plot_sample_indices}  # Extract the slices at the specified index
    
    figsize_mm = (60, 60)
    fontsize = 6
    rcparams = configure_plot_scaling(figsize_mm)
    rcparams['font.size'] = fontsize
    with temporary_rcparams(rcparams):
        fig, ax = plt.subplots(2, 2)
        for i, plot_index in enumerate(plot_sample_indices):
            row_index = i // 2
            col_index = i % 2
            
            ax[col_index, row_index].imshow(slices_to_plot[plot_index], cmap='gray')
            ax[col_index, row_index].scatter(voxel_to_highlight[0], voxel_to_highlight[1], color='red', marker="x")
            ax[col_index, row_index].set_title(f"Sample {plot_index}")
            # hide x and y axis
            ax[col_index, row_index].axis('off')

    
        #fig.suptitle(f"Monte Carlo Slices at z = {slice_index_to_plot}")
        fig.tight_layout()
    # Save figure 
    fig.savefig(output_filename_mc_slices, bbox_inches='tight')

    # extract the intensity values at the highlighted voxel for all monte carlo samples
    all_cubes_paths = [os.path.join(cubes_input_folder, f"monte_carlo_sample_{i-1}.mrc") for i in range(1, 51)]
    intensities = [load_map(path)[0][voxel_to_highlight[2], voxel_to_highlight[1], voxel_to_highlight[0]] for path in all_cubes_paths]
    intensities = np.array(intensities)

    figsize_mm = (30, 30)
    fontsize = 8
    rcparams_fig2 = configure_plot_scaling(figsize_mm, fontsize=fontsize)
    with temporary_rcparams(rcparams_fig2):
        fig2, ax2 = plt.subplots(1, 1)
        sns.histplot(intensities, bins=6, kde=True, ax=ax2)
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Count")
    
        fig2.tight_layout()
        fig2.savefig(output_filename_histogram, bbox_inches='tight')
    
    # save intensities as json
    histogram_data = {
        "intensities": intensities.tolist(),
        "voxel_to_highlight": voxel_to_highlight,
        "slice_index_to_plot": slice_index_to_plot,
        "cube_file_paths": cube_file_paths,
    }

    with open(histogram_data_json, 'w') as f:
        json.dump(histogram_data, f)
    
    print(f"Plot saved to {output_filename_mc_slices}. Please check.")
    print(f"Plot saved to {output_filename_histogram}. Please check.")
    print(f"Data saved to {histogram_data_json}. Please check.")


if __name__ == "__main__":
    main()

