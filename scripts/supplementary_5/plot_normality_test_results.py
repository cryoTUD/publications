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
num_subplots_new = 4
num_samples = 50
def plot_stack_images_3d(arrays, figsize, cmap='viridis'):
    """
    Plots a stack of 2D arrays in 3D space (one above the other in z-direction).

    Parameters:
    arrays (list): List of 2D NumPy arrays.
    cmap (str): Matplotlib colormap name.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(32)
    y = np.arange(32)
    x, y = np.meshgrid(x, y)

    for i, arr in enumerate(arrays):
        z = np.full_like(arr, i)  # stack at different z-levels
        ax.plot_surface(z, x, y, facecolors=plt.cm.get_cmap(cmap)(arr), rstride=1, cstride=1, antialiased=False, shade=False)
        # hide the grid lines
        ax.grid(False)
        # hide the axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


    ax.set_zlim(-1, len(arrays))
    ax.set_xlabel('Slice Index')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=240)
    plt.tight_layout()
    return fig


## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "raw", "general", "supplementary_5")
    emmernet_model_path = os.path.join(data_input_folder_main, "EMmerNet_highContext.hdf5")
    cubes_input_folder = os.path.join(data_input_folder_main, "8069_cube_77")
    # figure_input_folder = /add/your/path/here
    output_folder_main = os.path.join(data_archive_path, "processed", "general", "supplementary_5", "normality_test")
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_5", "normality_test")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(output_folder_main)  # check if the paths exist
    create_folders_if_they_do_not_exist(plot_output_folder)  # create output folders if they do not exist
    
    output_filename_slices = os.path.join(plot_output_folder, f"p-values_slices_{num_samples}.pdf")
    output_filename_p_distribution = os.path.join(plot_output_folder, f"p-values_distribution_num_samples_{num_samples}.pdf")

    mean_map_path = os.path.join(output_folder_main, f"mean_map_num_samples_{num_samples}.mrc")
    p_values_map_path = os.path.join(output_folder_main, f"p_values_map_num_samples_{num_samples}.mrc")
    significant_deviations_map_path = os.path.join(output_folder_main, f"significant_deviations_map_num_samples_{num_samples}.mrc")

    mean_map, apix = load_map(mean_map_path)
    mask_map = (mean_map > 0.05).astype(int)

    p_values_map, apix = load_map(p_values_map_path)
    significant_deviations_map, apix = load_map(significant_deviations_map_path)

    # print percentage of voxels which are significant
    num_significant_voxels = np.sum(significant_deviations_map > 0)
    total_voxels = np.prod(significant_deviations_map.shape)
    percentage_significant_voxels = (num_significant_voxels / total_voxels) * 100
    print(f"Percentage of significant voxels: {percentage_significant_voxels:.2f}%, {num_significant_voxels} out of {total_voxels} voxels")
    indices = np.linspace(0, 31, num_subplots_new, dtype=int)
    #indices=[8, 10, 12, 14]
    mean_map_slices = [mean_map[:, :, i] for i in indices]
    p_values_map_slices = [p_values_map[:, :, i] for i in indices]
    significant_deviations_map_slices = [significant_deviations_map[:, :, i] for i in indices]

    mean_map_slices = np.array(mean_map_slices)
    p_values_map_slices = np.array(p_values_map_slices)
    significant_deviations_map_slices = np.array(significant_deviations_map_slices)
    figsize_mm = (60, 80)
    fontsize = 6
    rcparams = configure_plot_scaling(figsize_mm)
    rcparams['font.size'] = fontsize
    with temporary_rcparams(rcparams):
        fig, ax = plt.subplots(3, num_subplots_new)
        for i, index in enumerate(indices):
            # plot the X cube slice in first two, p value in second and significance in third
            ax[0,i].imshow(mean_map[:,:, index], cmap='gray')
            # highlight the masked region
            #ax[0,i].imshow(mask_map[:,:, index], cmap='gray', alpha=0.5)
            # hide x and y tick
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            
            ax[1,i].imshow(p_values_map[:, :, index], cmap='rainbow')
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])

            
            ax[2,i].imshow(significant_deviations_map[:, :, index], cmap='gray')
            ax[2,i].set_xticks([])
            ax[2,i].set_yticks([])
    # Save figure 
    fig.savefig(output_filename_slices, bbox_inches='tight')

    # extract p values to plot distribution
    p_values_array = p_values_map.flatten()

    figsize_mm = (60, 40)
    fontsize = 8
    rcparams_fig2 = configure_plot_scaling(figsize_mm, fontsize=fontsize)
    with temporary_rcparams(rcparams_fig2):
        fig2, ax2 = plt.subplots(1, 1)
        sns.histplot(p_values_array, bins=31, kde=True, ax=ax2)
        ax2.set_xlabel("P-value")
        ax2.set_ylabel("Count")
        ax2.set_title("P-value Distribution (Shapiro-Wilk Test)")
        
    # Save figure
    fig2.tight_layout()
    fig2.savefig(output_filename_p_distribution, bbox_inches='tight')

    print(f"Saved files to {plot_output_folder}")
    print(f"Saved slices to {output_filename_slices}")
    print(f"Saved p-value distribution to {output_filename_p_distribution}")


if __name__ == "__main__":
    main()

