## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import pandas as pd
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.chapter_3_find_micelle import find_number_of_membranes, \
    extract_dummy_residues_from_pdb, get_membrane, find_best_plane, find_membrane_end
import math 
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams
from scipy.ndimage import sobel, uniform_filter
from tqdm import tqdm

# Set the seed for reproducibility
np.random.seed(42)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    slices_info_path = os.path.join(data_archive_path, "processed_data_output", "3_surfer", "training_data_analysis", "slices_extracted_0257.pickle")
    plot_output_folder = os.path.join(data_archive_path,  "figures_output", "3_surfer", "figures", "figure_2")
    assert_paths_exist(slices_info_path)
    create_folders_if_they_do_not_exist(plot_output_folder)
    
    output_filename = os.path.join(plot_output_folder, "slices.pdf")

    # Load the training data features
    slices_extracted = pd.read_pickle(slices_info_path)
    
    slices_raw = slices_extracted["slices_raw"]
    print(f"There are {len(slices_raw)} slices in the dataset.")
    slices_smooth = slices_extracted["slices_smooth"]
    slices_edge = slices_extracted["slices_edge"]
    pixels_raw = slices_extracted["pixels_raw"]
    pixels_smooth = slices_extracted["pixels_smooth"]
    pixels_edge = slices_extracted["pixels_edge"]
    
    # Plot the slices
    num_slices = 5 
    slice_indices = np.linspace(0, len(slices_raw) - 1, num_slices).astype(int) 
    slices_raw = np.array(slices_raw)
    slices = slices_raw[slice_indices]
    z_coords = slice_indices

    figsize_mm = (30, 30)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        import seaborn as sns
        sns.set_theme(context="paper")
        sns.set_style("white")
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        azimuthal_angle = 45
        elevation_angle = 30
        # save each slice as a separate figure
        for i, slice in enumerate(slices):
            #fig, ax = plt.subplots(figsize=figsize, dpi=600)
            fig = plt.figure(figsize=figsize, dpi=600)
            ax = fig.add_subplot(111, projection='3d')

            #ax.imshow(slice, cmap="gray")
            output_filename = os.path.join(plot_output_folder, f"slice_{z_coords[i]}_original.pdf")
            # hide axes
            ax.axis("off")
            x_fraction = 0
            y_fraction = 0
            x_array = np.arange(slice.shape[0])
            y_array = np.arange(slice.shape[1])
            x, y = np.meshgrid(x_array, y_array)
            ax.contourf(x, y, slice, zdir='z', offset=0, cmap="gray")
            ax.set_xlim(int(x_fraction*slice.shape[0]), int((1-x_fraction)*slice.shape[0]))
            ax.set_ylim(int(y_fraction*slice.shape[1]), int((1-y_fraction)*slice.shape[1]))
            ax.view_init(elevation_angle, azimuthal_angle)
            plt.tight_layout()
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close(fig)
        # save the figure
        #plt.savefig(output_filename)    
        
        # Plot the edge slices
        slices = np.array(slices_edge)[slice_indices]
        for i, slice in enumerate(slices):
            fig = plt.figure(figsize=figsize, dpi=600)
            ax = fig.add_subplot(111, projection='3d')

            
            output_filename = os.path.join(plot_output_folder, f"slice_{z_coords[i]}_edge.pdf")

            x_array = np.arange(slice.shape[0])
            y_array = np.arange(slice.shape[1])
            x, y = np.meshgrid(x_array, y_array)
            ax.contourf(x, y, slice, zdir='z', offset=0, cmap="gray")
            ax.set_xlim(int(x_fraction*slice.shape[0]), int((1-x_fraction)*slice.shape[0]))
            ax.set_ylim(int(y_fraction*slice.shape[1]), int((1-y_fraction)*slice.shape[1]))
            ax.view_init(elevation_angle, azimuthal_angle)
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close(fig)

    # Plot pixels_raw and pixels_edge in two separate plots
    figsize_mm = (38, 8)
    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        sns.set_theme(context="paper")
        sns.set_style("white")
        fig = plt.figure(figsize=figsize, dpi=600)
        ax = fig.add_subplot(111)
        ax.plot(pixels_raw, "k-")
        # hide axes
        ax.axis("off")
        output_filename = os.path.join(plot_output_folder, "pixel_intensity_raw.pdf")
        plt.savefig(output_filename)
        plt.close(fig)

        fig_edge = plt.figure(figsize=figsize, dpi=600)
        ax_edge = fig_edge.add_subplot(111)
        ax_edge.plot(pixels_edge, "k-")
        # hide axes
        ax_edge.axis("off")
        output_filename = os.path.join(plot_output_folder, "pixel_intensity_edge.pdf")
        plt.savefig(output_filename)
        plt.close(fig_edge)
    
    
    print(f"Plot saved in {plot_output_folder}. Please check.")

if __name__ == "__main__":
    main()

