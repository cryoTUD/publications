## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import cv2 
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from utils.chapter_4_functions import apply_clahe_to_frame

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
frame_start_location = 80 # in mm
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed_data_output", "4_tr_plunger", "extracting_trajectory")
    # figure_input_folder = /add/your/path/here
    input_filename = os.path.join(data_input_folder_main, "extracted_trajectories.pickle")
    plot_output_folder = os.path.join(data_archive_path, "figures_output", "4_tr_plunger", "figure_1", "trajectory_analysis")
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(data_input_folder_main, input_filename)
    create_folders_if_they_do_not_exist(plot_output_folder) # for output folders

    # Load the inputs 
    with open(input_filename, 'rb') as f:
        input_dictionary = pickle.load(f)
    
    figsize_mm = (60, 80)
    rcparams = configure_plot_scaling(figsize_mm)
    cmap = mpl.cm.get_cmap('turbo')
    norm = mpl.colors.Normalize(vmin=0, vmax=4)
    plot_velocities = list(input_dictionary["front"].keys())
    colors = [cmap(norm(i)) for i in plot_velocities]

    for view in input_dictionary.keys():
        for velocity in input_dictionary[view].keys():
            output_filename_velocity = os.path.join(plot_output_folder, f"{view}_{velocity}_trajectories.pdf")

            analysis_data = input_dictionary[view][velocity]["analysis_data"]

            frames = analysis_data["frames_for_plotting"]["frames"]
            frames = np.array(frames, dtype=float)
            combined_frames = np.mean(frames, axis=0)
            last_frame = analysis_data["frames_for_plotting"]["last_frame"]
            plot_frame = last_frame.astype(float) * 0.3 + combined_frames * 0.7

            # enhance the contrast of the image for better visualization
            # plot_frame = apply_clahe_to_frame(plot_frame.astype(np.uint8), clip_limit=5, tile_grid_size=(5, 20))
            # # filter out the striding artifacts using a Singular Value Decomposition
            # # convert to 2D array by taking the mean of the frames
            # plot_frame = plot_frame[:, :, 0]
            # U, S, V = np.linalg.svd(plot_frame, full_matrices=False)
            # S[250:] = 0
            # plot_frame = np.dot(U, np.dot(np.diag(S), V))
            # plot_frame = plot_frame - np.min(plot_frame)
            # plot_frame = plot_frame / np.max(plot_frame)
            # plot_frame = plot_frame * 255

            plot_frame = plot_frame.astype(np.uint8)
            mean_x_position = analysis_data["mean_x_position_pixels"]
            mean_y_position = analysis_data["mean_y_position_pixels"]
            std_x_position = analysis_data["std_x_position_pixels"]
            std_y_position = analysis_data["std_y_position_pixels"]

            pixel_size = analysis_data["pixel_size"]
#            print(pixel_size)
            with temporary_rcparams(rcparams):
                figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
                fig, ax = plt.subplots(1, 1, figsize=figsize)

                ax.imshow(plot_frame)
                ax.plot(mean_x_position, mean_y_position, color=colors[plot_velocities.index(velocity)], \
                        marker=".", markersize=1, linewidth=0.5)
                ax.errorbar(mean_x_position, mean_y_position, xerr=std_x_position, yerr=std_y_position, \
                        fmt='.', color=colors[plot_velocities.index(velocity)], markersize=1, linewidth=0.5)
                max_x_length =  plot_frame.shape[1] * pixel_size
                max_y_length =  plot_frame.shape[0] * pixel_size
 #               print(f"Max X length: {max_x_length}, Max Y length: {max_y_length}")
                frame_end_location = frame_start_location + max_y_length
 #               print(f"Frame end location: {frame_end_location}")
                ax.extent = (-max_x_length//2, max_x_length//2, 0, max_y_length)
                
                # Set X and Y ticks in mm
                xticks = ax.get_xticks()
                #yticks = ax.get_yticks()
                number_of_yticks = 5
                #start_of_yticks = frame_start_location

                #yticks = np.linspace(frame_start_location, frame_end_location, number_of_yticks)
                step = 5
                #yticks_in_mm = [82, 85, 88, 91, 94, 97, 100]
                yticks_in_mm = np.arange(frame_start_location, frame_end_location, step, dtype=int)
                y_real_to_pixels = lambda y: (y - frame_start_location) / pixel_size
                yticks_in_pixels = [y_real_to_pixels(y) for y in yticks_in_mm]
                #yticks_in_pixels = [(y-yticks_in_mm[0]) / pixel_size for y in yticks_in_mm]

                x_real_to_pixels = lambda x: (x + max_x_length/2) / pixel_size
                xticks_in_mm = [-5, 0, 5]
                xticks_in_pixels = [x_real_to_pixels(x)  for x in xticks_in_mm]
                ax.set_xticks(xticks_in_pixels)
                ax.set_xticklabels([f"{x}" for x in xticks_in_mm])
                #ax.set_xticklabels([f"{(x * pixel_size):.1f}" for x in xticks])
                #ax.set_yticklabels([f"{y * pixel_size + frame_start_location:.1f}" for y in yticks])
                #ax.set_yticklabels([f"{y}" for y in yticks])
                ax.set_yticks(yticks_in_pixels)
                ax.set_yticklabels(yticks_in_mm)

                #print(f"Y ticks = {ax.get_yticks()}")
                # xtick_labels = np.arange(0, max_x_length, 5)
                # ytick_labels = np.arange(0, max_y_length, 5)
                # ax.set_xticklabels([f"{x:.1f}" for x in xtick_labels])
                # ax.set_yticklabels([f"{y:.1f}" for y in ytick_labels])
                                        

                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")

                plt.tight_layout()
                plt.savefig(output_filename_velocity)
                plt.close()
                print(f"Plot saved to {output_filename_velocity}. Please check.")
    print("All plots saved. Please check.")


if __name__ == "__main__":
    main()

