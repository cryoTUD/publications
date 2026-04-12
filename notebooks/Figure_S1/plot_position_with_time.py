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
from utils.chapter_4_functions import get_trajectory

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
frame_start_location = 80 # in mm
plot_velocities = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4] # m/s
# plot_velocities = [0.5, 2, 4] # m/s
fps = 2577
plunger_top_level = 160 # mm
plunger_bottom_level = 60 # mm
max_acceleration = 300 # m/s^2
max_deceleration = 300 # m/s^2
view = "front"
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed_data_output", "4_tr_plunger", "extracting_trajectory")
    # figure_input_folder = /add/your/path/here
    input_filename = os.path.join(data_input_folder_main, "extracted_trajectories.pickle")
    plot_output_folder = os.path.join(data_archive_path, "figures_output", "4_tr_plunger", "figure_1", "trajectory_analysis")
    output_filename = os.path.join(plot_output_folder, "trajectories.pdf")  # output plot preferably in pdf format
    # plot_output_folder = /add/your/path/here
    # other output folder
    assert_paths_exist(data_input_folder_main, input_filename)
    create_folders_if_they_do_not_exist(plot_output_folder) # for output folders

    # Load the inputs 
    with open(input_filename, 'rb') as f:
        input_dictionary = pickle.load(f)
    

    plot_frame = input_dictionary[view][0.5]["analysis_data"]["frames_for_plotting"]["last_frame"]
    pixel_size = input_dictionary[view][0.5]["analysis_data"]["pixel_size"]
    plunger_positions = {} 

#    for view in input_dictionary.keys():
    for velocity in input_dictionary[view].keys():
        if velocity not in plot_velocities:
            continue

        analysis_data = input_dictionary[view][velocity]["analysis_data"]
        mean_y_position = analysis_data["mean_y_positions"]
        std_y_position = analysis_data["std_y_positions"]
        mean_x_position = analysis_data["mean_x_positions"]
        std_x_position = analysis_data["std_x_positions"]

        f, g, f_inv, times_ms, pos_mm = get_trajectory(
            plunger_top_level, plunger_bottom_level, velocity, max_acceleration, max_deceleration, return_everything=True)

        # find time when tweezer tip enters frame 
        final_position_mm = 20
        initial_position_mm = mean_y_position[0]

        final_position_plunger_coordinate = pos_mm[-1]
        print(f"final_position_mm: {final_position_mm}")
        print(f"initial_position_mm: {initial_position_mm}")
        print(f"final_position_plunger_coordinate: {final_position_plunger_coordinate}")
        starting_position_plunger_coordinate = final_position_plunger_coordinate - (final_position_mm - initial_position_mm)
        # find the time when tweezer tip enters the frame
        print(f"starting_position_plunger_coordinate: {starting_position_plunger_coordinate}")
        print(f"---")
        starting_time_ms = f_inv(starting_position_plunger_coordinate)
        plunger_positions_global = mean_y_position + starting_position_plunger_coordinate
        print(f"plunger_positions_global_0: {plunger_positions_global[0]}")

        plunger_positions_x = mean_x_position - (plot_frame.shape[1] * pixel_size) / 2
        
        times_ms = np.linspace(0, len(mean_y_position)/fps * 1000, len(mean_y_position))  + starting_time_ms
        plunger_positions[velocity] = {"times_ms": times_ms, \
                                    "mean_y_positions": plunger_positions_global, "std_y_positions": std_y_position, \
                                    "mean_x_positions": plunger_positions_x, "std_x_positions": std_x_position}
    
    figsize_mm = (50, 60)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcParams):
    #    sns.set_context("paper", rc=rcparams)
        # change font size
        rcParams['font.size'] = 6
        cmap = mpl.cm.get_cmap('turbo')
        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        colors = [cmap(norm(i)) for i in plot_velocities]
        fig, ax = plt.subplots(2, 1, figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4))
        for velocity in plot_velocities:
            times_ms = plunger_positions[velocity]["times_ms"]
            mean_y_positions = plunger_positions[velocity]["mean_y_positions"]
            std_y_positions = plunger_positions[velocity]["std_y_positions"]

            ax[0].plot(times_ms, mean_y_positions, label=f"{velocity} m/s", \
                    color=colors[plot_velocities.index(velocity)], linewidth=0.8)
            ax[0].fill_between(times_ms, mean_y_positions - std_y_positions, mean_y_positions + std_y_positions, \
                            alpha=0.3, color=colors[plot_velocities.index(velocity)])
        
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel("Y (mm)")    
        max_y_length =  plot_frame.shape[0] * pixel_size
        frame_end_location = frame_start_location + max_y_length
        print(f"Frame start location: {frame_start_location}, Frame end location: {frame_end_location}")
        ax[0].set_ylim(91, 103)
        yticks = [92, 94, 96, 98, 100, 102]
        ax[0].set_yticks(yticks)
        # invert y axis
        ax[0].invert_yaxis()
        # remove x axis
        ax[0].axes.get_xaxis().set_visible(False)

        # plot x positions 
        for velocity in plot_velocities:
            times_ms = plunger_positions[velocity]["times_ms"]
            mean_x_positions = plunger_positions[velocity]["mean_x_positions"]
            std_x_positions = plunger_positions[velocity]["std_x_positions"]

            ax[1].plot(times_ms, mean_x_positions, label=f"{velocity} m/s", \
                    color=colors[plot_velocities.index(velocity)], linewidth=0.8)
            ax[1].fill_between(times_ms, mean_x_positions - std_x_positions, mean_x_positions + std_x_positions, \
                            alpha=0.3, color=colors[plot_velocities.index(velocity)])
        
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylabel("X (mm)")
        ax[1].set_ylim(-6, 6)
        yticks = [-5, -2.5, 0, 2.5, 5]
        ax[1].set_yticks(yticks)
        xticks = [0, 50, 100, 150, 200, 250]
        ax[1].set_xticks(xticks)
        plt.tight_layout()
        plt.savefig(output_filename)


    print("All plots saved. Please check.")


if __name__ == "__main__":
    main()

