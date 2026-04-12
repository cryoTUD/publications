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
#plot_velocities = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4] # m/s
plot_velocities = [0.5, 2, 4] # m/s
fps = 2577
plunger_top_level = 160 # mm
plunger_bottom_level = 60 # mm
max_acceleration = 300 # m/s^2
max_deceleration = 300 # m/s^2
final_pixel_position = 879 # pixel. from calibration photo 

view = "front"
pixel_size = 0.02293 if view == "front" else 0.01803 # mm/pixel
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "processed_data_output", "4_tr_plunger", "extracting_trajectory")
    # figure_input_folder = /add/your/path/here
    input_filename = os.path.join(data_input_folder_main, "extracted_trajectories.pickle")
    plot_output_folder = os.path.join(data_archive_path, "figures_output", "4_tr_plunger", "figure_1", "trajectory_analysis")
    output_filename = os.path.join(plot_output_folder, "overshoot_with_velocity.pdf")  # output plot preferably in pdf format
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
    overshoots = {}
#    for view in input_dictionary.keys():
    for velocity in input_dictionary[view].keys():
        if velocity not in plot_velocities:
            continue

        analysis_data = input_dictionary[view][velocity]["analysis_data"]
        all_y_positions = analysis_data["all_y_positions"]
        for trial in range(all_y_positions.shape[0]):
            y_positions = all_y_positions[trial]
            final_position = y_positions[-1]

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

        times_ms = np.linspace(0, len(mean_y_position)/fps * 1000, len(mean_y_position))  + starting_time_ms
        plunger_positions[velocity] = {"times_ms": times_ms, "mean_y_positions": plunger_positions_global, "std_y_positions": std_y_position}
    
    # Analyse overshoot for each velocity 
    
    
    figsize_mm = (50, 30)
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcParams):
    #    sns.set_context("paper", rc=rcparams)
        # change font size
        rcParams['font.size'] = 6
        cmap = mpl.cm.get_cmap('turbo')
        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        colors = [cmap(norm(i)) for i in plot_velocities]
        fig, ax = plt.subplots(1, 1, figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4))
        
        
        plt.savefig(output_filename)


    print("All plots saved. Please check.")


if __name__ == "__main__":
    main()

