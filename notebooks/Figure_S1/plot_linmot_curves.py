## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])

import numpy as np
import pickle
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm

# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from utils.chapter_4_functions import get_trajectory
# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global variables
hue_labels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
has_decimal_point = lambda x: x != int(x)
# get simulated trajectory
plunger_top_level = 160 # mm
plunger_bottom_level = 60 # mm
max_acceleration = 300 # m/s^2
max_deceleration = 300 # m/s

## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "inputs")
    figure_input_folder = os.path.join(data_input_folder_main, "4_tr_plunger", "mechanical_characterisation", "curves_from_linmot")
    # other input folder 

    figure_output_folder_main = os.path.join(data_archive_path, "figures_output")
    plot_output_folder = os.path.join(figure_output_folder_main, "4_tr_plunger", "figures", "figure_1")
    # other output folder
    assert_paths_exist(figure_input_folder)
    create_folders_if_they_do_not_exist(plot_output_folder) # for output folders
    
    output_filename = os.path.join(plot_output_folder, "linmot_curves.pdf")  # output plot preferably in pdf format

    combined_df = pd.DataFrame()    
    files_with_keys = {
        f"{f.split('_')[2].split('.')[0]}_{f.split('_')[0]}" : \
                os.path.join(figure_input_folder, f) \
                for f in os.listdir(figure_input_folder) \
                if f.endswith('.csv') and "acc" not in f
            }

    for key, file in files_with_keys.items():
        velocity = key.split('_')[0]
        trial = key.split('_')[1][-1]

        if "p" in velocity:
            velocity = velocity.replace("p", ".")

        velocity = float(velocity) 

        df = pd.read_csv(file)
        df['velocity'] = velocity
        df['trial'] = trial
        
        combined_df = pd.concat([combined_df, df], axis=0)
    ## Plotting
    
    combined_df["Velocity"] = combined_df["MC SW Overview - Actual Velocity(m/s)"] * -1
    combined_df["Position"] = (combined_df["MC SW Overview - Actual Position(mm)"] - plunger_top_level) * -1

    figsize_mm = (40, 60) # width, height
    rcparams = configure_plot_scaling(figsize_mm)
    with temporary_rcparams(rcparams):
        # Plotting code here
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        cmap = mpl.cm.get_cmap('turbo')
        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        colors = [cmap(norm(i)) for i in hue_labels]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for velocity in hue_labels:
            f = get_trajectory(plunger_top_level, plunger_bottom_level, velocity, max_acceleration, max_deceleration)[0]
            travel_length = plunger_top_level - plunger_bottom_level
            positions = np.linspace(0.1, travel_length - 0.1, 1000) / 1000 # meters, 0.1 mm clearance to avoid out of bounds error
            positions_mm = positions * 1000
            simulated_velocities = f(positions)
            ax.plot(simulated_velocities, positions_mm, color=colors[hue_labels.index(velocity)], linewidth=0.8)
            
            # extract the data for one of the trial 
            experimental_data = combined_df[combined_df["velocity"] == velocity]
            # get the mean and standard deviation of the velocity
            mean_velocity = experimental_data.groupby("Position")["Velocity"].mean()
            std_velocity = experimental_data.groupby("Position")["Velocity"].std()
            ax.errorbar(mean_velocity, mean_velocity.index, xerr=std_velocity, color=colors[hue_labels.index(velocity)], linestyle='--', linewidth=0.8)
            frame_recording_start = positions_mm[-1] - 19 
            frame_recording_end = positions_mm[-1] + 5
            # draw a rectangle to show the recording frame
            #ax.axhline(frame_recording_start, color='black', linestyle='--', linewidth=0.8)
            #ax.axhline(frame_recording_end, color='black', linestyle='--', linewidth=0.8)
            ax.fill_betweenx([frame_recording_start, frame_recording_end], min(hue_labels), max(hue_labels), color='gray', alpha=0.3)    

        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Position (mm)")

        # Set x and y ticks
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        # Invert Y axis 
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

