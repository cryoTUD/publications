## IMPORTS
import os
import sys
sys.path.append(os.environ["THESIS_SCRIPTS_ARCHIVE_PATH"])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Custom imports
from utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from utils.plot_utils import temporary_rcparams, configure_plot_scaling
from matplotlib import rcParams


# Set the seed for reproducibility
np.random.seed(42)

# Global variables

## SETUP
def main():
    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    training_loss_path = os.path.join(data_archive_path, "inputs", "3_surfer", "network_training", "training_validation_loss_batchloss_20250208_103426.csv")
    output_folder = os.path.join(data_archive_path, "figures_output", "3_surfer", "figures", "figure_4")
    
    assert_paths_exist(training_loss_path)
    create_folders_if_they_do_not_exist(output_folder)

    output_filename = os.path.join(output_folder, "training_loss_curve.pdf")
    # Load the training loss data
    training_loss_data = pd.read_csv(training_loss_path)

    # Plot the training loss curve
    figsize_mm = (60, 40)
    rc_params = configure_plot_scaling(figsize_mm)
    
    with temporary_rcparams(rc_params):
        # Make the plot editable in Illustrator
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
        sns.set_theme(context="paper")

        ax.plot(training_loss_data["Epoch"], training_loss_data["Average Training Loss"], label="Training Loss")
        ax.plot(training_loss_data["Epoch"], training_loss_data["Validation Loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output_filename)
        plt.close(fig)


if __name__ == "__main__":
    main()
