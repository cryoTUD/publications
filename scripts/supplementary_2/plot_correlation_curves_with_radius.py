## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import pandas as pd
import random 

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling, pretty_lineplot_XY, pretty_lineplot_XY_multiple

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)

def main():
    # Setup environment and define paths
    data_archive_path = setup_environment()
    correlation_json_path = os.path.join(data_archive_path, "outputs", "supplementary_2", "neighborhood_correlation", "neighborhood_radius_vs_correlation_values.json")
    plot_output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2", "neighborhood_correlation", "correlation_curves")

    assert_paths_exist(correlation_json_path)
    create_folders_if_they_do_not_exist(plot_output_folder)

    with open(correlation_json_path, 'r') as f:
        corr_data = json.load(f)

    # Extract values from JSON
    radii = corr_data["radii"]
    correlations_atomic = corr_data["atomic"]
    correlations_restrained = corr_data["restrained_pseudomodel"]
    correlations_unrestrained = corr_data["unrestrained_pseudomodel"]
    correlations_unrestrained_ordered = corr_data["unrestrained_pseudomodel_ordered"]
    correlations_unrestrained_disordered = corr_data["unrestrained_pseudomodel_disordered"]
    correlations_restrained_ordered = corr_data["restrained_pseudomodel_ordered"]
    correlations_restrained_disordered = corr_data["restrained_pseudomodel_disordered"]

    xlabel = r"Neighborhood radius ($\AA$)"
    ylabel = "ADP Correlation"
    figsize_mm = (80, 80)
    figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
    fontsize = 8
    yticks = [0.5, 1]
    ylim = [0.2, 1.2]
    rcparams = configure_plot_scaling(figsize_mm)

    with temporary_rcparams(rcparams):
        pretty_lineplot_XY_multiple(
            [radii, radii],
            [correlations_atomic, correlations_unrestrained],
            xlabel, ylabel,
            figsize_cm=figsize_cm, fontsize=fontsize,
            legends=["Atomic model", "Unrestrained pseudomodel"],
            save_path=os.path.join(plot_output_folder, "correlation_atomic_unrestrained.pdf"),
            ylim=ylim, yticks=yticks,
        )

        pretty_lineplot_XY_multiple(
            [radii]*2,
            [correlations_atomic, correlations_restrained],
            xlabel, ylabel,
            figsize_cm=figsize_cm, fontsize=fontsize,
            legends=["Atomic model", "Restrained pseudomodel"],
            save_path=os.path.join(plot_output_folder, "correlation_atomic_restrained.pdf"),
            ylim=ylim, yticks=yticks,
        )

        pretty_lineplot_XY_multiple(
            [radii]*2,
            [correlations_unrestrained_ordered, correlations_unrestrained_disordered],
            xlabel, ylabel,
            figsize_cm=figsize_cm, fontsize=fontsize,
            legends=["Ordered", "Disordered"],
            save_path=os.path.join(plot_output_folder, "correlation_unrestrained_ordered_disordered.pdf"),
            ylim=ylim, yticks=yticks,
        )

        pretty_lineplot_XY_multiple(
            [radii]*2,
            [correlations_restrained_ordered, correlations_restrained_disordered],
            xlabel, ylabel,
            figsize_cm=figsize_cm, fontsize=fontsize,
            legends=["Ordered", "Disordered"],
            save_path=os.path.join(plot_output_folder, "correlation_restrained_ordered_disordered.pdf"),
            ylim=ylim, yticks=yticks,
        )

    print("All correlation line plots saved successfully.")

if __name__ == "__main__":
    main()