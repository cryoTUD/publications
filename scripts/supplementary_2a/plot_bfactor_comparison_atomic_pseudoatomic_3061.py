## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import invgamma

from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams, plot_correlations

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
def main():
    data_archive_path = setup_environment()

    input_path = os.path.join(data_archive_path, "structured_data", "supplementary_2a", "emd_3061_bfactor_distribution_data.json")
    output_folder = os.path.join(data_archive_path, "outputs", "supplementary_2a")
    output_path = os.path.join(output_folder, "emd_3061_bfactor_distribution_pseudo_and_atomic.pdf")

    assert_paths_exist(input_path)
    create_folders_if_they_do_not_exist(output_folder)

    with open(input_path, 'r') as f:
        data = json.load(f)

    pseudo_bfactors = np.array(data["pseudo_bfactors"])
    atomic_bfactors = np.array(data["atomic_bfactors"])


    x = np.linspace(0, max(pseudo_bfactors.max(), atomic_bfactors.max()), 500)

    figsize_mm = (70, 60)
    rcparams = configure_plot_scaling(figsize_mm)

    with temporary_rcparams(rcparams):
        figsize_cm = (figsize_mm[0] / 10, figsize_mm[1] / 10)
        
        plot_correlations(pseudo_bfactors, atomic_bfactors, \
            "Pseudo-atomic B-factor", "Atomic B-factor", "B-factor correlation", scatter=True,\
            figsize_cm = figsize_cm, fontscale=0.5, \
            filepath=output_path\
        )


    print(f"Saved distribution comparison plot to {output_path}")

if __name__ == "__main__":
    main()