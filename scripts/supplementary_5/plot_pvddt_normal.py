## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from scipy.stats import norm
import pandas as pd
import random 

# from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from tqdm import tqdm

# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
from scripts.utils.plot_utils import temporary_rcparams, configure_plot_scaling

# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)
pvddt_widths = {
    55 : [(0.20720723954852527, 0.00277008775309308),
((0.23237866, 0.024490297))],
    100 : [(0.07392615267142516, 0.0012911881196970626),
((0.12933666, 0.01982694))],
    -100 : [(0.2770322171611471, 0.004350773569283758),
((0.18277153, 0.020130664))],
    -90 : [(0.19472677623828819, 0.000977296505410216),
((0.16061346, 0.021057108))],
    8 : [(0.17375670265528192, 0.0020913385442902473),
((0.17398758, 0.036076315))],
    69 : [(0.12293087568734924, 0.0023283601814345),
((0.13584825, 0.0176837))],
    99 : [(0.13219700039468538, 0.0018968648545799527),
((0.20933445, 0.030922985))],
    75 : [(0.08377463640093188, 0.0019775468764152457),
((0.105589345, 0.018290846))],
    -85 : [(0.37949868126074837, 0.003908494735480005),
((0.30855212, 0.042653985))],
    -94 : [(0.11704425266610634, 0.0020676607198638936),
((0.08470046, 0.0176837))],
    -41 : [(0.15069132082562675, 0.0021073876646927744),
((0.13455817, 0.028281512))],
    87 : [(0.1625725818908578, 0.0028152013525251426),
((0.20562586, 0.030366581))],
    47 : [(0.057308067927738335, 0.0015709598842073879),
((0.07164933, 0.0176837))],
    -99 : [(0.19160946747438345, 0.002695978007434993),
((0.12820758, 0.02388548))],

}

mean_value = 0.2
std_value = 0.02

chosen_pvddt = [99, 87, 8, -85, -99]
## SETUP
def main():    
    # Setup environment and define paths
    data_archive_path = setup_environment()
    data_input_folder_main = os.path.join(data_archive_path, "inputs")
    # figure_input_folder = /add/your/path/here
    # other input folder 

    figure_output_folder_main = os.path.join(data_archive_path, "outputs", "supplementary_5")
    # plot_output_folder = /add/your/path/here
    # other output folder
    
    output_filename = os.path.join(figure_output_folder_main, "pvddt_normal_slice.pdf")  # output plot preferably in pdf format

    x_values_range = np.linspace(mean_value - 6 * std_value, mean_value + 6 * std_value, 1000)
    y_values = norm.pdf(x_values_range, mean_value, std_value)

    # create normal distribution for chosen pvddt values
    normal_distributions = {}
    transformed_distributions = {}

    for pvddt in chosen_pvddt:
        mean_original, std_original = pvddt_widths[pvddt][0]
        mean_original_old_reference, std_original_old_reference = pvddt_widths[pvddt][1]
        # transform to new reference
        z_score_original = (mean_original - mean_original_old_reference) / std_original_old_reference
        mean_transformed = mean_value + z_score_original * std_value
        std_transformed = std_original / std_original_old_reference * std_value

        print(f"{pvddt} : old: {mean_original},{std_original} | new: {mean_transformed},{std_transformed}")
        normal_distributions[pvddt] = norm.pdf(x_values_range, mean_transformed, std_transformed) 
        normal_distributions[pvddt] = normal_distributions[pvddt] / normal_distributions[pvddt].max()  # normalize to max 1
        transformed_distributions[pvddt] = {"voxel_mean" : mean_transformed, "voxel_std" : std_transformed, \
                                            "reference_mean" : mean_original_old_reference, "reference_std" : std_original_old_reference}
       
    figsize_mm = (50, 30) # width, height
    fontsize = 50
    rcparams = configure_plot_scaling(figsize_mm, fontsize)
    color_codes = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]
    with temporary_rcparams(rcparams):
        # Plotting code here
        fig, ax = plt.subplots(figsize=figsize_mm, dpi=600)
        ax.plot(x_values_range, y_values/y_values.max(), color='grey', linewidth=2)
        reference_normal = norm.rvs(loc=mean_value, scale=std_value, size=500)
        normal_pvddt_1 = norm.rvs(loc=transformed_distributions[chosen_pvddt[0]]["voxel_mean"],
                                    scale=transformed_distributions[chosen_pvddt[0]]["voxel_std"], size=5000)
        normal_pvddt_2 = norm.rvs(loc=transformed_distributions[chosen_pvddt[1]]["voxel_mean"],
                                    scale=transformed_distributions[chosen_pvddt[1]]["voxel_std"], size=1000)
        normal_pvddt_3 = norm.rvs(loc=transformed_distributions[chosen_pvddt[2]]["voxel_mean"],
                                    scale=transformed_distributions[chosen_pvddt[2]]["voxel_std"], size=1000)
        normal_pvddt_4 = norm.rvs(loc=transformed_distributions[chosen_pvddt[3]]["voxel_mean"],
                                    scale=transformed_distributions[chosen_pvddt[3]]["voxel_std"], size=1000)
        normal_pvddt_5 = norm.rvs(loc=transformed_distributions[chosen_pvddt[4]]["voxel_mean"],
                                    scale=transformed_distributions[chosen_pvddt[4]]["voxel_std"], size=1000)
        
        # sns.violinplot(reference_normal, ax=ax, color='grey', linewidth=2, alpha=0.5, label='Reference Normal')
        # sns.stripplot(normal_pvddt_1, ax=ax, color=color_codes[0], linewidth=2, alpha=0.5, label=f'PVDDT {chosen_pvddt[0]}')
        # sns.stripplot(normal_pvddt_2, ax=ax, color=color_codes[1], linewidth=2, alpha=0.5, label=f'PVDDT {chosen_pvddt[1]}')
        # sns.stripplot(normal_pvddt_3, ax=ax, color=color_codes[2], linewidth=2, alpha=0.5, label=f'PVDDT {chosen_pvddt[2]}')
        # sns.stripplot(normal_pvddt_4, ax=ax, color=color_codes[3], linewidth=2, alpha=0.5, label=f'PVDDT {chosen_pvddt[3]}')
        # sns.stripplot(normal_pvddt_5, ax=ax, color=color_codes[4], linewidth=2, alpha=0.5, label=f'PVDDT {chosen_pvddt[4]}')
        
        for i, pvddt in enumerate(chosen_pvddt):
            ax.plot(x_values_range, normal_distributions[pvddt], label=f'PVDDT {pvddt}', linewidth=2, color=color_codes[i])
            # shade area under the curve with solid color
            ax.fill_between(x_values_range, 0, normal_distributions[pvddt], color=color_codes[i], alpha=0.3)

        ax.set_xlabel('Intensity (arb. units)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        fig.tight_layout()
        

        fig.savefig(output_filename) 
        plt.close(fig)

    print(f"Plot saved to {output_filename}. Please check.")

if __name__ == "__main__":
    main()

