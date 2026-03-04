import os 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from locscale.include.emmer.ndimage.map_utils import load_map
from locscale.include.emmer.ndimage.profile_tools import frequency_array

data_archive_path = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data"

# DEFINE THE PATHS 
input_structured_data_folder = os.path.join(data_archive_path, "processed","structured_data", f"figure_{2}")
output_figures_folder = os.path.join(data_archive_path, "outputs", f"figure_{2}")
input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")

# Input file paths 
halfmap_1_fsc_average_pickle = os.path.join(input_structured_data_folder, f"fsc_average_halfmap_1_cycle50_masked.pickle")
halfmap_2_fsc_average_pickle = os.path.join(input_structured_data_folder, f"fsc_average_halfmap_2_cycle50_masked.pickle")
halfmap_1_path = os.path.join(data_archive_path, "processed", "pdbs", f"figure_{2}", "overfitting_analysis","emd_8702_half_map_1.map")

# Load the data
with open(halfmap_1_fsc_average_pickle, "rb") as f:
    fsc_cycles_halfmap1_with_averaging = pickle.load(f)
with open(halfmap_2_fsc_average_pickle, "rb") as f:
    fsc_cycles_halfmap2_with_averaging = pickle.load(f)

# Plot the FSC curves for each cycle
fsc_cycle_1_halfmap1 = fsc_cycles_halfmap1_with_averaging[1][2]
fsc_cycle_1_halfmap2 = fsc_cycles_halfmap2_with_averaging[1][2]


fsc_cycle_5_halfmap1 = fsc_cycles_halfmap1_with_averaging[5][2]
fsc_cycle_5_halfmap2 = fsc_cycles_halfmap2_with_averaging[5][2]

fsc_cycle_10_halfmap1 = fsc_cycles_halfmap1_with_averaging[10][2]
fsc_cycle_10_halfmap2 = fsc_cycles_halfmap2_with_averaging[10][2]



_, apix = load_map(halfmap_1_path)

list_of_cycle_1_fsc = [fsc_cycle_1_halfmap1, fsc_cycle_1_halfmap2]
list_of_cycle_5_fsc = [fsc_cycle_5_halfmap1, fsc_cycle_5_halfmap2]
list_of_cycle_10_fsc = [fsc_cycle_10_halfmap1, fsc_cycle_10_halfmap2]
freq = frequency_array(fsc_cycle_1_halfmap2, apix)

from contextlib import contextmanager
from matplotlib import rcParams

@contextmanager
def temporary_rcparams(updates):
    original_rcparams = rcParams.copy()
    try:
        rcParams.update(updates)
        yield
    finally:
        rcParams.update(original_rcparams)

def configure_plot_scaling(figsize_mm):
    from matplotlib import rcParams
    figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
    scale_factor = figsize_in[0] / 8  # Assume 8 inches as base width
    # add the scale factor to the rcParams

    updates_dict = {
        'font.size': 10 * scale_factor,
        'lines.linewidth': 2 * scale_factor,
        'lines.markersize': 12 * scale_factor,
        'xtick.labelsize': 32 * scale_factor,
        'ytick.labelsize': 32 * scale_factor,
        'axes.labelsize': 32 * scale_factor,
        'axes.titlesize': 32 * scale_factor,
        'legend.fontsize': 32 * scale_factor,
        'xtick.major.size': 12 * scale_factor,
        'ytick.major.size': 12 * scale_factor,
        'xtick.minor.size': 4 * scale_factor,
        'ytick.minor.size': 4 * scale_factor,
        'xtick.major.width': 3 * scale_factor,
        'ytick.major.width': 3 * scale_factor,
        'xtick.minor.width': 1 * scale_factor,
        'ytick.minor.width': 1 * scale_factor,
        # add pad space between ticks and labels
        'xtick.major.pad': 6 * scale_factor,
        'ytick.major.pad': 6 * scale_factor,
        'xtick.minor.pad': 6 * scale_factor,
        'ytick.minor.pad': 6 * scale_factor,
        # change width of markeredge 
        'lines.markeredgewidth': 1 * scale_factor,
        'legend.borderpad': 1 * scale_factor,

    }
    return updates_dict


def plot_fsc_average(fsc_average_halfmap_1, fsc_average_halfmap_2, figsize_mm=(45,26.5)):
    """
    Generate and save a plot of FSC averages for halfmaps 1 and 2.

    Parameters:
    - fsc_average_halfmap_1: dict, FSC average data for halfmap 1
    - fsc_average_halfmap_2: dict, FSC average data for halfmap 2
    - output_plot_folder: str, directory to save the output plot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import sys 
    sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])

    rcParams_updates = configure_plot_scaling(figsize_mm)
    rcParams_updates['legend.fontsize'] *= 0.8  # Reduce legend font size
    rcParams_updates['lines.linewidth'] *= 0.5  # Reduce line width
    rcParams_updates['lines.markersize'] *= 0.8  # Reduce marker size
    rcParams_updates['legend.borderpad'] *= 0.2  # Reduce border padding

    with temporary_rcparams(rcParams_updates):   
        # Prepare data for plotting
        xarray = list(fsc_average_halfmap_1.keys())
        yarray_halfmap2 = [fsc_average_halfmap_2[i][1] for i in fsc_average_halfmap_2.keys()]
        yarray_halfmap1 = [fsc_average_halfmap_1[i][1] for i in fsc_average_halfmap_1.keys()]

        # Plot data
        figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
        fig, ax = plt.subplots(figsize=figsize, dpi = 600)
        # plot halfmap 1 in white squares
        ax.plot(xarray, yarray_halfmap1, "ks-", label="Halfmap 1", \
                markeredgecolor="k", markerfacecolor="w")
        # plot halfmap 2 in black circles
        ax.plot(xarray, yarray_halfmap2, "ko-", label="Halfmap 2")
        xticks = [0, 15, 30]
        yticks = [0.45, 0.5, 0.55]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim(0, 30)
        ax.set_ylim(0.43, 0.57)

        # Customize plot
        ax.set_xlabel("Refinement Cycle")
        ax.set_ylabel(r'$\langle FSC \rangle$')
        ax.legend(loc="lower right")
        fig.tight_layout()
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2

    return fig

fig_average = plot_fsc_average(
            fsc_average_halfmap_1=fsc_cycles_halfmap1_with_averaging,
            fsc_average_halfmap_2=fsc_cycles_halfmap2_with_averaging,
            figsize_mm = (30, 18),
        )

fig_average.savefig("test_plotting_with_rcParams.eps", dpi=600)
print("Plot saved as test_plotting_with_rcParams.eps")