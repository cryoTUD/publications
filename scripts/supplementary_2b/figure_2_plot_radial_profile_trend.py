## IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
from scripts.utils.plot_utils import *
import numpy as np
import gemmi
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import random 
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.pyplot import cm
import matplotlib as mpl
import pickle

# Custom imports
from locscale.preprocessing.pseudomodel_classes import extract_model_from_mask
from locscale.include.emmer.ndimage.map_utils import load_map, measure_mask_parameters, save_as_mrc
from locscale.include.emmer.ndimage.filter import get_cosine_mask
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region, estimate_global_bfactor_map_standard
from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps, calculate_amplitude_correlation_maps
from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array

# Import helper functions 
from figure_2_functions import gradient_solver, create_modmap
from scripts.utils.plot_utils import pretty_plot_radial_profile, temporary_rcparams, configure_plot_scaling
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist
# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set global variables
SAVE_FIGURE = True
figure_number = 2
## SETUP 
def main():
    data_archive_path = setup_environment()

    # DEFINE THE PATHS 
    #input_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/pseudomodel_during_iterations/hybrid_pseudomodel_iterations"
    input_maps_folder = os.path.join(data_archive_path, "raw","maps")
    output_maps_folder = os.path.join(data_archive_path, "processed", "maps")
    input_pdbs_folder = os.path.join(data_archive_path, "raw", "pdbs")
    output_pdbs_folder = os.path.join(data_archive_path, "processed", "pdbs")
    output_general_folder = os.path.join(data_archive_path, "processed", "general")
    output_structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data")
    figure_output_folder = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}")
 
    output_pseudomodel_structure_folder = os.path.join(output_pdbs_folder, f"figure_{figure_number}", "pseudomodel_structures_iterations")
    output_pseudomodel_map_folder = os.path.join(output_maps_folder, f"figure_{figure_number}", "pseudomodel_maps")
    output_folder_for_data = os.path.join(output_structured_data_folder, f"figure_{figure_number}")
    output_processing_folder = os.path.join(output_general_folder, f"figure_{figure_number}_processed")
    
    create_folders_if_they_do_not_exist(\
        figure_output_folder,\
    )

    # Output paths
    suffix = "_3061"
    path_to_store_fsc_average_pickle = os.path.join(output_folder_for_data, f"fsc_average_iterations{suffix}.pickle")
    path_to_store_radial_profile_pickle = os.path.join(output_folder_for_data, f"radial_profile_iterations{suffix}.pickle")

    # Figure output paths
    radial_profile_fig_path = os.path.join(figure_output_folder, f"figure_{figure_number}_radial_profiles_iteration{suffix}.pdf")
    radial_profile_fig_path_png = os.path.join(figure_output_folder, f"figure_{figure_number}_radial_profiles_iteration{suffix}.png")
    zoom_radial_profile_fig_path = os.path.join(figure_output_folder, f"figure_{figure_number}_radial_profiles_iteration_zoom{suffix}.pdf")
    zoom_radial_profile_fig_path_png = os.path.join(figure_output_folder, f"figure_{figure_number}_radial_profiles_iteration_zoom.png")

    assert os.path.exists(path_to_store_fsc_average_pickle), f"Path does not exist: {path_to_store_fsc_average_pickle}"
    assert os.path.exists(path_to_store_radial_profile_pickle), f"Path does not exist: {path_to_store_radial_profile_pickle}"

    # Load the data
    with open(path_to_store_radial_profile_pickle, "rb") as f:
        radial_profiles_iterations = pickle.load(f)
    
    freq = radial_profiles_iterations["other_info"]["freq"]
    iterations = [x for x in radial_profiles_iterations.keys() if x != "other_info"]
    sharpened_profiles = [radial_profiles_iterations[i] * np.exp(freq * 20 / 4) for i in iterations]
    normalised_profiles = [sharpened_profiles[i]/sharpened_profiles[i].max() for i in iterations]
    normalised_profiles += [radial_profiles_iterations["other_info"]["reference_normalised"]*0.6]   # 1.3 for 8702 
    legend_labels = [f"iteration {i}" for i in iterations] + ["reference"]
    # Plot the radial profiles

    figsize_mm = (60,60)
    rcParams_new = configure_plot_scaling(figsize_mm=figsize_mm)
    rcParams_new["font.size"] = 8
    with temporary_rcparams(rcParams_new):
        figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
        # fig, ax1, ax2 = pretty_plot_radial_profile(freq, normalised_profiles,figsize_mm=figsize_mm, legends=legend_labels,
        #                                 plot_type="make_log", showlegend=False,fontscale=1.2, linewidth=1)
        
        list_of_squared_freq = [freq**2] * len(normalised_profiles)
        list_of_log_amplitudes = [np.log(x) for x in normalised_profiles]
        
        xlabel = r'Spatial Frequency, $d^{-2} (\AA^{-2})$'
        xlabel_top = r'Resolution, d $(\AA)$'
        ylabel = r'$ln  \langle \mid F \mid \rangle $'

        # map the colors to the iterations
        colors = cm.turbo(np.linspace(0, 1, len(normalised_profiles)-1))
        
        fig, ax1 = plt.subplots(figsize=figsize_in, dpi=600)
        ax2 = ax1.twiny()
        for i in range(len(normalised_profiles)-1):
            ax1.plot(list_of_squared_freq[i], list_of_log_amplitudes[i], \
                    color=colors[i], linewidth=1)
        
        # Plot the reference in black
        ax1.plot(list_of_squared_freq[-1], list_of_log_amplitudes[-1], \
                    color="black", linewidth=1, linestyle="--")
        
        xticks = [0, 0.05, 0.1, 0.15]
        ax1.set_xticks(xticks)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel_top)

        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())
        xticklabels = [round(1/np.sqrt(x),1) for x in ax1.get_xticks()]
        xticklabels[0] = r"$\infty$"
        ax2.set_xticklabels(xticklabels)

        ax1.set_yticks([0, -2, -4, -6])
        # change major and minor ticks
        # ax1.tick_params(axis='both', which='major', width=2, length=2)
        fig.tight_layout()
        
        fig.savefig(radial_profile_fig_path, dpi=600, bbox_inches='tight')
        fig.savefig(radial_profile_fig_path_png, dpi=600, bbox_inches='tight')

        # Zoomed in plot
    
    figsize_mm = (18,18)
    rcParams_new = configure_plot_scaling(figsize_mm=figsize_mm)
    with temporary_rcparams(rcParams_new):
        fig_zoom, ax1_zoom, ax2_zoom = pretty_plot_radial_profile(freq, normalised_profiles,figsize_mm=figsize_mm, legends=["emmap"]+legend_labels, \
                                        plot_type="make_log", showlegend=False,fontscale=0.2, linewidth=0.2, crop_freq=[50,5], ylims=[-4.5,-2.5], xlims=[1/50**2, 1/5**2] \
                                        )

        # Change the border line thickness 
        thickness = 0.5
        ax1_zoom.spines['top'].set_linewidth(thickness)
        ax1_zoom.spines['right'].set_linewidth(thickness)
        ax1_zoom.spines['bottom'].set_linewidth(thickness)
        ax1_zoom.spines['left'].set_linewidth(thickness)
        ax2_zoom.spines['top'].set_linewidth(thickness)
        ax2_zoom.spines['right'].set_linewidth(thickness)
        ax2_zoom.spines['bottom'].set_linewidth(thickness)
        ax2_zoom.spines['left'].set_linewidth(thickness)
        # Make the last plot into black color and dashed line
        ax1_zoom.lines[-1].set_color('black')
        ax1_zoom.lines[-1].set_linewidth(thickness)
        ax1_zoom.lines[-1].set_linestyle('--')    

        ax1_zoom.tick_params(axis='both', which='major', width=0.1, length=1)
        ax2_zoom.tick_params(axis='both', which='major', width=0.1, length=1)
        # Hide ticks in x and y
        ax1_zoom.set_xticks([])
        # ax2_zoom.set_xticks([])
        
        # #ax1_zoom.set_visible(False)
        # Hide ticks in x and y 
        ax1_zoom.set_xticks([])
        #ax2_zoom.set_xticks([])
        resolution_ticks = [10, 6.3, 5]
        yaxis_ticks = [-3, -4]
        fontsize = 3
        ax1_zoom.set_yticks(yaxis_ticks, size=0.01)
        ax1_zoom.set_yticklabels(yaxis_ticks, fontsize=fontsize)
        # # ax2_zoom set x limits between 20 and 5
        # ax2_zoom.set_xlim(20, 5)
        # ax2_zoom.set_xticks([x for x in ax2_zoom.get_xticks()], size=0.01) 
        # ax2_zoom.set_xticks(resolution_ticks)
        # print bounds of x axis
        #print(ax1_zoom.get_xbound())
        #print(ax2_zoom.get_xbound())
        #ax2_zoom.set_xbound([1/np.sqrt(x) for x in ax2_zoom.get_xbound()])
        #print(ax2_zoom.get_xbound())
        #ax2_zoom.set_xticks(resolution_ticks)
        ax2_zoom.set_xticklabels([f"{1/np.sqrt(x):.1f}" for x in ax2_zoom.get_xticks()], fontsize=fontsize)
        # # Hide the labels in x and y
        ax1_zoom.set_xlabel("")
        ax1_zoom.set_ylabel(r'$ln  \langle \mid F \mid \rangle $ ', fontsize=fontsize)
        ax2_zoom.set_xlabel("d ($\AA$)", fontsize=fontsize)
        # # Change the pad distance between labels and ticks and plot 
        pad_distance = 0.1
        ax1_zoom.xaxis.labelpad = pad_distance
        ax1_zoom.yaxis.labelpad = pad_distance
        ax2_zoom.xaxis.labelpad = pad_distance  
        ax2_zoom.yaxis.labelpad = pad_distance
        # Change the distance between plot and labels
        ax1_zoom.xaxis.set_tick_params(pad=0.01)
        ax1_zoom.yaxis.set_tick_params(pad=0.01)
        ax2_zoom.xaxis.set_tick_params(pad=0.01)
        ax2_zoom.yaxis.set_tick_params(pad=0.01)

        # hide ax2_zoom x axis tick marks but keep the labels
        ax2_zoom.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # add colorbar turbo 
        # cbar_ax = fig_zoom.add_axes([0.92, 0.15, 0.01, 0.7])
        # norm = Normalize(vmin=1, vmax=50)  # Mapping range between 1 and 50

        # colorbar = ColorbarBase(cbar_ax, cmap="turbo", norm=norm, orientation='vertical')
        #colorbar.set_label('Iteration number', fontsize=fontsize)
        #colorbar.ax.tick_params(labelsize=fontsize)

        fig_zoom.tight_layout()

        
        if SAVE_FIGURE:
            fig_zoom.savefig(zoom_radial_profile_fig_path, dpi=600)
            fig_zoom.savefig(zoom_radial_profile_fig_path_png, dpi=600)

        # Draw a rectangular color bar 
        fig_color, ax_color = create_colormap_rectangle(rect_width_mm=6.5, rect_height_mm=16)
        if SAVE_FIGURE:
            fig_color.savefig(os.path.join(figure_output_folder, f"figure_{figure_number}_colorbar.eps"), dpi=600)
            fig_color.savefig(os.path.join(figure_output_folder, f"figure_{figure_number}_colorbar.png"), dpi=600)

if __name__ == "__main__":
    # Print the start time
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {processing_time}")
    print("="*80)


    




    

