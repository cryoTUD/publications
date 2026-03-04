## IMPORTS
import os
import sys
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import sem, t
# Custom imports
from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist

# Set the seed for reproducibility
np.random.seed(42)

# Global variables
figure_number = 3
res_dict = {"0026" : 6.3, "0038" : 3.2, "0071" : 3.9, "0093" : 3.4, "0094" : 3.4, "0132" : 3.9, "0234" : 3.3, "0408" : 3.2, "0415" : 3.1, "4288" : 4.4, "0452" : 3.7, "0490" : 7.8, "0492" : 7.7, "0567" : 3.67, "0589" : 3.9, "0592" : 3.15, "0665" : 3.9, "0776" : 2.67, "10049" : 3.3, "10069" : 3.2, "10100" : 4.15, "10105" : 4.1, "10106" : 3.5, "10273" : 4.3, "10279" : 3.33, "10324" : 3.1, "10333" : 3.2, "10418" : 2.96, "10534" : 3.4, "10585" : 3.7, "10595" : 3.25, "10617" : 3.8, "20145" : 3.3, "20146" : 4.2, "20189" : 4.3, "20234" : 3.8, "20249" : 3.2, "20254" : 3.6, "20259" : 3.57, "20270" : 4, "20271" : 4.1, "20352" : 7.8, "20521" : 2.1, "20986" : 4.1, "21012" : 3.8, "21107" : 3.07, "21144" : 3.1, "21391" : 3.5, "3661" : 5.16, "3662" : 5.16, "3802" : 4.4, "3885" : 6.1, "3908" : 3.55, "4032" : 4.35, "4073" : 3.55, "4074" : 4.3, "4079" : 4.15, "4148" : 4, "4162" : 4.1, "4192" : 3.81, "4214" : 3.4, "4241" : 4.1, "4272" : 4.3, "4401" : 3.35, "4404" : 3.93, "4429" : 4.4, "4588" : 3.6, "4589" : 3.7, "4593" : 3.7, "4728" : 4.8, "4746" : 3.47, "4759" : 3.8, "4888" : 2.8, "4889" : 2.9, "4890" : 3.1, "4907" : 3.2, "4917" : 3.9, "4918" : 4.5, "4941" : 4, "4983" : 3.5, "7009" : 3.7, "7041" : 3.7, "7065" : 6.5, "7090" : 6.5, "7334" : 3.9, "7335" : 3.5, "8911" : 3.7, "8958" : 3.7, "8960" : 3.7, "9258" : 3.6, "9259" : 3.9, "9931" : 3.3, "9934" : 3.22, "9935" : 3.08, "9939" : 2.83, "9941" : 2.95, "9695" : 3.64, "0193" : 4.3, "0257" : 3.7, "0264" : 4.6, "0499" : 2.7, "10401" : 3.77, "20449" : 2.88, "20849" : 3.77, "4611" : 3.2, "4646" : 4.34, "4733" : 3.65, "4789" : 3.2, "7133" : 3.1, "7882" : 3.32, "8069" : 4.04, "9112" : 3.1, "9298" : 4.5, "9374" : 3.5, "0282" : 3.26, "0311" : 4.2, "0560" : 3.2, "10365" : 3.1, "20220" : 3.5, "20226" : 3.2, "3545" : 5.9, "4141" : 6.7, "4531" : 2.83, "4571" : 3.3, "4997" : 3.96, "7127" : 4.1, "7573" : 3.2, "8702" : 3.55, "9610" : 3.5}

## SETUP
def main():
    # Setup environment and define paths
    data_archive_path = setup_environment()

    # Define the path for the pickle file
    structured_data_folder = os.path.join(data_archive_path, "processed", "structured_data", f"figure_{figure_number}")
    pickle_file = os.path.join(structured_data_folder, f"figure_{figure_number}_phase_correlations.pickle")
    
    # Output paths
    output_plots_folder = os.path.join(data_archive_path, "outputs", f"figure_{figure_number}")
    
    # Assert the pickle file exists
    assert_paths_exist(pickle_file)
    create_folders_if_they_do_not_exist(output_plots_folder)

    

    # Load the phase correlations data from the pickle file
    with open(pickle_file, "rb") as f:
        phase_correlations_data = pickle.load(f)

    fontsize = 3
    plt.rcParams.update({'font.size': fontsize})
    sns.set_theme(context="paper", font="Helvetica", font_scale=1)
    sns.set_style("white")

    figsize_mm = (31.896*2, 27.809*2)
    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    fig_phase_corr, ax_phase_corr = plt.subplots(figsize=figsize)
    fig_amp_corr, ax_amp_corr = plt.subplots(figsize=figsize)
    # Normalize the resolution values
    res_values_for_analysis = np.array([res_dict[emdb_id] for emdb_id in phase_correlations_data.keys()])
    norm = Normalize(vmin=np.min(res_values_for_analysis), vmax=np.max(res_values_for_analysis))
    res_values_for_emdb_ids_in_this_analysis = {emdb_id: res for emdb_id, res in res_dict.items() if emdb_id in phase_correlations_data}
    sorted_res_values_for_this_analysis_dictionary = {k: v for k, v in sorted(res_values_for_emdb_ids_in_this_analysis.items(), key=lambda item: item[1])}

    cmap = cm.rainbow

    # Assign colors to each EMDB ID based on resolution 
    colors = {emdb_id: cmap(norm(res)) for emdb_id, res in res_dict.items()}

    # Iterate over EMDB IDs and plot
    for emdb_id, resolution in tqdm(sorted_res_values_for_this_analysis_dictionary.items(), desc="Plotting EMDB IDs"):
        data = phase_correlations_data[emdb_id]
        freq = np.array(data["freq"])
        phase_corr = np.array(data["phase_correlations_unsharp"])
        amp_corr = np.array(data["amplitude_correlations_unsharp"])

        # Convert to DataFrame for Seaborn
        phase_df = pd.DataFrame({
            'Frequency': np.tile(freq, phase_corr.shape[0]),
            'Correlation': phase_corr.flatten(),
            'Type': 'Phase'
        })
        amp_df = pd.DataFrame({
            'Frequency': np.tile(freq, amp_corr.shape[0]),
            'Correlation': amp_corr.flatten(),
            'Type': 'Amplitude'
        })

        # Phase Correlation Plot
        sns.lineplot(x="Frequency", y="Correlation", data=phase_df, ax=ax_phase_corr, label=f"{emdb_id}({resolution:.1f} $\AA$)",\
                     errorbar=('ci', 95), color=colors[emdb_id]
                    )
        # Amplitude Correlation Plot
        sns.lineplot(x="Frequency", y="Correlation", data=amp_df, ax=ax_amp_corr, label=f"{emdb_id}({resolution:.1f} $\AA$)",\
                     errorbar=('ci', 95), color=colors[emdb_id]
                    )
        
    # Set the labels
    ax_phase_corr.set_xlabel(r"Spatial Frequency ($\AA^{-1}$)")
    ax_phase_corr.set_ylabel("Phase Correlation")
    ax_phase_corr.set_ylim(-0.1, 1.1)
    ax_phase_corr.set_yticks([0, 0.5, 1])
    ax_phase_corr.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=fontsize)
    # change x ticks fontsize
    #ax_phase_corr.tick_params(axis='x')

    ax_amp_corr.set_xlabel(r"Spatial Frequency ($\AA^{-1}$)")
    ax_amp_corr.set_ylabel("Amplitude Correlation")
    ax_amp_corr.set_ylim(-0.1, 1.1)
    ax_amp_corr.set_yticks([0, 0.5, 1])
    ax_amp_corr.legend(loc='lower left', fontsize=fontsize)

    # Space between ticks and labels and ticks and axes
    ax_phase_corr.tick_params(axis='both', which='major', pad=1)
    ax_amp_corr.tick_params(axis='both', which='major', pad=1)
    # change y ticks fontsize

    # change x ticks fontsize
    #ax_amp_corr.tick_params(axis='x')

    fig_phase_corr.tight_layout()
    fig_amp_corr.tight_layout()

    # Save the plots
    fig_save_phase_corr = os.path.join(output_plots_folder, f"figure_{figure_number}_phase_correlations.pdf")
    fig_save_amp_corr = os.path.join(output_plots_folder, f"figure_{figure_number}_amplitude_correlations_unsharp.pdf")
    fig_save_phase_corr_png = os.path.join(output_plots_folder, f"figure_{figure_number}_phase_correlations.png")
    fig_save_amp_corr_png = os.path.join(output_plots_folder, f"figure_{figure_number}_amplitude_correlations_unsharp.png")
    fig_phase_corr.savefig(fig_save_phase_corr, format="pdf", bbox_inches="tight", dpi=600)
    fig_phase_corr.savefig(fig_save_phase_corr_png, format="png", bbox_inches="tight", dpi=600)
    fig_amp_corr.savefig(fig_save_amp_corr, format="pdf", bbox_inches="tight", dpi=600)
    fig_amp_corr.savefig(fig_save_amp_corr_png, format="png", bbox_inches="tight", dpi=600)

    print(f"Saved the phase correlation plot to {fig_save_phase_corr}")
    print(f"Saved the amplitude correlation plot to {fig_save_amp_corr}")

 





if __name__ == "__main__":
    main()
