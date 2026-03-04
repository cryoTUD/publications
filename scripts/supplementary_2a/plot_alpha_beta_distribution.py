# 1) IMPORTS 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])  # This is mandatory  [oai_citation:0‡pattern_analysis_script.py](file-service://file-7C9tD7A3baLjYuPCR88Yve)

# General imports
import json
import glob
import numpy as np

import random
import matplotlib.pyplot as plt
from matplotlib import ticker
from datetime import datetime
import tobvalid
# Helper functions
from scripts.utils.general import setup_environment, create_folders_if_they_do_not_exist, assert_paths_exist
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams
# 2) Set the seed for reproducibility  
random.seed(42)  # seed for reproducibility  [oai_citation:1‡pattern_analysis_script.py](file-service://file-7C9tD7A3baLjYuPCR88Yve)
np.random.seed(42)

# 3) Global variables
EMDB_PDB_ids_training = ["0026_6gl7", "7573_6crv",  "0665_6oa9", "0038_6gml", "0071_6gve", "0093_6gyn", "0094_6gyo", "0132_6h3c", "0234_6hjn", "0408_6nbd", "0415_6nbq", "4288_6fo2", "0452_6nmi", "0490_6nr8", "0492_6nra", "0567_6o0h", "0589_6nmi", "0592_6o1m", "0776_6ku9", "10049_6rx4", "10069_6s01", "10100_6s5t", "10105_6s6t", "10106_6s6u", "10273_6sof", "10279_6sp2", "10324_6swe", "10333_6swy", "10418_6t9n", "10534_6tni", "10585_6ttu", "10595_6tut", "10617_6xt9", "20145_6oo4", "20146_6oo5", "20189_6osy", "20234_6p19", "20249_6p4h", "20254_6p5a", "20259_6p62", "20270_6p7v", "20271_6p7w", "20352_6pik", "20521_6pxm", "20986_6v0b", "21012_6v1i", "21107_6v8o", "21144_6vbu", "21391_6vv5", "3661_5no2", "3662_5no3", "3802_5of4", "3885_6el1", "3908_6eoj", "4032_5lc5", "4073_5lmn", "4074_5lmo", "4079_5lmt", "4148_5m3m", "4162_6ezo", "4192_6f6w", "4214_6fai", "4241_6fe8", "4272_6fki", "4401_6i2x", "4404_6i3m", "4429_6i84", "4588_6qm5", "4589_6qm6", "4593_6qma", "4728_6r5k", "4746_6r7x", "4759_6r8f", "4888_6ric", "4889_6rid", "4890_6rie", "4907_6rkd", "4917_6rla", "4918_6rlb", "4941_6rn3", "4983_6rqj", "7009_6ave", "7041_6b3q", "7065_6b7y", "7090_6bf6", "7334_6c23", "7335_6c24", "8911_6dt0", "8958_6e1n", "8960_6e1p", "9258_6muw", "9259_6mux", "9931_6k7g", "9934_6k7i", "9935_6k7j", "9939_6k7l", "9941_6k7m", "9695_6iok"]
EMDB_PDB_ids_validation = ["0193_6hcg", "0257_6hra", "0264_6hs7", "0499_6nsk", "10401_6t8h", "20449_6pqo", "20849_6uqk", "4611_6qp6", "4646_6qvb", "4733_6r69", "4789_6rb9", "7133_6bqv", "7882_6dg7", "8069_5i08", "9112_6mgv", "9298_6mzc", "9374_6nhv"]
EMDB_PDB_ids_epsilon = ["0282_6huo", "0311_6hz5", "0560_6nzu", "10365_6t23", "20220_6oxl", "20226_6p07", "3545_5mqf", "4141_5m1s", "4531_6qdw", "4571_6qk7", "4997_6rtc", "7127_6bpq",  "8702_5vkq", "9610_6adq"]
EMDB_PDB_ids_all = EMDB_PDB_ids_training + EMDB_PDB_ids_validation + EMDB_PDB_ids_epsilon

emdb_to_pdb = {x.split("_")[0]: x.split("_")[1] for x in EMDB_PDB_ids_all}

emdb_contains_membrane = lambda emdb_id: os.path.exists(f"/home/abharadwaj1/papers/elife_paper/figure_information/data/pdb_containing_membrane/{emdb_to_pdb[emdb_id]}.pdb")
pdb_contains_membrane = lambda pdb_id: os.path.exists(f"/home/abharadwaj1/papers/elife_paper/figure_information/data/pdb_containing_membrane/{pdb_id}.pdb")
# 4) SETUP 
def main():
    data_archive_path = setup_environment()  # This is mandatory  [oai_citation:2‡pattern_analysis_script.py](file-service://file-7C9tD7A3baLjYuPCR88Yve)

    # — Input JSON files (absolute pattern)
    json_pattern = os.path.join(
        "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data",
        "structured_data", "supplementary_2a", "tobvalid_analysis",
        "albe_results",
        #"pseudomodel",
        "PDB_*_unrefined_shifted_servalcat_refined_servalcat_refined_mixture.json",
        #"emd_*_FDR_confidence_final_gradient_pseudomodel_proper_element_composition_averaged_mixture.json"
    )
    json_files = glob.glob(json_pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found matching: {json_pattern}")

    # — Output figure folder
    output_fig_folder = os.path.join(
        data_archive_path, "outputs", f"supplementary_2a"
    )
    create_folders_if_they_do_not_exist(output_fig_folder)

    # — Extract α and √β from all JSONs
    alphas = []
    sqrt_betas = []
    alphas_first_model = []
    sqrt_betas_first_model = []
    alphas_second_model = []
    sqrt_betas_second_model = []
    has_membrane = []
    print("Length of json_files:", len(json_files))
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        for a, b in zip(data["Alpha"], data["Beta"]):
            alphas.append(a)
            sqrt_betas.append(np.sqrt(b))
        alphas_first_model.append(data["Alpha"][0])
        sqrt_betas_first_model.append(np.sqrt(data["Beta"][0]))
        alphas_second_model.append(data["Alpha"][1])
        sqrt_betas_second_model.append(np.sqrt(data["Beta"][1]))
        emdb_id = os.path.basename(jf).split("_")[1]
        #has_membrane.append(emdb_contains_membrane(emdb_id))

    alphas = np.array(alphas)
    sqrt_betas = np.array(sqrt_betas)

    # — Load 2D KDE and grid from tobvalid templates
    d = os.path.dirname(tobvalid.__file__)
    xx = np.load(os.path.join(d, "templates", "xx.npy"))
    yy = np.load(os.path.join(d, "templates", "yy.npy"))
    kde = np.load(os.path.join(d, "templates", "albe_kde.npy"))

    # — Plot: scatter + filled contour
    figsize_mm = (60, 50)
    rcparams = configure_plot_scaling(figsize_mm)
    fontsize = 8
    rcparams["font.size"] = fontsize
    with temporary_rcparams(rcparams):
        figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        fig, ax = plt.subplots(1, 1, figsize=figsize_in, dpi=600)
        #ax.scatter(alphas, sqrt_betas, marker='x', alpha=0.1)
        ax.scatter(alphas_first_model, sqrt_betas_first_model, marker='x', alpha=0.1, label='mode 1', color='blue')
        ax.scatter(alphas_second_model, sqrt_betas_second_model, marker='.', alpha=0.1, label='mode 2', color='orange')
        # # mark membrane-containing structures
        # for a, b, m in zip(alphas_second_model, sqrt_betas_second_model, has_membrane):
        #     if m:
        #         ax.scatter(a, b, marker='o', color='black', s=1, alpha=0.3)
        
        # for a, b, m in zip(alphas_first_model, sqrt_betas_first_model, has_membrane):
        #     if m:
        #         ax.scatter(a, b, marker='*', color='red', s=1, alpha=0.3)
        N = 30
        locator = ticker.MaxNLocator(N + 1, min_n_ticks=N)
        lev = locator.tick_values(kde.min(), kde.max())
        cfset = ax.contourf(xx, yy, kde, cmap='Reds', levels=lev[1:])
        fig.colorbar(cfset)

        # — Axis limits
        y_max = 75 # max(max(alphas.max() * 3, sqrt_betas.max()) + 3, 45)
        x_max = y_max // 3
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])

        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\sqrt{\beta}$')
        # plt.title("albe_distribution")
        plt.tight_layout()

        # — Save as PDF (DPI 600)
        output_pdf = os.path.join(output_fig_folder, "albe_distribution_atomic_model.pdf")
        #fig.legend(fontsize=fontsize//2)
    fig.savefig(output_pdf, format='pdf', dpi=600)
    print(f"Figure saved to: {output_pdf}")


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing time: {end_time - start_time}")
    print("=" * 80)