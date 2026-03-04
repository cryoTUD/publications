# Write a script to plot the correlation curves for iterative and non-iterative pseudo-model refinement

from genericpath import isfile
import os
import sys
import shutil
import subprocess
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

EMDB_PDB_ids_training = ["0026_6gl7", "7573_6crv",  "0665_6oa9", "0038_6gml", "0071_6gve", "0093_6gyn", "0094_6gyo", "0132_6h3c", "0234_6hjn", "0408_6nbd", "0415_6nbq", "4288_6fo2", "0452_6nmi", "0490_6nr8", "0492_6nra", "0567_6o0h", "0589_6nmi", "0592_6o1m", "0776_6ku9", "10049_6rx4", "10069_6s01", "10100_6s5t", "10105_6s6t", "10106_6s6u", "10273_6sof", "10279_6sp2", "10324_6swe", "10333_6swy", "10418_6t9n", "10534_6tni", "10585_6ttu", "10595_6tut", "10617_6xt9", "20145_6oo4", "20146_6oo5", "20189_6osy", "20234_6p19", "20249_6p4h", "20254_6p5a", "20259_6p62", "20270_6p7v", "20271_6p7w", "20352_6pik", "20521_6pxm", "20986_6v0b", "21012_6v1i", "21107_6v8o", "21144_6vbu", "21391_6vv5", "3661_5no2", "3662_5no3", "3802_5of4", "3885_6el1", "3908_6eoj", "4032_5lc5", "4073_5lmn", "4074_5lmo", "4079_5lmt", "4148_5m3m", "4162_6ezo", "4192_6f6w", "4214_6fai", "4241_6fe8", "4272_6fki", "4401_6i2x", "4404_6i3m", "4429_6i84", "4588_6qm5", "4589_6qm6", "4593_6qma", "4728_6r5k", "4746_6r7x", "4759_6r8f", "4888_6ric", "4889_6rid", "4890_6rie", "4907_6rkd", "4917_6rla", "4918_6rlb", "4941_6rn3", "4983_6rqj", "7009_6ave", "7041_6b3q", "7065_6b7y", "7090_6bf6", "7334_6c23", "7335_6c24", "8911_6dt0", "8958_6e1n", "8960_6e1p", "9258_6muw", "9259_6mux", "9931_6k7g", "9934_6k7i", "9935_6k7j", "9939_6k7l", "9941_6k7m", "9695_6iok"]
EMDB_PDB_ids_validation = ["0193_6hcg", "0257_6hra", "0264_6hs7", "0499_6nsk", "10401_6t8h", "20449_6pqo", "20849_6uqk", "4611_6qp6", "4646_6qvb", "4733_6r69", "4789_6rb9", "7133_6bqv", "7882_6dg7", "8069_5i08", "9112_6mgv", "9298_6mzc", "9374_6nhv"]
EMDB_PDB_ids_epsilon = ["0282_6huo", "0311_6hz5", "0560_6nzu", "10365_6t23", "20220_6oxl", "20226_6p07", "3545_5mqf", "4141_5m1s", "4531_6qdw", "4571_6qk7", "4997_6rtc", "7127_6bpq",  "8702_5vkq", "9610_6adq"]
EMDB_PDB_ids_all = EMDB_PDB_ids_training + EMDB_PDB_ids_validation + EMDB_PDB_ids_epsilon

res_dict = {"0026" : 6.3, "0038" : 3.2, "0071" : 3.9, "0093" : 3.4, "0094" : 3.4, "0132" : 3.9, "0234" : 3.3, "0408" : 3.2, "0415" : 3.1, "4288" : 4.4, "0452" : 3.7, "0490" : 7.8, "0492" : 7.7, "0567" : 3.67, "0589" : 3.9, "0592" : 3.15, "0665" : 3.9, "0776" : 2.67, "10049" : 3.3, "10069" : 3.2, "10100" : 4.15, "10105" : 4.1, "10106" : 3.5, "10273" : 4.3, "10279" : 3.33, "10324" : 3.1, "10333" : 3.2, "10418" : 2.96, "10534" : 3.4, "10585" : 3.7, "10595" : 3.25, "10617" : 3.8, "20145" : 3.3, "20146" : 4.2, "20189" : 4.3, "20234" : 3.8, "20249" : 3.2, "20254" : 3.6, "20259" : 3.57, "20270" : 4, "20271" : 4.1, "20352" : 7.8, "20521" : 2.1, "20986" : 4.1, "21012" : 3.8, "21107" : 3.07, "21144" : 3.1, "21391" : 3.5, "3661" : 5.16, "3662" : 5.16, "3802" : 4.4, "3885" : 6.1, "3908" : 3.55, "4032" : 4.35, "4073" : 3.55, "4074" : 4.3, "4079" : 4.15, "4148" : 4, "4162" : 4.1, "4192" : 3.81, "4214" : 3.4, "4241" : 4.1, "4272" : 4.3, "4401" : 3.35, "4404" : 3.93, "4429" : 4.4, "4588" : 3.6, "4589" : 3.7, "4593" : 3.7, "4728" : 4.8, "4746" : 3.47, "4759" : 3.8, "4888" : 2.8, "4889" : 2.9, "4890" : 3.1, "4907" : 3.2, "4917" : 3.9, "4918" : 4.5, "4941" : 4, "4983" : 3.5, "7009" : 3.7, "7041" : 3.7, "7065" : 6.5, "7090" : 6.5, "7334" : 3.9, "7335" : 3.5, "8911" : 3.7, "8958" : 3.7, "8960" : 3.7, "9258" : 3.6, "9259" : 3.9, "9931" : 3.3, "9934" : 3.22, "9935" : 3.08, "9939" : 2.83, "9941" : 2.95, "9695" : 3.64, "0193" : 4.3, "0257" : 3.7, "0264" : 4.6, "0499" : 2.7, "10401" : 3.77, "20449" : 2.88, "20849" : 3.77, "4611" : 3.2, "4646" : 4.34, "4733" : 3.65, "4789" : 3.2, "7133" : 3.1, "7882" : 3.32, "8069" : 4.04, "9112" : 3.1, "9298" : 4.5, "9374" : 3.5, "0282" : 3.26, "0311" : 4.2, "0560" : 3.2, "10365" : 3.1, "20220" : 3.5, "20226" : 3.2, "3545" : 5.9, "4141" : 6.7, "4531" : 2.83, "4571" : 3.3, "4997" : 3.96, "7127" : 4.1, "7573" : 3.2, "8702" : 3.55, "9610" : 3.5}

symmetry_dictionary = {'0026': 'C2', '0038': 'C1', '0071': 'D2', '0093': 'C4', '0094': 'C4', '0132': 'C2', '0234': 'C3', '0408': 'C2', '0415': 'C1', '4288': 'C2', '0452': 'C1', '0490': 'C1', '0492': 'C1', '0567': 'D2', '0589': 'C1', '0592': 'C2', '0665': 'C1', '0776': 'C3', '10049': 'C1', '10069': 'C1', '10100': 'C1', '10105': 'C1', '10106': 'C2', '10273': 'C1', '10279': 'C6', '10324': 'C1', '10333': 'C1', '10418': 'C4', '10534': 'C1', '10585': 'C1', '10595': 'C1', '10617': 'C1', '20145': 'C2', '20146': 'C2', '20189': 'C3', '20234': 'C1', '20249': 'C1', '20254': 'C2', '20259': 'C3', '20270': 'C1', '20271': 'C1', '20352': 'C2', '20521': 'O', '20986': 'C5', '21012': 'C9', '21107': 'C1', '21144': 'C1', '21391': 'C3', '3661': 'C1', '3662': 'C1', '3802': 'C1', '3885': 'C10', '3908': 'C1', '4032': 'C1', '4073': 'C1', '4074': 'C1', '4079': 'C1', '4148': 'C1', '4162': 'C2', '4192': 'C1', '4214': 'C1', '4241': 'C1', '4272': 'C1', '4401': 'C1', '4404': 'C2', '4429': 'C1', '4588': 'C2', '4589': 'C2', '4593': 'C2', '4728': 'C1', '4746': 'C2', '4759': 'C1', '4888': 'C1', '4889': 'C1', '4890': 'C1', '4907': 'D3', '4917': 'C2', '4918': 'C1', '4941': 'C1', '4983': 'C1', '7009': 'C3', '7041': 'C1', '7065': 'C1', '7090': 'C2', '7334': 'C1', '7335': 'C1', '8911': 'C2', '8958': 'C2', '8960': 'C2', '9258': 'C2', '9259': 'C1', '9931': 'C1', '9934': 'C1', '9935': 'C1', '9939': 'C1', '9941': 'C1', '9695': 'C1', '0193': 'C15', '0257': 'C1', '0264': 'C5', '0499': 'C6', '10401': 'C1', '20449': 'C4', '20849': 'C4', '4611': 'C2', '4646': 'C2', '4733': 'C1', '4789': 'C7', '7133': 'C4', '7882': 'C5', '8069': 'C3', '9112': 'C2', '9298': 'C1', '9374': 'C1', '0282': 'C1', '0311': 'C2', '0560': 'C2', '10365': 'C1', '20220': 'C1', '20226': 'C1', '3545': 'C1', '4141': 'C1', '4531': 'C1', '4571': 'C1', '4997': 'C2', '7127': 'C4', '7573': 'C3', '8702': 'C4', '9610': 'C2'}


# %%

input_files = {}
input_files_MB = {}

emdb_pdbs_present = []
for emdb_pdb in EMDB_PDB_ids_all:
    input_files_model_free = get_input_files_for_correlation_curves(emdb_pdb, "MF")
    input_files_model_based = get_input_files_for_correlation_curves(emdb_pdb, "MB")
    
    required_files = ["target_pdb_path", "unsharpened_map_file","unrestrained_pseudomodel_refined"]
    print("Checking if all required files exist for {}".format(emdb_pdb))
    print("===============================================")
    # Check if all required files exist and print a warning if not
    missing_files = []
    for required_file in required_files:
        if not os.path.exists(input_files_model_free[required_file]):
            print("\t{} missing".format(required_file), "MF", input_files_model_free[required_file])
            missing_files.append((required_file, "MF"))
        if input_files_model_based[required_file] is None:
            print("\t{} missing".format(required_file), "MB")
            missing_files.append((required_file, "MB"))


    if len(missing_files) == 0:
        emdb_pdbs_present.append(emdb_pdb)
        input_files[emdb_pdb] = {}
        for key in required_files:
            input_files[emdb_pdb][key] = input_files_model_free[key]
        input_files_MB[emdb_pdb] = {}
        for key in required_files:
            input_files_MB[emdb_pdb][key] = input_files_model_based[key]
    

# Number of EMDBs with all required files
print("Number of EMDBs with all required files: {}".format(len(input_files)))
print("===============================================")
print("Number of EMDBs with missing files: {}".format(missing_files))        
print("===============================================")

# import pickle
# output_folder_MB = "/home/abharadwaj1/dev/map_sharpening/locscale_analysis/neighborhood_correlations_output/correlation_dictionary_3"
# # def read_all_pickle_files():
# #     neighborhood_bfactor_correlation_emdbs = {}
# #     for pickle_file in os.listdir(output_folder_MB):
# #         if pickle_file.endswith(".pickle"):
# #             with open(os.path.join(output_folder_MB, pickle_file), "rb") as f:
# #                 neighborhood_bfactor_correlation_emdbs[pickle_file.split("_")[0]] = pickle.load(f)
# #     return neighborhood_bfactor_correlation_emdbs

# # neighborhood_bfactor_correlation_emdbs_MB = read_all_pickle_files()


# import matplotlib.pyplot as plt
# from locscale.include.emmer.pdb.pdb_tools import neighborhood_bfactor_correlation_sample
# from tqdm import tqdm
# import seaborn as sns
output_folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/neighborhood_correlations/correlation_curves_pickle_files/correlation_curve_2_correct_MB_pickle_files"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# bfactor_neighborhood_correlation_iterative_all_emdb = {}
# bfactor_neighborhood_correlation_non_averaged_all_emdb = {}
# neighborhood_bfactor_correlation_emdbs_MB = {}
# for emdb_pdb in input_files.keys():
#     if emdb_pdb == "0026_6gl7":
#         continue
    
#     emdb, pdb = emdb_pdb.split("_")

    
#     refined_iterative_model_path = input_files[emdb_pdb]["target_pdb_path"]
#     refined_non_averaged_model_path = input_files[emdb_pdb]["alpha_pseudomodel_refined_path"]
#     refined_pdb_path = input_files_MB[emdb_pdb]["target_pdb_path"]
#     if refined_pdb_path is None:
#       print("Refined pdb path does not exist for {}".format(emdb_pdb))
#       continue

#     bfactor_neighborhood_correlation_iterative = neighborhood_bfactor_correlation_sample(refined_iterative_model_path, max_radius=10, num_steps=10)
#     bfactor_neighborhood_correlation_non_averaged = neighborhood_bfactor_correlation_sample(refined_non_averaged_model_path, max_radius=10, num_steps=10)
#     bfactor_neighborhood_correlations_MB = neighborhood_bfactor_correlation_sample(refined_pdb_path, max_radius=10, num_steps=10)

#     bfactor_neighborhood_correlation_iterative_all_emdb[emdb_pdb] = bfactor_neighborhood_correlation_iterative
#     bfactor_neighborhood_correlation_non_averaged_all_emdb[emdb_pdb] = bfactor_neighborhood_correlation_non_averaged
#     neighborhood_bfactor_correlation_emdbs_MB[emdb_pdb] = bfactor_neighborhood_correlations_MB    

#     # Plot the correlation curves for iterative refinement and save the figure
#     figpath = os.path.join(output_folder, "{}_correlation_curves_pseudomodel_refinement.png".format(emdb))
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)
#     # change font to Helvetica
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = 'Helvetica'
#     plt.rcParams['font.size'] = 12


#     radii = bfactor_neighborhood_correlation_iterative.keys()
#     correlations_iterative = [x[2][0] for x in bfactor_neighborhood_correlation_iterative.values()]
#     correlations_non_averaged = [x[2][0] for x in bfactor_neighborhood_correlation_non_averaged.values()]
#     correlations_MB = [x[2][0] for x in bfactor_neighborhood_correlations_MB.values()][:len(correlations_iterative)]


#     ax.plot(radii, correlations_iterative, "ko-", label="Iterative refinement")
#     ax.plot(radii, correlations_non_averaged, "bo-", label="Non-averaged refinement")
#     ax.plot(radii, correlations_MB, "ro-", label="Model-Based refinement")

#     ax.set_xlabel(r"Radius ($\AA$)")
#     ax.set_ylabel("Correlation")

#     ax.legend()

#     ax.set_title("EMDB: {} resolution = {}".format(emdb_pdb.split("_")[0], res_dict[emdb_pdb.split("_")[0]]))
#     fig.tight_layout()

#     fig.savefig(figpath)
#     plt.close(fig)

    
#     print("Done")


    
#     #Plot in two subplots
    
# import pickle
# import matplotlib.pyplot as plt

# # # dump the data to a pickle file in output_folder

# with open(os.path.join(output_folder, "bfactor_neighborhood_correlation_iterative_all_emdb.pickle"), "wb") as f:
#     pickle.dump(bfactor_neighborhood_correlation_iterative_all_emdb, f)

# with open(os.path.join(output_folder, "bfactor_neighborhood_correlation_non_averaged_all_emdb.pickle"), "wb") as f:
#     pickle.dump(bfactor_neighborhood_correlation_non_averaged_all_emdb, f)

# with open(os.path.join(output_folder, "bfactor_neighborhood_correlation_MB_all_emdb.pickle"), "wb") as f:
#     pickle.dump(neighborhood_bfactor_correlation_emdbs_MB, f)

# import pandas as pd
# bfactor_neighborhood_correlation_iterative_all_emdb = pd.read_pickle(os.path.join(output_folder, "bfactor_neighborhood_correlation_iterative_all_emdb.pickle"))
# bfactor_neighborhood_correlation_non_averaged_all_emdb = pd.read_pickle(os.path.join(output_folder, "bfactor_neighborhood_correlation_non_averaged_all_emdb.pickle"))
# neighborhood_bfactor_correlation_emdbs_MB = pd.read_pickle(os.path.join(output_folder, "bfactor_neighborhood_correlation_MB_all_emdb.pickle"))

# # Compute the dice coefficient between the iterative refinement, new refinement and new refinement optimised with the MB refinement as the reference

# from scipy.stats import pearsonr

# correlation_iterative_all_emdb = {}
# correlation_non_averaged_all_emdb = {}

# correlation_curve_iterative_all_emdb = {}
# correlation_curve_non_averaged_all_emdb = {}
# correlation_curves_MB_all_emdb = {}
# for emdb_pdb in bfactor_neighborhood_correlation_iterative_all_emdb.keys():
#     print(emdb_pdb)
#     test_curve_iterative = bfactor_neighborhood_correlation_iterative_all_emdb[emdb_pdb]
#     test_curve_non_averaged = bfactor_neighborhood_correlation_non_averaged_all_emdb[emdb_pdb]

#     emdb, pdb = emdb_pdb.split("_")
#     if emdb_pdb not in neighborhood_bfactor_correlation_emdbs_MB.keys():
#         print(f"skipping {emdb_pdb}")
#         continue
#     reference_curve = neighborhood_bfactor_correlation_emdbs_MB[emdb_pdb]

#     try:
#         r_value_iterative, p_value_iterative = pearsonr([x[2][0] for x in test_curve_iterative.values()], [x[2][0] for x in reference_curve.values()][:10])
#         r_value_non_averaged, p_value_new = pearsonr([x[2][0] for x in test_curve_non_averaged.values()], [x[2][0] for x in reference_curve.values()][:10])
#     except:
#         print(f"nans discarded for {emdb_pdb}")
#         continue

#     correlation_iterative_all_emdb[emdb_pdb] = r_value_iterative
#     correlation_non_averaged_all_emdb[emdb_pdb] = r_value_non_averaged
    
#     correlation_curve_iterative_all_emdb[emdb_pdb] = [x[2][0] for x in test_curve_iterative.values()]
#     correlation_curve_non_averaged_all_emdb[emdb_pdb] = [x[2][0] for x in test_curve_non_averaged.values()]
#     correlation_curves_MB_all_emdb[emdb_pdb] = [x[2][0] for x in reference_curve.values()][:10]

# print("Length of correlation_iterative_all_emdb: ", len(correlation_iterative_all_emdb))
# print("Length of correlation_non_averaged_all_emdb: ", len(correlation_non_averaged_all_emdb))
# print("Length of correlation_curves_MB_all_emdb: ", len(correlation_curves_MB_all_emdb)) 

# fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=600)

# # Plot the correlation curves for iterative refinement and save the figure
# figpath = os.path.join(output_folder, "correlation_coefficients_pseudomodel_refinement_with_MB.eps")

# correlation_values_iterative = [x for x in list(correlation_iterative_all_emdb.values()) if not np.isnan(x)]
# correlation_values_non_averaged = [x for x in list(correlation_non_averaged_all_emdb.values()) if not np.isnan(x)]

# ax.violinplot([correlation_values_non_averaged, correlation_values_iterative], showmeans=False, showmedians=True)
# ax.set_xticks([1, 2])
# ax.set_xticklabels(["Unrestrained refinement", "Restrained refinement"], rotation=45, ha="right")
# ax.set_ylabel("Pearson correlation coefficient")

# fig.tight_layout()
# fig.savefig(figpath)

# # Dump JSON file with the correlation coefficients
# import json
# correlation_data = {
#     "unrestrained_pseudomodel_similarity_values": correlation_non_averaged_all_emdb,
#     "restrained_pseudomodel_similarity_values": correlation_iterative_all_emdb,
# }

# with open(os.path.join(output_folder, "similarity_of_correlation_curves_with_MB_all_emdb.json"), "w") as f:
#     json.dump(jsonify_dictionary(correlation_data), f, indent=4)

# print("DONE WITH EXTRACTING CURVES")

# def pretty_lineplot_XY_multiple_with_shade(xdata_list, list_of_ydata_list, xlabel, ylabel, figsize_cm=(14,8),fontsize=10, \
#                         marker="o", markersize=3,fontscale=1,font="Helvetica", \
#                         linewidth=1,legends=None, title=None):
#     import matplotlib.pyplot as plt
#     from matplotlib.pyplot import cm
#     from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
#     import seaborn as sns
#     import matplotlib 
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42
#     # set the global font size for the plot

        
#     plt.rcParams.update({'font.size': fontsize})
#     figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
#     fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
#     sns.set_theme(context="paper", font=font, font_scale=fontscale)
#     # Set font size for all text in the figure
#     sns.set_style("white")
#     colors_list = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
#     for i,ydata_list in enumerate(list_of_ydata_list):
#         # Plot the data by shading the mean and standard deviation for each data set
#         xdata_array = np.array(xdata_list)
#         ydata_array = np.array(ydata_list)

#         ydata_mean = np.mean(ydata_array, axis=0)
#         ydata_std = np.std(ydata_array, axis=0)

#         ydata_max = ydata_mean + ydata_std
#         ydata_min = ydata_mean - ydata_std

#         print(xdata_array.min(), xdata_array.max())
#         print(ydata_mean.min(), ydata_mean.max())
#         print(ydata_min.min(), ydata_min.max())
#         print(ydata_max.min(), ydata_max.max())

#         # Plot the mean and standard deviation as a shaded region
#         ax.fill_between(xdata_array, ydata_min, ydata_max, alpha=0.2, color=colors_list[i])
#         ax.plot(xdata_array, ydata_mean, linewidth=linewidth, color=colors_list[i], label=colors_list[i])

#         # Add a text stating the number of data sets
#         ax.text(0.75, 0.95, f"N = {ydata_array.shape[0]}", ha="left", va="top", transform=ax.transAxes)
        
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel, rotation=90, ha="center")
    
#     plt.tight_layout()
#     plt.ylim(0.2,1.2)
#     plt.yticks([0.5,1.0])

#     if title is not None:
#         plt.title(title)
#     return fig

    
# xdata = list(range(1,11))
# ydata_MB = list(correlation_curves_MB_all_emdb.values())
# ydata_iterative = list(correlation_curve_iterative_all_emdb.values())
# ydata_non_averaged = list(correlation_curve_non_averaged_all_emdb.values())

# fig_MB_iterative = pretty_lineplot_XY_multiple_with_shade(
#     xdata, [ydata_MB, ydata_iterative], 
#      xlabel=r"Neighborhood Radius ($\AA$)", ylabel="ADP Correlation", figsize_cm=(8,8),fontsize=8)


# fig_MB_unrestrained = pretty_lineplot_XY_multiple_with_shade(
#     xdata, [ydata_MB, ydata_non_averaged],
#     xlabel=r"Neighborhood Radius ($\AA$)", ylabel="ADP Correlation", figsize_cm=(8,8),fontsize=8)

# fig_MB_iterative.savefig(os.path.join(output_folder, "correlation_curves_MB_iterative_refinement.eps"))
# fig_MB_unrestrained.savefig(os.path.join(output_folder, "correlation_curves_MB_unrestrained_refinement.eps"))

    



