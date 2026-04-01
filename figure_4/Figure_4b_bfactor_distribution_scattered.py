#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:43:41 2022

@author: alok
"""
## Script to plot bfactor distribution of a map

import os
import mrcfile
from locscale.include.emmer.ndimage.map_tools import get_bfactor_distribution, get_atomic_model_mask
import numpy as np
from locscale.include.emmer.ndimage.map_tools import get_bfactor_distribution_multiple
from locscale.utils.plot_utils import plot_correlations_multiple, plot_correlations
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(14,8)})
sns.set_theme(context="paper", font="Helvetica", font_scale=4.5)
sns.set_style("white")

local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")


rmsd_magnitudes = [0,100,200,500,1000,1500,2000]

print("Preparing Figure 4(b) and Supplementary 4(c)")

modelmap_name_prefix = "pdb6y5a_modelmap_no_overlap_rmsd_{}.mrc"
list_of_emmap_path = [os.path.join(input_folder, "model_maps",modelmap_name_prefix.format(rmsd)) for rmsd in rmsd_magnitudes]

## get atomic model mask
pdb_path = os.path.join(input_folder, "pdb6y5a_additional_refined.pdb")

model_mask_path = get_atomic_model_mask(list_of_emmap_path[0], pdb_path)

fsc_resolution = 2.8

bfactors_emmap_list = get_bfactor_distribution_multiple(list_of_emmap_path, model_mask_path, fsc_resolution, num_centers=1000)
bfactor_array_list = {}
for emmap_path in list_of_emmap_path:
    emmap_name = os.path.basename(emmap_path)
    bfactor_array = np.array([x[0] for x in bfactors_emmap_list[emmap_name].values()])
    bfactor_array_list[emmap_name] = bfactor_array





#%% 
plt.figure(1)
# Output correlation of two RMSD
bfactor_arrays_tuple = [(bfactor_array_list[modelmap_name_prefix.format(0)],
                         bfactor_array_list[modelmap_name_prefix.format(rmsd)], "{} $\AA$".format(int(rmsd/100))) for rmsd in [200,2000]]
xlabel = "Wilson B-factor""\n""unperturbed model map ($\AA^{2}$)"
ylabel="Wilson B-factor""\n""perturbed model map ($\AA^{2}$)"
filename="Figure_4b_bfactor_correlation_perturbed_2_20A_wilson_new.eps"
ylims = [20, 130]
plot_correlations_multiple(bfactor_arrays_tuple, scatter=True, x_label=xlabel, y_label=ylabel, 
                           output_folder=output_folder, filename=filename, alpha=0.1, ylims = [20, 130], 
                           )


#%%
#Output correlation coefficient trend
plt.figure(2)
from scipy.stats import pearsonr
from locscale.utils.plot_utils import pretty_lineplot_XY
correlations = []
for rmsd in rmsd_magnitudes:
    correlation = pearsonr(bfactor_array_list[modelmap_name_prefix.format(0)],bfactor_array_list[modelmap_name_prefix.format(rmsd)])[0]
    correlations.append(correlation)

correlations = np.array(correlations)

filename_trend = os.path.join(output_folder, "Figure_4b_bfactor_correlation_coefficient_trend_r2_label.eps")
pretty_lineplot_XY(np.array(rmsd_magnitudes)/100, correlations, "RMSD ($\AA$)", r' $\langle R^2\rangle$ ''\n''Local b-factor correlation', 
                   linewidth=2,fontscale=3, figsize=(14,8),filename=filename_trend)



#%%
from locscale.utils.plot_utils import plot_correlations_multiple_single_plot

bfactor_arrays_tuple = [[(bfactor_array_list[modelmap_name_prefix.format(0)],
                             bfactor_array_list[modelmap_name_prefix.format(rmsd)], "{} $\AA$".format(int(rmsd/100)))] for rmsd in rmsd_magnitudes]
xlabel = "Wilson B-factor unperturbed model map ($\AA^{2}$)"
ylabel="Wilson B-factor""\n""perturbed model map ($\AA^{2}$)""\n"
filename="Figure_S4_c_bfactor_correlation_perturbed_all_combined.eps".format(int(rmsd/100))

plot_correlations_multiple_single_plot(bfactor_arrays_tuple, scatter=True, x_label=xlabel, y_label=ylabel, 
                               output_folder=output_folder, filename=filename, alpha=0.1, ylims = [20, 130],
                               fontscale=1, figsize=(8,2))


#%%
'''

## Not using this because all plots are combined into one single plot
# Output correlation of two RMSD
for rmsd in rmsd_magnitudes:
    bfactor_arrays_tuple = [(bfactor_array_list[modelmap_name_prefix.format(0)],
                             bfactor_array_list[modelmap_name_prefix.format(rmsd)], "{} $\AA$".format(int(rmsd/100))) for rmsd in [rmsd]]
    xlabel = "Wilson B-factor""\n""unperturbed model map ($\AA^{2}$)"
    ylabel="Wilson B-factor""\n""perturbed model map ($\AA^{2}$)"
    output_folder=input_folder
    filename="bfactor_correlation_perturbed_{}A_wilson_new.eps".format(int(rmsd/100))

    plot_correlations_multiple(bfactor_arrays_tuple, scatter=True, x_label=xlabel, y_label=ylabel, 
                               output_folder=output_folder, filename=filename, alpha=0.1, ylims = [20, 130],
                               fontscale=1.5, figsize=(14,8))
'''