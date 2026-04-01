
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from locscale.include.emmer.ndimage.profile_tools import frequency_array, crop_profile_between_frequency, estimate_bfactor_standard
from locscale.include.emmer.pdb.pdb_tools import find_wilson_cutoff
import os
import pwlf

def verify_profile(profile):
    '''
    Processing takes place by: taking only wilson region of the profile. Ignore the amplitudes upto 10A

    Parameters
    ----------
    freq : TYPE
        DESCRIPTION.
    profile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
 
    freq, amplitude = profile
    if np.isnan(amplitude).any():
        
        return False
    
    if np.any(amplitude<=0):
        
        return False
    
    if len(amplitude) != 256:
        return False
    
    wilson_cutoff = find_wilson_cutoff(num_atoms=amplitude[0])

    bfactor = estimate_bfactor_standard(freq, amplitude=amplitude, wilson_cutoff=wilson_cutoff, fsc_cutoff=1, standard_notation=True)
    
    if bfactor > 10:
        return False
    
    else:
        return True

def get_pdb_id_from_file(pdb_name):
    split_name = pdb_name.split("_")
    if split_name[0] == "pdb":
        return split_name[1]
    else:
        return split_name[0]
    
    
def clean_profile_data(input_dictionary, selected_pdbs_list):
    clean_dictionary = {}
    for pdb_name in input_dictionary.keys():
        amplitude = np.array(input_dictionary[pdb_name]['amplitude'])
        apix = input_dictionary[pdb_name]['apix']
        freq = frequency_array(amplitude,apix)
        
        if isinstance(amplitude, np.ndarray):
            amplitude_is_array = True
        else:
            amplitude_is_array = False
        
        if np.isfinite(amplitude).all():
            all_values_in_amplitude_is_finite = True
        else:
            all_values_in_amplitude_is_finite = False
        
        if np.isnan(amplitude).any():
            all_element_is_number = False
        else:
            all_element_is_number = True
        
        if verify_profile((freq, amplitude)):
            profile_verified = True
        else:
            profile_verified = False
        
        pdb_id_from_pdb_file_name = get_pdb_id_from_file(pdb_name)

        if pdb_id_from_pdb_file_name in selected_pdbs_list:
            pdb_unitcell_less_than_256 = True
        else:
            pdb_unitcell_less_than_256 = False
            
        #print(amplitude_is_array,all_values_in_amplitude_is_finite,all_element_is_number,profile_verified,pdb_unitcell_less_than_256)
        
        if amplitude_is_array and all_values_in_amplitude_is_finite and all_element_is_number and profile_verified and pdb_unitcell_less_than_256:                
            clean_dictionary[pdb_name] = amplitude
    
    return clean_dictionary

def normalise(x):
    return x/x.max()


    
def plot_radial_profile_seaborn(freq, list_of_profiles, font=16, ylims=None, crop_first=10, crop_end=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    freq = freq[crop_first:-crop_end]
    
    sns.set_theme(context="paper", font="Helvetica", font_scale=1.5)
    sns.set_style("white")
    kwargs = dict(linewidth=3)

    profile_list = np.array(list_of_profiles)
    average_profile = np.einsum("ij->j", profile_list) / len(profile_list)

    variation = []
    for col_index in range(profile_list.shape[1]):
        col_extract = profile_list[:,col_index]
        variation.append(col_extract.std())

    variation = np.array(variation)
        
    y_max = average_profile + variation
    y_min = average_profile - variation
    
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=freq, y=average_profile[crop_first:-crop_end], **kwargs)
    ax.fill_between(freq, y_min[crop_first:-crop_end], y_max[crop_first:-crop_end], alpha=0.3)
    ax.set_xlabel('Spatial Frequency $1/d [\AA^{-1}]$',fontsize=font)
    ax.set_ylabel('$\mid F \mid $',fontsize=font)
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels([round(1/x,1) for x in ax.get_xticks()])
    ax2.set_xlabel('$d [\AA]$',fontsize=font)
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()
    plt.show()
    
    return fig


    
#%%
local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
input_folder = os.path.join(local_faraday_folder,"raw_data")
output_folder = os.path.join(local_faraday_folder,"plot_output")

    

helix_profiles_pickle = os.path.join(input_folder,'secondary_structure_analysis','helix_profiles_process_0.pickle')
sheet_profiles_pickle = os.path.join(input_folder,'secondary_structure_analysis','sheet_profiles_process_1.pickle')
dna_profiles_pickle = os.path.join(input_folder,'secondary_structure_analysis','dna_profiles_process_2.pickle')
rna_profiles_pickle = os.path.join(input_folder,'secondary_structure_analysis','rna_profiles_process_3.pickle')

selected_pdb_pickle_file = os.path.join(input_folder, 'secondary_structure_analysis',"selected_pdb.pickle")

with open(selected_pdb_pickle_file, 'rb') as f:
    selected_pdb_file = pickle.load(f)
cleaned_profiles_dictionary = {}

def get_cleaned_profiles_from_pickle(pickle_file_path):
    import pickle
    selected_pdb_pickle_file = os.path.join(input_folder, 'secondary_structure_analysis',"selected_pdb.pickle")

    with open(selected_pdb_pickle_file, 'rb') as f:
        selected_pdb_file = pickle.load(f)
    with open(pickle_file_path, 'rb') as f:
        profiles_raw = pickle.load(f)
    
    profiles_cleaned = clean_profile_data(profiles_raw, selected_pdb_file)
    
    return profiles_cleaned

cleaned_profiles_dictionary['helix'] = get_cleaned_profiles_from_pickle(helix_profiles_pickle)
cleaned_profiles_dictionary['sheet'] = get_cleaned_profiles_from_pickle(sheet_profiles_pickle)
cleaned_profiles_dictionary['dna'] = get_cleaned_profiles_from_pickle(dna_profiles_pickle)
cleaned_profiles_dictionary['rna'] = get_cleaned_profiles_from_pickle(rna_profiles_pickle)




print("Cleaned input data!")



helix_profiles = list(cleaned_profiles_dictionary['helix'].values())
sheet_profiles = list(cleaned_profiles_dictionary['sheet'].values())
dna_profiles = list(cleaned_profiles_dictionary['dna'].values())
rna_profiles = list(cleaned_profiles_dictionary['rna'].values())



frequency = frequency_array(profile_size=256, apix=0.5)




#%%

def plot_list_of_list_of_radial_profile_seaborn(freq, list_of_list_of_profiles, profile_types, fontscale=3, font="Helvetica",fontsize=28, ylims=None, crop_first=10, crop_end=1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    figsize = (7,14)
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    freq = freq[crop_first:-crop_end]
    
    sns.set_theme(context="paper", font="Helvetica", font_scale=2)
    sns.set_style("white")
    
    kwargs = dict(linewidth=3, color="black")
    fig, ax = plt.subplots(len(profile_types),1)
    for i,list_of_profiles in enumerate(list_of_list_of_profiles):
        num_profiles = len(list_of_profiles)
        profile_text = profile_types[i] + " "+"(N={})".format(num_profiles)
        profile_list = np.array(list_of_profiles)
        average_profile = np.einsum("ij->j", profile_list) / len(profile_list)
    
        variation = []
        for col_index in range(profile_list.shape[1]):
            col_extract = profile_list[:,col_index]
            variation.append(col_extract.std())
    
        variation = np.array(variation)
            
        y_max = average_profile + variation
        y_min = average_profile - variation
        
        
        
        sns.lineplot(x=freq, y=average_profile[crop_first:-crop_end], ax=ax[i], **kwargs)
        
        ax[i].fill_between(freq, y_min[crop_first:-crop_end], y_max[crop_first:-crop_end], alpha=1, )
        ax[i].text(x=0.4,y=3000,s=profile_text)
        
        
        ax[i].set_xlabel('Spatial Frequency, $d^{-1} [\AA^{-1}]$')
        ax[i].get_xaxis().set_visible(True)
        
        if i == 0:    
            ax[i].set_ylabel(r'$\langle \mid F \mid \rangle $')
            ax[i].get_yaxis().set_visible(True)
        else:
            ax[i].get_yaxis().set_visible(True)
        
        if ylims is not None:
            ax[i].set_ylim(ylims)
        
        
        ax2 = ax[i].twiny()
        ax2.set_xticks(ax[i].get_xticks())
        ax2.set_xbound(ax[i].get_xbound())
        ax2.set_xticklabels([round(1/x,1) for x in ax[i].get_xticks()])
        ax2.set_xlabel('Resolution, $d [\AA]$')

        
    plt.tight_layout()
    #plt.show()
    
    return fig

plot_types = ["Helix Profiles","Sheet Profiles","DNA Profiles"]

list_of_list_of_profiles = [helix_profiles, sheet_profiles, dna_profiles]

total_fig = plot_list_of_list_of_radial_profile_seaborn(frequency, list_of_list_of_profiles, plot_types, ylims=[-500,4000], fontscale=3)

filename = os.path.join(output_folder, "Figure_5a_Helix_Sheet_DNA_profile.eps")
total_fig.savefig(filename, dpi=600, bbox_inches="tight")

#%%
plot_types2 = ["Helix Profiles","Sheet Profiles","RNA Profiles"]

list_of_list_of_profiles2 = [helix_profiles, sheet_profiles, rna_profiles]

total_fig2 = plot_list_of_list_of_radial_profile_seaborn(frequency, list_of_list_of_profiles2, plot_types2, ylims=[-500,4000], fontscale=3)

filename = os.path.join(output_folder, "Figure_S5c_Helix_Sheet_RNA_profile.eps")
total_fig2.savefig(filename, dpi=600, bbox_inches="tight")