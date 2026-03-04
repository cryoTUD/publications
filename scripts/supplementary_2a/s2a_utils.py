def filter_atomic_positions_by_mask(atomic_positions, mask_path, threshold=0.5):
    from locscale.include.emmer.ndimage.map_utils import load_map
    from locscale.include.emmer.ndimage.map_utils import convert_mrc_to_pdb_position, convert_pdb_to_mrc_position
    from locscale.include.emmer.pdb.pdb_utils import get_atomic_point_map
    import numpy as np
    from tqdm import tqdm
    # atomic positions is a numpy array of shape (N, 3)
    # mask_path is the path to the mask file
    mask, apix = load_map(mask_path)
    mask_binarised = mask > threshold
    # convert atomic positions to mrc coordinates
    atomic_positions_mrc = convert_pdb_to_mrc_position(atomic_positions, apix)
    # # get the atomic point map
    # atomic_point_map = get_atomic_point_map(atomic_positions_mrc, mask.shape)
    # # filter the atomic positions by the mask
    # atomic_positions_in_mask = atomic_point_map * mask_binarised
    # atomic_positions_in_mask_array = np.array(np.where(atomic_positions_in_mask)).T
    # # convert the atomic positions in mask to pdb coordinates
    # filtered_atomic_positions = convert_mrc_to_pdb_position(atomic_positions_in_mask_array, apix)
    # filtered_atomic_positions = np.array(filtered_atomic_positions)
    filtered_atomic_positions = []
    for i, position in tqdm(enumerate(atomic_positions)):
        mrc_pos = atomic_positions_mrc[i]
        if mask_binarised[mrc_pos[0], mrc_pos[1], mrc_pos[2]] == 1:
            filtered_atomic_positions.append(position)
    filtered_atomic_positions = np.array(filtered_atomic_positions)

    #filtered_atomic_positions = atomic_positions[mask_binarised[atomic_point_map]]
    # assert the shape of the filtered atomic positions is the same as the original atomic positions
    print(f"Filtered atomic positions shape: {filtered_atomic_positions.shape}")
    print(f"Original atomic positions shape: {atomic_positions.shape}")
    
    return filtered_atomic_positions


def get_atomic_bfactor_correlation(pseudomodel_path, atomic_model_path, n_samples=200, mask_path=None):
    from locscale.include.emmer.pdb.pdb_to_map import detect_pdb_input
    from locscale.include.emmer.pdb.pdb_utils import get_coordinates
    from locscale.include.emmer.ndimage.map_utils import load_map
    import gemmi
    from tqdm import tqdm
    import numpy as np 
    #np.random.seed(42)
    window_size_A = 12.5
    pseudo_st = detect_pdb_input(pseudomodel_path)
    atomic_st = detect_pdb_input(atomic_model_path)
    
    # Neighbor Search initialize
    
    ns_pseudo = gemmi.NeighborSearch(pseudo_st[0], pseudo_st.cell, window_size_A).populate()
    ns_atomic = gemmi.NeighborSearch(atomic_st[0], atomic_st.cell, window_size_A).populate()
    
    atomic_positions = np.array(get_coordinates(atomic_st))
    if mask_path is not None:
        filtered_atomic_positions = filter_atomic_positions_by_mask(atomic_positions, mask_path)
    else:   
        filtered_atomic_positions = atomic_positions

    # sample 1000 random points from the atomic model
    random_indices_perm = np.random.permutation(filtered_atomic_positions.shape[0])
    shuffled_atomic_positions = filtered_atomic_positions[random_indices_perm]
    #random_indices = np.random.choice(np.arange(filtered_atomic_positions.shape[0]), n_samples)
    #random_atomic_positions = filtered_atomic_positions[random_indices]

    bfactor_comparison = {}
    i = 0 
    while len(bfactor_comparison) < n_samples:
        position = shuffled_atomic_positions[i]
        i += 1
        gemmi_position = gemmi.Position(position[0], position[1], position[2])
        neighbors_atomic = ns_atomic.find_atoms(gemmi_position, '\0', radius=window_size_A)
        neighbors_pseudo = ns_pseudo.find_atoms(gemmi_position, '\0', radius=window_size_A)

        atoms_atomic = [atomic_st[0][x.chain_idx][x.residue_idx][x.atom_idx] for x in neighbors_atomic]
        atoms_pseudo = [pseudo_st[0][x.chain_idx][x.residue_idx][x.atom_idx] for x in neighbors_pseudo]

        atomic_bfactor_list = np.array([x.b_iso for x in atoms_atomic])
        pseudo_bfactor_list = np.array([x.b_iso for x in atoms_pseudo])

        average_atomic_bfactor = atomic_bfactor_list.mean()
        average_pseudo_bfactor = pseudo_bfactor_list.mean()
        if np.isnan(average_atomic_bfactor) or np.isnan(average_pseudo_bfactor):
            continue
        bfactor_comparison[tuple(position)] = (average_atomic_bfactor, average_pseudo_bfactor)
    return bfactor_comparison

def check_if_bfactors_proper(bfactors_array):
    import numpy as np
    if np.isnan(bfactors_array).any():
        return False
    
    if np.isinf(bfactors_array).any():
        return False
    
    bfactor_range = np.max(bfactors_array) - np.min(bfactors_array)
    if bfactor_range < 10:
        return False

    return True

def return_model_map_path(processing_files_folder):
    import os 
    if not os.path.exists(processing_files_folder):
        print(f"Processing files folder {processing_files_folder} does not exist.")
        return None
    list_of_files = [os.path.join(processing_files_folder, f) for f in os.listdir(processing_files_folder) if "4locscale" in f]
    # check if symmetry present
    if len(list_of_files) == 2:
        file_containing_symmetry = [f for f in list_of_files if "symmetry.mrc" in f][0]
    elif len(list_of_files) == 1:
        file_containing_symmetry = list_of_files[0]
    else:
        print(f"No model map found in {processing_files_folder}")
        return None
    
    return file_containing_symmetry

def get_input_files_for_adp_correlation_atomicmodel_pseudomodel(emdb_pdb, dataset_type, averagedPseudomodel=True):
    import os
    
    data_folder_model_free_C = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_model_free_C1"
    data_folder_model_based_C = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_model_based_C"
    data_folder_hybrid_C = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_hybrid_C"
    data_folder_refinement = "/home/abharadwaj1/papers/elife_paper/figure_information/archive_data/structured_data/supplementary_2a/bfactor_refinement_all_using_halfmaps"
    data_folder_alpha = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/emmernet_dataset_hpc_runs/dataset_alpha"

    emdb, pdb = emdb_pdb.split("_")

    unrestrained_parent_dataset_folder = data_folder_alpha    
    emdb_pdb_unrestrained_dataset = os.path.join(unrestrained_parent_dataset_folder, emdb_pdb, emdb_pdb)
    processing_files_unrestrained = os.path.join(emdb_pdb_unrestrained_dataset, f"emd_{emdb}_MF_locscale_processing_files_try3")
    unrestrained_pseudomodel_refined = os.path.join(
            processing_files_unrestrained, f"shifted_emd_{emdb}_FDR_confidence_final_gradient_pseudomodel_servalcat_refined.pdb")
    
    emdb_pdb_version_C_dataset = os.path.join(data_folder_model_based_C, emdb_pdb)
    halfmap_1_path = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_half_map_1.map")
    halfmap_2_path = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_half_map_2.map")
    mask_path = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_FDR_confidence_final.map")
    if dataset_type == "MF":
        emdb_pdb_version_C_dataset = os.path.join(data_folder_model_free_C, emdb_pdb)
        emdb_pdb_version_C_processing_files = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_model_free_locscale_processing_C1")
        restrained_parent_dataset_folder = data_folder_refinement
        emdb_pdb_restrained_dataset = os.path.join(restrained_parent_dataset_folder, emdb_pdb)
        processing_files_restrained = os.path.join(emdb_pdb_restrained_dataset, "model_free")
        unsharpened_map_file = os.path.join(emdb_pdb_version_C_processing_files, "EMD_{}_unsharpened_fullmap.mrc".format(int(emdb)))
        model_map_path = return_model_map_path(emdb_pdb_version_C_processing_files)
        pseudomodel_suffix = "_averaged" if averagedPseudomodel else ""
        target_pdb_path = os.path.join(
            processing_files_restrained, f"emd_{emdb}_FDR_confidence_final_gradient_pseudomodel_proper_element_composition{pseudomodel_suffix}.cif")
        
    if dataset_type == "MB":
        emdb_pdb_version_C_dataset = os.path.join(data_folder_model_based_C, emdb_pdb)
        emdb_pdb_version_C_processing_files = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_model_based_locscale_processing_C")
        restrained_parent_dataset_folder = data_folder_refinement
        emdb_pdb_restrained_dataset = os.path.join(restrained_parent_dataset_folder, emdb_pdb)
        processing_files_restrained = os.path.join(emdb_pdb_restrained_dataset, "model_based")
        unsharpened_map_file = os.path.join(emdb_pdb_version_C_processing_files, "EMD_{}_unsharpened_fullmap.mrc".format(int(emdb)))
        
        target_pdb_path = os.path.join(
            processing_files_restrained, f"PDB_{pdb}_unrefined_shifted_servalcat_refined_servalcat_refined.mmcif")
        model_map_path = return_model_map_path(emdb_pdb_version_C_processing_files)
        
    if dataset_type == "hybrid":
        emdb_pdb_version_C_dataset = os.path.join(data_folder_hybrid_C, emdb_pdb)
        emdb_pdb_version_C_processing_files = os.path.join(emdb_pdb_version_C_dataset, f"emd_{emdb}_hybrid_locscale_processing_C")
        model_map_path = return_model_map_path(emdb_pdb_version_C_processing_files)
        unsharpened_map_file = os.path.join(emdb_pdb_version_C_processing_files, "EMD_{}_unsharpened_fullmap.mrc".format(int(emdb)))
        target_pdb_path = os.path.join(
            emdb_pdb_version_C_processing_files, f"PDB_{pdb}_unrefined_shifted_servalcat_refined_shifted_integrated_pseudoatoms_proper_element_composition.cif")
       
    input_files_emdb = {
        "emdb_pdb": emdb_pdb,
        "unsharpened_map_file": unsharpened_map_file,
        "target_pdb_path": target_pdb_path,
        "unrestrained_pseudomodel_refined":unrestrained_pseudomodel_refined,
        "halfmap_1_path": halfmap_1_path,
        "halfmap_2_path": halfmap_2_path,
        "mask_path": mask_path,
        "model_map_path": model_map_path,
    }

    return input_files_emdb


def statistic(x, y):
    from scipy.stats import kstest
    return kstest(x, y).statistic

def invgamma_permutation_test(x, invgamma_params):
    from scipy.stats import permutation_test
    from scipy.stats import invgamma
    rvs_sample = invgamma.rvs(*invgamma_params, size=1000)
    perm_test = permutation_test((x,rvs_sample), statistic=statistic, n_resamples=10000)
    return perm_test


def plot_invgamma_fit(emdb_pdb, bfactor_list_pseudo, bfactor_list_atomic):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import invgamma

    pseudo_bfactors, atomic_bfactors = bfactor_list_pseudo[emdb_pdb], bfactor_list_atomic[emdb_pdb]
    pseudo_bfactors = np.array(pseudo_bfactors)
    atomic_bfactors = np.array(atomic_bfactors)
    
    invgamma_fit_pseudo = invgamma.fit(pseudo_bfactors)
    invgamma_fit_atomic = invgamma.fit(atomic_bfactors)

    x_pseudo = np.linspace(pseudo_bfactors.min(), pseudo_bfactors.max(), 1000)
    x_atomic = np.linspace(atomic_bfactors.min(), atomic_bfactors.max(), 1000)
    y_pseudo = invgamma.pdf(x_pseudo, *invgamma_fit_pseudo)
    y_atomic = invgamma.pdf(x_atomic, *invgamma_fit_atomic)

    fig, ax = plt.subplots(1, 2, figsize=(6, 2))
    #sns.set_theme(context="paper", font="Helvetica", font_scale=1)
    # Set font size for all text in the figure
    #sns.set_style("white")
    sns.histplot(pseudo_bfactors, label="Pseudo-atomic", ax=ax[0], stat="density", bins=30)
    sns.histplot(atomic_bfactors, label="Atomic", ax=ax[1], stat="density", bins=30)
    ax[0].set_xlabel("B-factor")
    ax[0].set_ylabel("Frequency")
    ax[0].plot(x_pseudo, y_pseudo, label="fit", color="red")
    ax[1].plot(x_atomic, y_atomic, label="fit", color="red")

    ax[1].set_xlabel("B-factor")
    ax[1].set_ylabel("Frequency")
    
    
    fig.suptitle(f"EMDB-PDB: {emdb_pdb}")

    plt.tight_layout()
    
    perm_test_pseudo = invgamma_permutation_test(pseudo_bfactors, invgamma_fit_pseudo)
    perm_test_atomic = invgamma_permutation_test(atomic_bfactors, invgamma_fit_atomic)
    
    D_pseudo = perm_test_pseudo.statistic
    D_atomic = perm_test_atomic.statistic
    p_value_pseudo = perm_test_pseudo.pvalue
    p_value_atomic = perm_test_atomic.pvalue
    
    pseudo_text = f"$D$ = {D_pseudo:.2f} \n $p$ = {p_value_pseudo:.2f}"
    atomic_text = f"$D$ = {D_atomic:.2f} \n $p$ = {p_value_atomic:.2f}"
    
    ax[0].text(.6, .7, pseudo_text,transform=ax[0].transAxes)
    ax[1].text(.6, .7, atomic_text,transform=ax[1].transAxes)
    
    
    return fig

from locscale.include.emmer.ndimage.profile_tools import frequency_array 

def compute_fsc_curve_emdb_pdb(input_files_mf, input_files_hyb, input_files_mb, emdb_pdb):
    import os
    input_files_combined = {
        "unsharpened_map_file": input_files_mf["unsharpened_map_file"],
        "mask_path": input_files_mf["mask_path"],
        "pseudomodel_modmap_path": input_files_mf["model_map_path"],
        "hybrid_modmap_path": input_files_hyb["model_map_path"],
        "atomic_modmap_path": input_files_mb["model_map_path"],
        "emdb_pdb": emdb_pdb
    }

    # assert all required files exist
    required_files = ["unsharpened_map_file", "mask_path", "pseudomodel_modmap_path", "hybrid_modmap_path", "atomic_modmap_path"]
    for required_file in required_files:
        if not os.path.exists(input_files_combined[required_file]):
            print(f"Required file {required_file} does not exist: {input_files_combined[required_file]}")
            return None

    #return compute_fsc_curve_for_data(input_files_combined)
    return None 
    
def compute_fsc_curve_for_data(emdb_pdb_input_files):
    import numpy as np
    from locscale.include.emmer.ndimage.map_utils import load_map
    from locscale.include.emmer.ndimage.profile_tools import frequency_array
    from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
    
    emdb_pdb = emdb_pdb_input_files["emdb_pdb"]
    unsharpened_map_path = emdb_pdb_input_files["unsharpened_map_file"]
    confidence_mask_path = emdb_pdb_input_files["mask_path"]
    pseudomodel_modmap_path = emdb_pdb_input_files["pseudomodel_modmap_path"]
    hybrid_modmap_path = emdb_pdb_input_files["hybrid_modmap_path"]
    atomic_modmap_path = emdb_pdb_input_files["atomic_modmap_path"]
    

    unsharpened_map, apix = load_map(unsharpened_map_path)
    pseudomodel_modmap, apix = load_map(pseudomodel_modmap_path)
    hybrid_modmap, apix = load_map(hybrid_modmap_path)
    atomic_modmap, apix = load_map(atomic_modmap_path)
    
    # compute fsc curves
    fsc_unsharpened_pseudomodel = calculate_fsc_maps(unsharpened_map, pseudomodel_modmap)
    fsc_unsharpened_hybrid = calculate_fsc_maps(unsharpened_map, hybrid_modmap)
    fsc_unsharpened_atomic = calculate_fsc_maps(unsharpened_map, atomic_modmap)
    
    freq = frequency_array(fsc_unsharpened_atomic, apix)
    
    wilson_cutoff_ang = 10 
    wilson_cutoff = 1 / (wilson_cutoff_ang) 
    
    wilson_index = np.where(freq > wilson_cutoff)[0][0]
    
    fsc_average_guinier_atomic = np.mean(fsc_unsharpened_atomic[:wilson_index])
    fsc_average_guinier_hybrid = np.mean(fsc_unsharpened_hybrid[:wilson_index])
    fsc_average_guinier_pseudomodel = np.mean(fsc_unsharpened_pseudomodel[:wilson_index])
    
    fsc_average_wilson_atomic = np.mean(fsc_unsharpened_atomic[wilson_index:])
    fsc_average_wilson_hybrid = np.mean(fsc_unsharpened_hybrid[wilson_index:])
    fsc_average_wilson_pseudomodel = np.mean(fsc_unsharpened_pseudomodel[wilson_index:])
    
    return_values = {
        "fsc_average_guinier_atomic": fsc_average_guinier_atomic,
        "fsc_average_guinier_hybrid": fsc_average_guinier_hybrid,
        "fsc_average_guinier_pseudomodel": fsc_average_guinier_pseudomodel,
        "fsc_average_wilson_atomic": fsc_average_wilson_atomic,
        "fsc_average_wilson_hybrid": fsc_average_wilson_hybrid,
        "fsc_average_wilson_pseudomodel": fsc_average_wilson_pseudomodel,
        "fsc_unsharpened_atomic": fsc_unsharpened_atomic,
        "fsc_unsharpened_hybrid": fsc_unsharpened_hybrid,
        "fsc_unsharpened_pseudomodel": fsc_unsharpened_pseudomodel,
        "freq": freq,
        "apix": apix,
        "emdb_pdb": emdb_pdb
    }
    
    return return_values

