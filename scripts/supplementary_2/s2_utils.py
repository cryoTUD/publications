def compute_fsc_cycle(cycle, halfmap1_path, halfmap2_path,refined_model_map_path):
    import os 
    import numpy as np
    from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps 
    #refined_model_cycle = refined_modelmap_per_iteration[cycle]

    assert os.path.exists(refined_model_map_path), f"Refined model map path {refined_model_map_path} does not exist"
    assert os.path.exists(halfmap1_path), f"Halfmap1 path {halfmap1_path} does not exist"
    assert os.path.exists(halfmap2_path), f"Halfmap2 path {halfmap2_path} does not exist"
    
    fsc_vals_halfmap1 = calculate_fsc_maps(refined_model_map_path, halfmap1_path)
    fsc_vals_halfmap2 = calculate_fsc_maps(refined_model_map_path, halfmap2_path)

    fsc_average_halfmap1 = (cycle, np.mean(fsc_vals_halfmap1), fsc_vals_halfmap1)
    fsc_average_halfmap2 = (cycle, np.mean(fsc_vals_halfmap2), fsc_vals_halfmap2)

    results = { 
        "cycle" : cycle,
        "halfmap1" : fsc_average_halfmap1,
        "halfmap2" : fsc_average_halfmap2
    }
    return results

def jsonify_dictionary(input_dict):
    import numpy as np
    # convert pickle object to json object
    new_dict = {}
    for key, value in input_dict.items():
        key = str(key) 
        value_is_iterable = isinstance(value, (list, tuple, np.ndarray))
        value_is_dict = isinstance(value, dict)
        value_is_float = isinstance(value, float)
        value_is_int = isinstance(value, (np.int64, int, np.int32))
        value_is_string = isinstance(value, str)
        
        print("key: {}, value_is_iterable: {}, value_is_dict: {}, value_is_float: {}, \
              value_is_int: {}, value_is_string: {}".format(key, value_is_iterable, \
                                                            value_is_dict, value_is_float, value_is_int, value_is_string))
        
        if value_is_dict:
            new_value = jsonify_dictionary(value)
        elif value_is_iterable:
            new_value = [str(x) for x in value]
        elif not value_is_string:
            new_value = str(value)
        
        new_dict[key] = new_value
        
    
    return new_dict 


def get_input_files_for_correlation_curves(emdb_pdb, dataset_type):
    import os
    
    data_folder_model_free_C = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_model_free_C1"
    data_folder_model_based_C = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/delftblue_runs_1/locscale_dataset_version_model_based_C"
    data_folder_alpha = "/tudelft/abharadwaj1/staff-bulk/tnw/BN/AJ/AB/emmernet_dataset/emmernet_dataset_hpc_runs/dataset_alpha"

    emdb, pdb = emdb_pdb.split("_")

    unrestrained_parent_dataset_folder = data_folder_alpha    
    emdb_pdb_unrestrained_dataset = os.path.join(unrestrained_parent_dataset_folder, emdb_pdb, emdb_pdb)
    processing_files_unrestrained = os.path.join(emdb_pdb_unrestrained_dataset, f"emd_{emdb}_MF_locscale_processing_files_try3")
    unrestrained_pseudomodel_refined = os.path.join(
            processing_files_unrestrained, f"shifted_emd_{emdb}_FDR_confidence_final_gradient_pseudomodel_servalcat_refined.pdb")
    
    if dataset_type == "MF":
        restrained_parent_dataset_folder = data_folder_model_free_C
        emdb_pdb_restrained_dataset = os.path.join(restrained_parent_dataset_folder, emdb_pdb)
        
        processing_files_restrained = os.path.join(emdb_pdb_restrained_dataset, f"emd_{emdb}_model_free_locscale_processing_C1")
        unsharpened_map_file = os.path.join(processing_files_restrained, "EMD_{}_unsharpened_fullmap.mrc".format(int(emdb)))
        target_pdb_path = os.path.join(
            processing_files_restrained, f"emd_{emdb}_FDR_confidence_final_gradient_pseudomodel_proper_element_composition_shifted_bfactors.pdb")
    if dataset_type == "MB":
        restrained_parent_dataset_folder = data_folder_model_based_C
        emdb_pdb_restrained_dataset = os.path.join(restrained_parent_dataset_folder, emdb_pdb)
        processing_files_restrained = os.path.join(emdb_pdb_restrained_dataset, f"emd_{emdb}_model_based_locscale_processing_C")
        unsharpened_map_file = os.path.join(processing_files_restrained, "EMD_{}_unsharpened_fullmap.mrc".format(int(emdb)))
        target_pdb_path = os.path.join(
            processing_files_restrained, f"PDB_{pdb}_unrefined_shifted_servalcat_refined.pdb")
        

    input_files_emdb = {
        "emdb_pdb": emdb_pdb,
        "unsharpened_map_file": unsharpened_map_file,
        "target_pdb_path": target_pdb_path,
        "unrestrained_pseudomodel_refined":unrestrained_pseudomodel_refined,
    }

    return input_files_emdb

def copy_files_to_folder(file, folder):
    import os
    import shutil
    # if copied file already exists then ignore 
    test_copied_path = os.path.join(folder, os.path.basename(file))
    if os.path.exists(test_copied_path):
        print("File already exists: {}".format(test_copied_path))
        return test_copied_path
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    new_path = shutil.copy(file, folder)
    return new_path
