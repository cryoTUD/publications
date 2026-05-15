import os 
import numpy as np 
import h5py

parent_modeldata_directory = "/home/abharadwaj1/dev/map_sharpening/emmernet/default_parking/locscale2_training_test"

cubedata_directory = os.path.join(parent_modeldata_directory, "cubedata_directory")
cubedata_training = os.path.join(cubedata_directory, "cubedata_training")
cubedata_validation = os.path.join(cubedata_directory, "cubedata_validation")



def create_combined_dataset(cubedata_directory, combined_dataset_filename):
    import os
    import json 
    from tqdm import tqdm
    import h5py
    emdb_ids_in_cubedata = [x for x in os.listdir(cubedata_directory) if os.path.isdir(os.path.join(cubedata_directory, x))]
    print("Number of emdb_ids in cubedata: ", len(emdb_ids_in_cubedata))
    emdb_dirs = [os.path.join(cubedata_directory, x) for x in emdb_ids_in_cubedata]
    # Create a hdf5 file to store the combined dataset
    combined_dataset_path = os.path.join(cubedata_directory, combined_dataset_filename)
   
    with h5py.File(combined_dataset_path, "w") as combined_dataset:
        all_augmentation_types = ["bfactor", "rotation", "gaussian_blur"]
        for emdb_dir in tqdm(emdb_dirs):
            emdb_id = os.path.basename(emdb_dir)
            # if str(emdb_id) not in random_sample_float and emdb_id not in random_sample_validation_float:
            #     #print(f"{emdb_id} not in random sample")
            #     continue
            # read teh XY_filenames_matched json file
            xy_filenames_json_file = os.path.join(emdb_dir, f"XY_filenames_matched_{os.path.basename(emdb_dir)}.json")
            if not os.path.isfile(xy_filenames_json_file):
                print(f"XY_filenames_matched json file does not exist for {emdb_dir}")
                continue

            xy_filenames_list = json.load(open(xy_filenames_json_file, "r"))

            for i, xy_pair in enumerate(xy_filenames_list):
                x_key, y_key = xy_pair
                x_augmentation_type = x_key.split("_")[2] 
                if x_augmentation_type == "gaussian":
                    x_augmentation_type = "gaussian_blur"
                y_augmentation_type = y_key.split("_")[2]
                x_augmentation_number = x_key.split("_")[3] if x_augmentation_type != "gaussian_blur" else 0
                y_augmentation_number = y_key.split("_")[3]
                x_center = [x for x in x_key.split("_")[5:8]]
                y_center = [x for x in y_key.split("_")[5:8]]

                x_h5_basename = "cubes_original_map.h5" if x_augmentation_type == "original" else f"cubes_{x_augmentation_type}_{x_augmentation_number}_map.h5"
                x_h5_file = os.path.join(emdb_dir, f"X_emmap_cubes_{os.path.basename(emdb_dir)}","cubes",f"{x_augmentation_type}", x_h5_basename)
                y_h5_basename = "cubes_original_map.h5" if y_augmentation_type == "original" else f"cubes_{y_augmentation_type}_{y_augmentation_number}_map.h5"
                y_h5_file = os.path.join(emdb_dir, f"Y_locscale_cubes_{os.path.basename(emdb_dir)}","cubes",f"{y_augmentation_type}", y_h5_basename)
                if not os.path.isfile(x_h5_file):
                    print(f"{x_h5_file} does not exist")
                    continue
                if not os.path.isfile(y_h5_file):
                    print(f"{y_h5_file} does not exist")
                    continue
                
                # create a group for each pair of x and y datasets
                group = combined_dataset.create_group(f"{i}_{emdb_id}")
                # create an external link to the x and y datasets in the respective h5 files
                group[f"{i}_x_{emdb_id}_{x_key}"] = h5py.ExternalLink(x_h5_file, x_key)
                group[f"{i}_y_{emdb_id}_{y_key}"] = h5py.ExternalLink(y_h5_file, y_key)
                
                
    # print the length of keys in the combined dataset
    with h5py.File(combined_dataset_path, "r") as combined_dataset:
        print("Number of keys in the combined dataset: ", len(combined_dataset.keys()))
                
    return combined_dataset_path

cubedata_dir = cubedata_training
combined_dataset_filename = "combined_training_dataset.h5"
combined_dataset_path = create_combined_dataset(cubedata_dir, combined_dataset_filename)


cubedata_dir_validation = cubedata_validation
combined_dataset_filename = "combined_validation_dataset.h5"
combined_dataset_path = create_combined_dataset(cubedata_dir_validation, combined_dataset_filename)