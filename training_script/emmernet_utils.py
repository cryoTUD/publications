import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py
from scipy.ndimage import gaussian_filter


def try_to_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error executing function {func.__name__}:")
            print(f"Args: {args}")
            print(f"Kwargs: {kwargs}")
            print(f"Error: {e}")
            return 420
    return wrapper

def load_smoothened_mask(mask_path):
    from locscale.include.emmer.ndimage.map_utils import load_map 
    from locscale.include.emmer.ndimage.filter import get_cosine_mask 
    
    mask, apix = load_map(mask_path)
    mask_binarize = (mask >= 0.99).astype(np.int_)
    mask_smooth = get_cosine_mask(mask_binarize, 3)
    mask_binarize = (mask_smooth >= 0.5).astype(np.int_)
    
    return mask_binarize, apix

def extract_all_cube_centers(im_input, step_size, cube_size):
    '''
    Utility function to extract all cube centers from a 3D density map in a rolling window fashion
    
    '''
    im_shape = im_input.shape[0]
    length, width, height = im_input.shape

    # extract centers of all cubes in the 3D map based on the step size
    cubecenters = []
    for i in range(0, length, step_size):
        for j in range(0, width, step_size):
            for k in range(0, height, step_size):
                # i,j,k are corner of the cube 
                # we need to find the center of the cube
                center_k = k + cube_size//2
                center_j = j + cube_size//2
                center_i = i + cube_size//2

                # check if the center is within the map
                if center_k < length and center_j < width and center_i < height:
                    center_within_map = True
                else:
                    center_within_map = False
                
                # check if bounding box is within the map
                if k + cube_size < length and j + cube_size < width and i + cube_size < height:
                    bounding_box_within_map = True
                else:
                    bounding_box_within_map = False
                
                if center_within_map and bounding_box_within_map:
                    cubecenters.append((center_i, center_j, center_k))
                
                if center_within_map and not bounding_box_within_map:
                    # Check which dimension is out of bounds
                    if k + cube_size >= length:
                        diff  = k + cube_size - length
                        center_k = center_k - diff
                    if j + cube_size >= width:
                        diff  = j + cube_size - width
                        center_j = center_j - diff
                    if i + cube_size >= height:
                        diff  = i + cube_size - height
                        center_i = center_i - diff
                    cubecenters.append((center_i, center_j, center_k))
    
    return cubecenters

def filter_cubecenters_by_mask(cubecenters, mask, cube_size, signal_to_noise_cubes, max_num_cubes=None):
    '''
    Utility function to filter cube centers by a mask

    '''
    from locscale.include.emmer.ndimage.map_utils import extract_window
    import random

    random.seed(42)

    signal_cubes_centers = []
    noise_cubes_centers = []
    for center in cubecenters:
        cube = extract_window(mask, center=center, size=cube_size)
        if cube.sum() > 5:
            signal_cubes_centers.append(center)
        else:
            noise_cubes_centers.append(center)

    num_signal_cubes = len(signal_cubes_centers)
    num_noise_cubes = len(noise_cubes_centers)

    required_noise_cubes = int(num_signal_cubes / signal_to_noise_cubes)
    if num_noise_cubes < required_noise_cubes:
        sampled_noise_cubes = noise_cubes_centers
    else:
        sampled_noise_cubes = random.sample(noise_cubes_centers, required_noise_cubes)
    
    filtered_cubecenters = signal_cubes_centers + sampled_noise_cubes

    if max_num_cubes is not None:
        filtered_cubecenters = random.sample(filtered_cubecenters, max_num_cubes)
    return filtered_cubecenters, signal_cubes_centers, sampled_noise_cubes


def extract_cubes_from_cubecenters_h5py(emmap_path, cubecenters, signal_cubes, cube_size, output_dir, filename_id, save_assembly_test=False):
    from locscale.include.emmer.ndimage.map_utils import extract_window, load_map, save_as_mrc
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"

    # Extract EMDB information from the emmap_path
    basename = os.path.basename(emmap_path)
    emdb_id = basename.split("_")[1]
    emmap, apix = load_map(emmap_path)

    cubes = {}
    all_filenames = {}
    h5py_file = h5py.File(os.path.join(output_dir, f"cubes_{filename_id}_{emdb_id}.h5"), 'w')
    for i,center in enumerate(cubecenters):
        cube = extract_window(emmap, center=center, size=cube_size)
        cube = np.expand_dims(cube, axis=3)
        # save cube
        is_signal = center in signal_cubes
        cube_type = "signal_cube" if is_signal else "noise_cube"
        cube_dataset_name = f"{cube_type}_{filename_id}_{i}_{emdb_id}_{center[0]}_{center[1]}_{center[2]}"

        h5py_file.create_dataset(cube_dataset_name, data=cube)
        cubes[i] = {'cube': cube, 'center': center, 'filename': cube_dataset_name}
        all_filenames[tuple(center)] = cube_dataset_name
    
    h5py_file.close()
    
    if save_assembly_test:    
        cubes_assembled = assemble_cubes(cubes_dictionary=cubes, im_shape=emmap.shape, cube_size=cube_size, average=True, draw_grid=True)
        project_assembled = plot_projections(cubes_assembled, return_figure=True)
        project_volume = plot_projections(emmap, return_figure=True)
        savepath = os.path.join(output_dir, "cubes_assembled_Y.png")
        project_assembled.savefig(savepath)
        savepath = os.path.join(output_dir, "preprocessed_volume.png")
        project_volume.savefig(savepath)
        save_as_mrc(cubes_assembled, os.path.join(output_dir, f"cubes_assembled_{emdb_id}.mrc"), apix=1)
        

    return all_filenames

def extract_cubes_for_augmented_map(augmented_maps_dict, cubecenters, rotated_cubecenters, signal_cubes, cube_size):
    '''
    Utility function to extract all cubes from a 3D density map in a rolling window fashion
    
    '''
    from locscale.include.emmer.ndimage.map_utils import extract_window

    parent_directory = os.path.dirname(augmented_maps_dict["original"])
    cubes_dir = os.path.join(parent_directory, "cubes")
    if not os.path.exists(cubes_dir):
        os.mkdir(cubes_dir)
    cube_filenames_extracted = {}
    for augmentation_type in augmented_maps_dict.keys():
        output_dir = os.path.join(cubes_dir, augmentation_type)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        cube_filenames_extracted[augmentation_type] = {}
        if augmentation_type in ["bfactor","gaussian_blur"]:
            
            for aug_id in augmented_maps_dict[augmentation_type].keys():
                filename_id = f"{augmentation_type}_{aug_id}"
                aug_map_path = augmented_maps_dict[augmentation_type][aug_id]
                
                
                filenames = extract_cubes_from_cubecenters_h5py(\
                                            aug_map_path, cubecenters, signal_cubes, cube_size, output_dir, filename_id)
                
                cube_filenames_extracted[augmentation_type][aug_id] = filenames
                
        elif augmentation_type in ["rotation"]:
            save_assembly_test = 1
            for aug_id in augmented_maps_dict[augmentation_type].keys():
                filename_id = f"{augmentation_type}_{aug_id}"
                aug_map_path = augmented_maps_dict[augmentation_type][aug_id]
                
                rotated_cubecenter_id = rotated_cubecenters[str(aug_id)]["cubecenters"]
                signal_cubes_id = rotated_cubecenters[str(aug_id)]["signal_cubes"]
                
                
                filenames = extract_cubes_from_cubecenters_h5py(\
                                        aug_map_path, rotated_cubecenter_id, signal_cubes_id, cube_size, output_dir, filename_id, save_assembly_test=bool(save_assembly_test))
                
                cube_filenames_extracted[augmentation_type][aug_id] = filenames
                save_assembly_test = save_assembly_test * 0

        else: 
            assert augmentation_type == "original"
            filename_id = f"{augmentation_type}"
            aug_map_path = augmented_maps_dict[augmentation_type]
            
            filenames = extract_cubes_from_cubecenters_h5py(\
                                        aug_map_path, cubecenters, signal_cubes, cube_size, output_dir, filename_id, save_assembly_test=True)
            
            cube_filenames_extracted[augmentation_type] = {"0":filenames}
            
    cube_information_filename = os.path.join(cubes_dir, "cube_information.json")
    with open(cube_information_filename, 'w') as f:
        json.dump(jsonify_dictionary(cube_filenames_extracted), f, indent=4)        
    return cube_filenames_extracted


def assemble_cubes(cubes_dictionary, im_shape, cube_size, average=True, draw_grid=False, mask=None):
    '''
    Utility function to assemble cubes into a 3D density map
    
    '''
    from locscale.include.emmer.ndimage.map_utils import extract_window
    if isinstance(im_shape, int):
        imshape = (im_shape, im_shape, im_shape)
    else:
        imshape = im_shape
    
    im = np.zeros(imshape)
    average_map = np.zeros(imshape)
    for cubes in cubes_dictionary.values():
        center_ijk = cubes['center']
        ci, cj, ck = center_ijk
        if mask is not None:
            ni, nj, nk = (cube_size,cube_size,cube_size)
            cube = mask[ci-ni//2:ci+ni//2, cj-nj//2:cj+nj//2, ck-nk//2:ck+nk//2]
        else:
            cube = cubes['cube'].reshape(cube_size,cube_size,cube_size)    
            ni, nj, nk = cube.shape

        im[ci-ni//2:ci+ni//2, cj-nj//2:cj+nj//2, ck-nk//2:ck+nk//2] += cube
        average_map[ci-ni//2:ci+ni//2, cj-nj//2:cj+nj//2, ck-nk//2:ck+nk//2] += 1
    
    if average:
        nonzero_indices = np.where(average_map != 0)
        im[nonzero_indices] /= average_map[nonzero_indices]
    
    if draw_grid:
        # draw grid lines on the image to visualize the cubes that were extracted
        grid = np.zeros(imshape)
        for cubes in cubes_dictionary.values():
            center_ijk = cubes['center']
            ci, cj, ck = center_ijk
            grid[ci, cj, ck] = 2
        
        im_norm = (im - im.min()) / (im.max() - im.min()) # normalize the image
        im_norm_grid = im_norm + grid # add the grid to the image
        im = im_norm_grid


    
    return im


## Augment the maps 

def bfactor_augment_map(emmap_path, num_augment, fsc_resolution, wilson_cutoff=10):
    '''
    Utility function to augment a map with B-factor noise
    
    '''
    from locscale.include.emmer.ndimage.map_tools import sharpen_maps, estimate_global_bfactor_map
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
    from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile, frequency_array, estimate_bfactor_standard

    emmap, apix = load_map(emmap_path)
    rp_emmap = compute_radial_profile(emmap)
    freq = frequency_array(rp_emmap, apix)

    bfactor = estimate_bfactor_standard(freq, rp_emmap, wilson_cutoff=wilson_cutoff, fsc_cutoff=fsc_resolution, standard_notation=True)
    new_bfactors = np.random.uniform(low=0, high=400, size=num_augment)
    augmented_maps = {}
    for i in range(num_augment):
        new_bfactor = new_bfactors[i]
        bfactor_diff = new_bfactor - bfactor  # if bfactor_diff is negative, the map will be sharpened else blurred
        sharpened_map = sharpen_maps(emmap, apix, global_bfactor=bfactor_diff) 
        augmented_maps[i] = {
            'map': sharpened_map, 
            'setup': {'bfactor': int(round(new_bfactor))}
        }
    
    return augmented_maps

def rotation_augment_map(emmap_path, num_augments=4, setup=None):
    '''
    Utility function to augment a map with rotation noise
    
    '''
    from scipy.ndimage import rotate
    from scipy.ndimage import shift
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc

    rotation_augmented_maps = {}
    emmap, apix = load_map(emmap_path)
    if setup is not None:
        for i in setup.keys():
            angle = int(setup[i]['angle'])
            axis_of_rotation = setup[i]['rotaxis'][2] if len(setup[i]['rotaxis']) > 1 else setup[i]['rotaxis']
            axes = (1,2) if axis_of_rotation == "x" else (0,2) if axis_of_rotation == "y" else (0,1)
            rotated_map = rotate(emmap, angle, axes=axes, reshape=False)
            # shift the map to add translation noise
            translation_setup_string = setup[i]['translate']
            translation_setup = np.array([int(x) for x in translation_setup_string.strip('[]').split()])
            translated_map = shift(rotated_map, translation_setup, order=0)
            
            rotation_augmented_maps[i] = {
                'map': translated_map, 
                'setup': {'angle': angle, 'rotaxis': axis_of_rotation, 'translate': translation_setup}
            }
    
            

    else:
        for i in range(num_augments):
            angle = np.random.randint(low=0, high=360, size=1)[0]
            axis_of_rotation = np.random.choice(['x', 'y', 'z'], size=1, replace=False)
            if axis_of_rotation == 'x':
                rotated_map = rotate(emmap, angle, axes=(1,2), reshape=False)
            elif axis_of_rotation == 'y':
                rotated_map = rotate(emmap, angle, axes=(0,2), reshape=False)
            elif axis_of_rotation == 'z':
                rotated_map = rotate(emmap, angle, axes=(0,1), reshape=False)
            
            # shift the map to add translation noise
            translation_setup = np.random.randint(low=-10, high=10, size=3)
            translated_map = shift(rotated_map, translation_setup, order=0)

            rotation_augmented_maps[i] = {
                'map': translated_map, 
                'setup': {'angle': angle, 'rotaxis': axis_of_rotation, 'translate': translation_setup}
            }
    
    return rotation_augmented_maps

def gaussian_blur_augment_map(emmap_path, num_augments=1, setup=None):
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc

    gaussian_blur_augmented_maps = {}
    emmap, apix = load_map(emmap_path)
    
    if setup is not None: 
        for i in setup.keys():
            sigma = setup[i]['sigma']
            blurred_map = gaussian_filter(emmap, sigma=sigma)
            gaussian_blur_augmented_maps[i] = {
                'map': blurred_map,
                'setup': {'sigma': sigma}
            }
    else:
        for i in range(num_augments):
            sigma = np.random.uniform(low=5, high=20, size=1)[0]
            blurred_map = gaussian_filter(emmap, sigma=sigma)
            gaussian_blur_augmented_maps[i] = {
                'map': blurred_map, 
                'setup': {'sigma': sigma}
            }
        
    return gaussian_blur_augmented_maps

def get_rotation_setup(augmentation_log_file):
    augmentation_info = json.load(open(augmentation_log_file, 'r'))
    rotation_augmentation_info = augmentation_info['rotation']
    return rotation_augmentation_info 

def find_rotated_cubecenters(cubecenters, mask_path, augmentation_log_file, cube_size, signal_to_noise_cubes, max_num_cubes=None):
    import json 
    rotation_setup = get_rotation_setup(augmentation_log_file)
    num_augments = len(rotation_setup.keys())
    rotated_masks_dict = rotation_augment_map(mask_path, num_augments, rotation_setup)
    
    rotation_cubecenters = {}
    
    for i in rotated_masks_dict.keys():
        mask_rotated = rotated_masks_dict[i]['map']
        cubecenters_i, signal_cubes_i, _ = filter_cubecenters_by_mask(cubecenters, mask_rotated, cube_size=cube_size, \
                                                                      signal_to_noise_cubes=signal_to_noise_cubes, max_num_cubes=max_num_cubes)
        rotation_cubecenters[str(i)] = {
            "cubecenters" : cubecenters_i,
            "signal_cubes" : signal_cubes_i
        }
    
    return rotation_cubecenters
        
    
def augment_maps(emmap_path, fsc_resolution, wilson_cutoff, augmentation_log_file=None,\
                bfactor_augment=True, rotation_augment=True, gaussian_blur_augment=True):
    '''
    Utility function to augment a map with B-factor, rotation and gaussian blur noise using the above functions
    '''
    import json 
    if augmentation_log_file is not None:
        rotation_setup = get_rotation_setup(augmentation_log_file)
        bfactor_augment = False 
        gaussian_blur_augment = False 
        rotation_augment = True 
    else:
        rotation_setup = None 
    
    augmented_maps = {}
    if bfactor_augment:
        bfactor_augmentation = 2
        bfactor_augmented_maps = bfactor_augment_map(emmap_path, num_augment=bfactor_augmentation, fsc_resolution=fsc_resolution, wilson_cutoff=wilson_cutoff)
        augmented_maps["bfactor"] = bfactor_augmented_maps
    if rotation_augment:
        rotation_augmented_maps = rotation_augment_map(emmap_path, num_augments=4, setup=rotation_setup)
        augmented_maps["rotation"] = rotation_augmented_maps
    if gaussian_blur_augment:
        gaussian_blur_augmented_maps = gaussian_blur_augment_map(emmap_path, num_augments=1)
        augmented_maps["gaussian_blur"] = gaussian_blur_augmented_maps 
    
    return augmented_maps

def preprocess_the_augmented_maps(emmap_path, augmented_maps, aug_log_file_path, output_dir, map_type):
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
    
    emmap, apix = load_map(emmap_path)
    standardize = True if map_type == "X" else False
    preprocessed_emmap_map = preprocess_emmap(emmap=emmap, apix=apix, standardize=standardize)
    preprocessed_emmap_path = os.path.join(output_dir, "augmented_map_original.mrc")
    save_as_mrc(preprocessed_emmap_map, preprocessed_emmap_path, apix=apix)
    
    augmented_maps_dict = {"original" : preprocessed_emmap_path}
    augmentation_information = {"original": {"setup":"nothing_interesting"}} # 0 is the original map
    
    for augmentation_type in augmented_maps.keys():
        number_of_augmentations = len(augmented_maps[augmentation_type])
        augmented_maps_dict[augmentation_type] = {}
        augmentation_information[augmentation_type] = {}
        for aug_id in augmented_maps[augmentation_type].keys():
            aug_map = augmented_maps[augmentation_type][aug_id]['map']
            
            # Preprocess the augmented map
            aug_map_preprocessed = preprocess_emmap(emmap=aug_map, apix=apix, standardize=standardize)
            aug_map_setup = augmented_maps[augmentation_type][aug_id]['setup']

            # Convert kwargs values to string
            aug_map_setup_str = {k: str(v) for k, v in aug_map_setup.items()}
            
            # save the augmented map
            aug_map_path = os.path.join(output_dir, "augmented_map_{}_{}.mrc".format(augmentation_type, aug_id))
            save_as_mrc(aug_map_preprocessed, aug_map_path, apix=1)
            
            # Save the augmentation information
            augmentation_information[augmentation_type][aug_id] = aug_map_setup_str

            augmented_maps_dict[augmentation_type][aug_id] = aug_map_path 
    
    # Save the augmentation information to the log file
    with open(aug_log_file_path, "w") as aug_log_file:
        json.dump(augmentation_information, aug_log_file, indent=4)
    
    return augmentation_information, augmented_maps_dict
    
def create_and_extract_cubes_for_all_augmentation_per_map(emmap_path, mask_path, output_dir, fsc_resolution, map_type, \
                                            wilson_cutoff=10, step_size=16, cube_size=32, signal_to_noise_cubes=4,\
                                            augmentation_log_file=None, cubecenter_info=None, max_num_cubes=None):
    '''
    Utility function to augment a map with B-factor, rotation and gaussian blur noise using the above functions
    '''
    import json
    from tqdm import tqdm
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
    from locscale.include.emmer.ndimage.filter import get_cosine_mask

    
    emmap, apix = load_map(emmap_path)

    augmented_maps = augment_maps(emmap_path, fsc_resolution, rotation_augment=2, wilson_cutoff=wilson_cutoff, \
                                    augmentation_log_file=augmentation_log_file)
    
    ## Pre-process the maps while saving them

    assert os.path.exists(output_dir), "Output directory does not exist"
    # create a log file to store the augmentation information
    aug_log_file_path = os.path.join(output_dir, "augmentation_log.json")
    
    augmentation_information, augmented_maps_dict = preprocess_the_augmented_maps(augmented_maps=augmented_maps, emmap_path=emmap_path, \
                                                            aug_log_file_path=aug_log_file_path, output_dir=output_dir, map_type=map_type)
    
    mask_binarize, apix = load_smoothened_mask(mask_path)
    preprocessed_mask = preprocess_emmap(emmap=mask_binarize, apix=apix, standardize=False)
    
    # save the preprocessed mask 
    mask_basename = os.path.basename(mask_path)
    preprocessed_mask_path = os.path.join(output_dir, mask_basename[:-4]+"_preprocessed.mrc")
    save_as_mrc(preprocessed_mask, preprocessed_mask_path, apix=1)
    
    # if cubecenter_info is not None ,extract filtered cubes and signal cubes 
    if cubecenter_info is not None:
        filtered_cubecenters,signal_cubes,rotated_cubecenters  = extract_filtered_centers(cubecenter_info)
        
    else:
        cubecenters = extract_all_cube_centers(preprocessed_mask, step_size, cube_size)
        filtered_cubecenters, signal_cubes, _ = filter_cubecenters_by_mask(cubecenters, preprocessed_mask,\
                                                                cube_size=cube_size, signal_to_noise_cubes=signal_to_noise_cubes, max_num_cubes=max_num_cubes)
        # Find the cubecenters for the rotated augmented maps 
        rotated_cubecenters = find_rotated_cubecenters(\
                    cubecenters, preprocessed_mask_path, aug_log_file_path,cube_size=cube_size, \
                        signal_to_noise_cubes=signal_to_noise_cubes, max_num_cubes=max_num_cubes)
                    
    cube_filenames_extracted = extract_cubes_for_augmented_map(augmented_maps_dict, filtered_cubecenters, \
                                    rotated_cubecenters, signal_cubes=signal_cubes, cube_size=cube_size)
        
    return cube_filenames_extracted

@try_to_run
def chunk_and_save_emdb_pdb_cubes(emdb_id, step_size, cube_size, cubedata_dir, collection_directory, max_num_cubes=None):
    '''
    This function takes in top level information about an EMDB PDB entry and extracts all cubes from the EMDB PDB entry
    
    '''
    import pandas as pd
    import json 
    from locscale.include.emmer.ndimage.map_utils import load_map
    assert os.path.exists(cubedata_dir), f"Directory {cubedata_dir} does not exist"
    assert os.path.exists(collection_directory), f"Directory {collection_directory} does not exist"
    emdb_cubedata_dir = os.path.join(cubedata_dir, emdb_id)
    if not os.path.exists(emdb_cubedata_dir):
        os.mkdir(emdb_cubedata_dir)

    matched_filenames_path = os.path.join(emdb_cubedata_dir, f"XY_filenames_matched_{emdb_id}.json")
    if os.path.exists(matched_filenames_path):
        return matched_filenames_path
    
    collected_filenames_path = os.path.join(collection_directory, "collected_file_names.json")
    
    #collected_filenames = pd.read_json(collected_filenames_path).to_dict()
    with open(collected_filenames_path, 'r') as f:
        collected_filenames = json.load(f)
        

    try: 
        unsharpened_collected_path = collected_filenames["X_emmap_paths"][emdb_id]
        locscale_collected_path = collected_filenames["Y_emmap_paths"][emdb_id]
        mask_collected_path = collected_filenames["mask_paths"][emdb_id]

        assert os.path.exists(unsharpened_collected_path), f"Unsharpened map {unsharpened_collected_path} does not exist"
        assert os.path.exists(locscale_collected_path), f"Locscale map {locscale_collected_path} does not exist"
        assert os.path.exists(mask_collected_path), f"Mask {mask_collected_path} does not exist"
        
    except Exception as e:
        print("Skipping EMDB: ", emdb_id)
        print("!------------------------------------------!")
        raise e
    

    # Approximate the FSC resolution and Wilson cutoff to calculate B-factor for augmentation
    
    emmap, apix = load_map(unsharpened_collected_path)
    fsc_resolution = 2 * apix if apix < 3 else 6  # Aproximate the FSC resolution using pixel size
    wilson_cutoff = 10


    # create a directory to store the cubes for X: unsharpened map and Y: locscale maps
    emmap_cubes_dir = os.path.join(emdb_cubedata_dir, f"X_emmap_cubes_{emdb_id}")
    locscale_cubes_dir = os.path.join(emdb_cubedata_dir, f"Y_emmap_cubes_{emdb_id}")
    
    if not os.path.exists(emmap_cubes_dir):
        os.mkdir(emmap_cubes_dir)
    if not os.path.exists(locscale_cubes_dir):
        os.mkdir(locscale_cubes_dir)
    
    # Extract cubes from the EM map
    X_cubes_filenames_dictionary = create_and_extract_cubes_for_all_augmentation_per_map(
            emmap_path=unsharpened_collected_path, mask_path=mask_collected_path, \
            step_size=step_size, cube_size=cube_size, signal_to_noise_cubes=4,\
            output_dir=emmap_cubes_dir, fsc_resolution=fsc_resolution, wilson_cutoff=wilson_cutoff,\
            map_type="X", max_num_cubes=max_num_cubes)
    
    X_augmentation_log_file = os.path.join(emmap_cubes_dir, "augmentation_log.json")
    X_cube_center_info = os.path.join(emmap_cubes_dir, "cubes","cube_information.json")
    
    Y_cubes_filenames_dictionary = create_and_extract_cubes_for_all_augmentation_per_map(
            emmap_path=locscale_collected_path, mask_path=mask_collected_path, \
            step_size=step_size, cube_size=cube_size, signal_to_noise_cubes=4,\
            output_dir=locscale_cubes_dir, fsc_resolution=fsc_resolution, wilson_cutoff=wilson_cutoff,\
            augmentation_log_file=X_augmentation_log_file, cubecenter_info=X_cube_center_info,\
            map_type="Y", max_num_cubes=max_num_cubes)
    

    # # Extract cube centers from the cube_info_emmap
    # cubecenters_from_emmap = list(cube_info_emmap[0]['cube_center_info'].values())

    # # Extract cubes from the locscale map
    # cube_info_locscale, Y_cubes_filenames_dictionary = extract_cubes_from_cubecenters(
    #         emmap_path=locscale_collected_path, cubecenters=cubecenters_from_emmap, cube_size=cube_size, output_dir=locscale_cubes_dir)

    XY_filenames_matched = match_XY_filenames(X_cubes_filenames_dictionary, Y_cubes_filenames_dictionary)

    with open(matched_filenames_path, 'w') as f:
        json.dump(XY_filenames_matched, f, indent=4)

    print(f"  Done: {emdb_id}  →  {emdb_cubedata_dir}")
    return XY_filenames_matched

def match_XY_filenames(X_filenames_dict, Y_filenames_dict):
    XY_matched_list = []
    
    for augmentation_type_X in X_filenames_dict:
        augmentation_type_Y = augmentation_type_X if augmentation_type_X == "rotation" else "original"
        for aug_id_X in X_filenames_dict[augmentation_type_X].keys():
            aug_id_Y = str(aug_id_X) if augmentation_type_X == "rotation" else "0"
            
            for center in X_filenames_dict[augmentation_type_X][aug_id_X]:
                cube_filename_X = X_filenames_dict[augmentation_type_X][aug_id_X][center]
                cube_filename_Y = Y_filenames_dict[augmentation_type_Y][aug_id_Y][center]
                
                XY_matched_list.append(tuple([cube_filename_X, cube_filename_Y]))
    
    
    return XY_matched_list
        
def clean_up_list(filtered_cubecenters_raw):
    filtered_cubecenters_clean = [tuple(x.replace("(","").replace(")","").replace(" ","").split(",")) for x in filtered_cubecenters_raw]
    filtered_cubecenters = [tuple([int(x[0]),int(x[1]),int(x[2])]) for x in filtered_cubecenters_clean]
    return filtered_cubecenters

def extract_filtered_centers(cubeinfo_path):
    import json 
    cubeinfo = json.load(open(cubeinfo_path,"r"))
    
    rotated_cubecenters = {}
    filtered_cubecenters_raw = [x for x in list(cubeinfo["original"]['0'].keys())]
    filtered_cubecenters = clean_up_list(filtered_cubecenters_raw)
    
    signal_cubes_raw = [x for x in filtered_cubecenters_raw if os.path.basename(cubeinfo["original"]['0'][x]).split("_")[0] == "signal"]
    signal_cubes = clean_up_list(signal_cubes_raw)
    
    for aug_id in cubeinfo["rotation"].keys():
        rot_filtered_cubecenters_raw = list(cubeinfo["rotation"][aug_id].keys())
        rot_filtered_cubecenters = clean_up_list(rot_filtered_cubecenters_raw)
        rot_signal_cubes_raw = [x for x in rot_filtered_cubecenters_raw if os.path.basename(cubeinfo["rotation"][aug_id][x]).split("_")[0] == "signal"]
        rot_signal_cubes = clean_up_list(rot_signal_cubes_raw)
        
        rotated_cubecenters[aug_id] = {
            "cubecenters" : rot_filtered_cubecenters,
            "signal_cubes" : rot_signal_cubes }
    
    
    return filtered_cubecenters,  signal_cubes, rotated_cubecenters

def preprocess_emmap(emmap, apix, standardize):
    '''
    Function to preprocess the EM map path
    
    '''
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc, resample_map
    from locscale.emmernet.run_emmernet import standardize_map

    
    emmap_resampled = resample_map(emmap, apix=apix, apix_new=1.0, order=2)
    if standardize:
        emmap_standardized = standardize_map(emmap_resampled)
    else:
        emmap_standardized = emmap_resampled

    emmap_preprocessed = emmap_standardized
    return emmap_preprocessed

def copy_files(source_path, destination_folder):
    import shutil
    destination_path = os.path.join(destination_folder, os.path.basename(source_path))
    if not os.path.exists(destination_path):
        shutil.copy(source_path, destination_path)

    return destination_path


def create_combined_dataset(cubedata_directory, combined_dataset_filename):
    import os
    import json 
    from tqdm import tqdm
    import h5py
    emdb_ids_in_cubedata = [x for x in os.listdir(cubedata_directory) if os.path.isdir(os.path.join(cubedata_directory, x))]
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
                y_h5_file = os.path.join(emdb_dir, f"Y_emmap_cubes_{os.path.basename(emdb_dir)}","cubes",f"{y_augmentation_type}", y_h5_basename)
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
                
                
    with h5py.File(combined_dataset_path, "r") as combined_dataset:
        print(f"Combined dataset: {len(combined_dataset.keys())} entries")
                
    return combined_dataset_path

def collect_all_data(collection_directory, training_targets_json, num_maps_training=None, num_maps_validation=None):
    '''
    Function to preprocess all EMDB PDB entries
    
    '''
    assert os.path.exists(collection_directory), f"Directory {collection_directory} does not exist"
    import json
    from tqdm import tqdm
    assert os.path.exists(training_targets_json), f"File {training_targets_json} does not exist" 
    import random
    random.seed(42)
    with open(training_targets_json, "r") as f:
        local_files_locscale = json.load(f)
    
    
    X_emmap_paths_new = {}
    Y_emmap_paths_new = {}
    mask_paths_new = {}
    emdb_keys_all = list(local_files_locscale.keys())
    
    if num_maps_training is not None:
        emdb_keys = random.sample(emdb_keys_all, num_maps_training+num_maps_validation)
    else:
        emdb_keys = emdb_keys_all
    
    for emdb in tqdm(emdb_keys, desc="Collecting all data"):
        emdb_input_files = local_files_locscale[emdb]
        X_path = emdb_input_files["X"]
        Y_path = emdb_input_files["Y"]
        mask_path = emdb_input_files["M"]
        X_path_copied = copy_files(X_path, collection_directory)
        Y_path_copied = copy_files(Y_path, collection_directory)
        mask_path_copied = copy_files(mask_path, collection_directory)

        X_emmap_paths_new[emdb] = X_path_copied
        Y_emmap_paths_new[emdb] = Y_path_copied
        mask_paths_new[emdb] = mask_path_copied

# Save the paths to a json file
    collected_file_names = {
        "X_emmap_paths": X_emmap_paths_new,
        "Y_emmap_paths": Y_emmap_paths_new,
        "mask_paths": mask_paths_new,
        "emdb_keys": emdb_keys
        
    }
    with open(os.path.join(collection_directory, "collected_file_names.json"), "w") as f:
        json.dump(collected_file_names, f)
    
    return collected_file_names

    
def prepare_dataset_for_all_emdbs_parallel(emdb_ids_to_prepare, cubedata_directory, collection_directory, combined_h5_filename, step_size=24, cube_size=32, n_jobs=10, max_cubes=None):
    '''
    Function to prepare the dataset for all EMDB PDB entries in parallel
    
    '''
    from joblib import Parallel, delayed
    import os
    import json
    import pickle
    from tqdm import tqdm

    assert os.path.exists(cubedata_directory), f"Directory {cubedata_directory} does not exist"
    from datetime import datetime
    
    if max_cubes is not None:
        max_num_cubes_per_entry = max_cubes // len(emdb_ids_to_prepare)
    else:
        max_num_cubes_per_entry = None

    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(chunk_and_save_emdb_pdb_cubes)(
                        emdb_id=emdb_id, step_size = step_size ,cube_size = cube_size, \
                        cubedata_dir = cubedata_directory, collection_directory=collection_directory, \
                        max_num_cubes=max_num_cubes_per_entry) \
                        for emdb_id in emdb_ids_to_prepare)
    X_filenames_dataset = []
    Y_filenames_dataset = []
    XY_filenames_dataset = []
    for result in results:
        if result not in [420, 840]:
            XY_filenames_dataset.extend(result)

    num_results = len(results)
    num_skipped = len([r for r in results if r in [420, 840]])
    print(f"Done: {num_results - num_skipped}/{num_results} entries processed ({num_skipped} skipped)")

    # Save the filenames of the cubes as a json file
    X_filenames_dataset_json = os.path.join(cubedata_directory, "X_filenames_dataset.json")
    Y_filenames_dataset_json = os.path.join(cubedata_directory, "Y_filenames_dataset.json")
    XY_filenames_dataset_pickle = os.path.join(cubedata_directory, "XY_filenames_dataset.pickle")
    XY_filenames_dataset_json = os.path.join(cubedata_directory, "XY_filenames_dataset.json")
    # with open(X_filenames_dataset_json, 'w') as f:
    #     json.dump(X_filenames_dataset, f)
    # with open(Y_filenames_dataset_json, 'w') as f:
    #     json.dump(Y_filenames_dataset, f)
    
    with open(XY_filenames_dataset_pickle, 'wb') as f:
        pickle.dump(XY_filenames_dataset, f)
    with open(XY_filenames_dataset_json, 'w') as f:
        json.dump(XY_filenames_dataset, f)

    # Combine h5 files into a single h5 file with external links to the original h5 files
    combined_file_path = create_combined_dataset(cubedata_directory, combined_h5_filename)
    
    return X_filenames_dataset, Y_filenames_dataset, XY_filenames_dataset



### PLOT FUNCTIONS



def project_map(emmap, projection_axis, projection_type="mean"):
    """
    Project the map along a given axis
    """
    if projection_type == "mean":
        fun = np.nanmean
    elif projection_type == "max":
        fun = np.nanmax
    elif projection_type == "min":
        fun = np.nanmin
    else:
        raise ValueError(f"Unknown projection type {projection_type}")

    if projection_axis == "x":
        
        return fun(emmap, axis=2)
    elif projection_axis == "y":
        return fun(emmap, axis=1)
    elif projection_axis == "z":
        return fun(emmap, axis=0)
    else:
        raise ValueError(f"Projection axis {projection_axis} is not valid. Choose from x, y, z")

def plot_projections(emmap, cmap="viridis", show_colorbar=False, projection_type="mean", return_figure=False):
    """
    Plot the projections of the map
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    projection_in_x = project_map(emmap, "x",projection_type)
    im_x=axes[0].imshow(projection_in_x, cmap=cmap)
    axes[0].set_title("X projection")
    # show axis colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_x, cax=cax, orientation="vertical", cmap=cmap)
        
    projection_in_y = project_map(emmap, "y",projection_type)
    im_y=axes[1].imshow(projection_in_y, cmap=cmap)
    axes[1].set_title("Y projection")
    # show axis colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_y, cax=cax, orientation="vertical", cmap=cmap)
        # hide y axis ticks
        axes[1].set_yticks([])

    projection_in_z = project_map(emmap, "z",projection_type)
    im_z=axes[2].imshow(projection_in_z, cmap=cmap)
    axes[2].set_title("Z projection")
    # show colorbar
    if show_colorbar:
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_z, cax=cax, orientation="vertical", cmap=cmap)
        # hide y axis ticks
        axes[2].set_yticks([])
    
    if return_figure:
        return fig
    else:
        plt.show()

def jsonify_dictionary(input_dict):
    # convert pickle object to json object
    new_dict = {}
    for key, value in input_dict.items():
        key = str(key) 
        value_is_iterable = isinstance(value, (list, tuple, np.ndarray))
        value_is_dict = isinstance(value, dict)
        value_is_float = isinstance(value, float)
        value_is_int = isinstance(value, (np.int64, int, np.int32))
        value_is_string = isinstance(value, str)

        
        if value_is_dict:
            new_value = jsonify_dictionary(value)
        elif value_is_iterable:
            new_value = [str(x) for x in value]
        else:
            new_value = str(value)
        
        new_dict[key] = new_value
        
    
    return new_dict 

def pretty_print_dictionary(d, indent=1):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty_print_dictionary(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def remove_augmented_cubes(list_of_cube_tuples):
    """
    Remove augmented cubes from the list of cubes
    """
    list_of_new_cube_tuples = []
    for cube_tuple in list_of_cube_tuples:
        x_cube = cube_tuple[0]
        x_cube_basename = os.path.basename(x_cube)
        augmentation_type = x_cube_basename.split("_")[3]
        if augmentation_type == "0":
            list_of_new_cube_tuples.append(cube_tuple)
    
    return list_of_new_cube_tuples

def get_dirname_nth_level(path, n):
    dirname = path
    for i in range(n):
        dirname = os.path.dirname(dirname)
    return dirname

def get_emdb_id_from_cube_path(cube_path):
    emdb_id = os.path.basename(get_dirname_nth_level(cube_path, 4))
    return emdb_id
