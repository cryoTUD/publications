import os
import numpy as np
from locscale.include.emmer.ndimage.map_utils import load_map, resample_map
from locscale.include.emmer.ndimage.filter import get_cosine_mask
from locscale.include.emmer.ndimage.fsc_util import calculate_phase_correlation_maps
from locscale.include.emmer.ndimage.profile_tools import frequency_array

def preprocess_map(emmap, apix, standardize=True):
    """
    Resample and standardize a map.
    """
    # Resample the map to 1A per pixel
    emmap_resampled = resample_map(emmap, apix=apix, apix_new=1)

    # Standardize the map
    if standardize:
        emmap_standardized = (emmap_resampled - np.mean(emmap_resampled)) / np.std(emmap_resampled)
        return emmap_standardized
    else:
        return emmap_resampled

def load_maps(input_folder, emdb_id, format_string):
    """
    Load a map file from the folder based on the EMDB ID and file type.
    """
    map_path = os.path.join(input_folder, format_string.format(emdb_id))
    emmap, apix = load_map(map_path)
    return emmap, apix

def calculate_phase_correlations(unsharp_map, target_map, mask, num_samples=100):
    from locscale.include.emmer.ndimage.map_utils import (
        extract_window,
        load_map,
        get_all_voxels_inside_mask,
        resample_map,
    )
    from locscale.include.emmer.ndimage.filter import get_cosine_mask
    from locscale.include.emmer.ndimage.fsc_util import (
        calculate_phase_correlation_maps,
        calculate_amplitude_correlation_maps,
    )
    from locscale.include.emmer.ndimage.profile_tools import (
        frequency_array,
        compute_radial_profile,
    )
    from locscale.emmernet.run_emmernet import load_emmernet_model
    from locscale.utils.file_tools import get_locscale_path
    import random

    locscale_path = get_locscale_path()
    ## Load the model
    model_path = os.path.join(locscale_path, "locscale", "emmernet", "emmernet_models", "emmernet", "EMmerNet_highContext.hdf5")
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
    emmernet_type = "emmernet_high_context"
    emmernet_model_folder = os.path.dirname(model_path)
    verbose = True
    cuda_visible_devices_string = ""
    input_dictionary = {
        "trained_model": emmernet_type,
        "emmernet_model_folder": emmernet_model_folder,
        "verbose": verbose,
        "model_path": model_path,
        "cuda_visible_devices_string": cuda_visible_devices_string
    }

    emmernet_model = load_emmernet_model(input_dictionary)
    print("Loaded the model from {}".format(model_path))

    emmap_path_unsharp = os.path.abspath(unsharp_map)
    emmap_path_target = os.path.abspath(target_map)
    mask_path = os.path.abspath(mask)

    # Load maps
    emmap_unsharp_raw, apix_unsharp = load_map(emmap_path_unsharp)
    emmap_target_raw, apix_target = load_map(emmap_path_target)
    mask_raw, apix_mask = load_map(mask_path)

    # Preprocess maps
    emmap_unsharp = preprocess_map(emmap_unsharp_raw, apix_unsharp, standardize=True)
    emmap_target = preprocess_map(emmap_target_raw, apix_target, standardize=True)
    mask = preprocess_map(mask_raw, apix_mask, standardize=False)

    # Binarize and smooth mask
    mask_binarized = (mask > 0.99).astype(np.int_)
    mask_smooth = get_cosine_mask(mask_binarized, 5)

    # Get all voxels inside mask
    all_voxels_inside_mask = get_all_voxels_inside_mask(mask, mask_threshold=0.99)
    # Exclude voxels that are too close to the edge
    buffer_size = 32
    all_voxels = []
    for voxel in all_voxels_inside_mask:
        if (
            voxel[0] > buffer_size
            and voxel[0] < mask.shape[0] - buffer_size
            and voxel[1] > buffer_size
            and voxel[1] < mask.shape[1] - buffer_size
            and voxel[2] > buffer_size
            and voxel[2] < mask.shape[2] - buffer_size
        ):
            all_voxels.append(voxel)


    if len(all_voxels) < num_samples:
        print("Warning: Number of voxels inside mask is less than 100.")
        num_samples = len(all_voxels)
    sampled_voxels = random.sample(all_voxels, num_samples)

    # print("Sampled voxels (first 3):", sampled_voxels[:3])

    window_size_pix = 32  # Since we resampled to apix=1, window size is 32 pixels

    # Initialize lists to store correlations
    phase_correlations_unsharp = []
    amplitude_correlations_unsharp = []
    phase_correlations_target = []
    amplitude_correlations_target = []
    phase_correlations_unsharp_target = []
    amplitude_correlations_unsharp_target = []

    batch_size = 10  # Set batch size to 10

    # Iterate over sampled voxels in batches
    for i in range(0, len(sampled_voxels), batch_size):
        batch_voxels = sampled_voxels[i:i+batch_size]
        num_cubes = len(batch_voxels)
        cube_size = window_size_pix

        # Initialize arrays to store the windows
        cubes_batch_X = np.empty((num_cubes, cube_size, cube_size, cube_size, 1))
        windows_unsharp = []
        windows_target = []

        for idx, center in enumerate(batch_voxels):
            print(f"Processing voxel {i+idx+1}/{num_samples} at position {center}")

            # Extract windows
            window_unsharp = extract_window(emmap_unsharp, center, window_size_pix)
            window_target = extract_window(emmap_target, center, window_size_pix)

            windows_unsharp.append(window_unsharp)
            windows_target.append(window_target)

            # Add window to batch array
            cubes_batch_X[idx] = window_unsharp[..., np.newaxis]

        # Predict using the model on the batch
        cubes_batch_predicted = emmernet_model.predict(
            x=cubes_batch_X, batch_size=batch_size, verbose=0
        )

        # Process predictions
        cubes_batch_predicted = np.squeeze(cubes_batch_predicted, axis=-1)  # Shape: (num_cubes, cube_size, cube_size, cube_size)

        for idx in range(num_cubes):
            prediction_cube = cubes_batch_predicted[idx]
            window_unsharp = windows_unsharp[idx]
            window_target = windows_target[idx]

            # Compute correlations with unsharp map window
            phase_corr_unsharp = calculate_phase_correlation_maps(window_unsharp, prediction_cube)
            amplitude_corr_unsharp = calculate_amplitude_correlation_maps(window_unsharp, prediction_cube)

            phase_correlations_unsharp.append(phase_corr_unsharp)
            amplitude_correlations_unsharp.append(amplitude_corr_unsharp)

            # Compute correlations with target map window
            phase_corr_target = calculate_phase_correlation_maps(window_target, prediction_cube)
            amplitude_corr_target = calculate_amplitude_correlation_maps(window_target, prediction_cube)

            phase_correlations_target.append(phase_corr_target)
            amplitude_correlations_target.append(amplitude_corr_target)

            # Compute correlations between unsharp and target windows
            phase_corr_unsharp_target = calculate_phase_correlation_maps(window_unsharp, window_target)
            amplitude_corr_unsharp_target = calculate_amplitude_correlation_maps(window_unsharp, window_target)

            phase_correlations_unsharp_target.append(phase_corr_unsharp_target)
            amplitude_correlations_unsharp_target.append(amplitude_corr_unsharp_target)

            # Compute radial profiles
            rp_pred = compute_radial_profile(prediction_cube)
            rp_target = compute_radial_profile(window_target)
            rp_unsharp = compute_radial_profile(window_unsharp)

    

    # Convert lists to numpy arrays
    phase_correlations_unsharp = np.array(phase_correlations_unsharp)
    amplitude_correlations_unsharp = np.array(amplitude_correlations_unsharp)
    phase_correlations_target = np.array(phase_correlations_target)
    amplitude_correlations_target = np.array(amplitude_correlations_target)
    phase_correlations_unsharp_target = np.array(phase_correlations_unsharp_target)
    amplitude_correlations_unsharp_target = np.array(amplitude_correlations_unsharp_target)


    # Frequency array using frequency_array function
    apix = 1  # Since maps have been resampled to apix=1
    freq = frequency_array(phase_correlations_unsharp[0], apix)

    output_dictionary = {
        "freq": freq,
        "phase_correlations_unsharp": phase_correlations_unsharp,
        "amplitude_correlations_unsharp": amplitude_correlations_unsharp,
        "phase_correlations_target": phase_correlations_target,
        "amplitude_correlations_target": amplitude_correlations_target,
        "phase_correlations_unsharp_target": phase_correlations_unsharp_target,
        "amplitude_correlations_unsharp_target": amplitude_correlations_unsharp_target,
    }

    return output_dictionary

def plot_correlations(x_array, y_array,  x_label, y_label, title_text, \
                    scatter=False, figsize_cm=(14,8),font="Arial",fontsize=10,\
                    fontscale=1,hue=None,find_correlation=True, alpha=0.3, filepath=None,\
                    xticks=None, yticks=None, xlim=None, ylim=None, hue_array=None):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    from scipy import stats
    from sklearn.metrics import r2_score
    import matplotlib 
    import pandas as pd
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig = plt.figure(figsize=figsize, dpi=600) # dpi=600 for publication quality
    #sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    #sns.set_style("white")
    print(f"Length of hue_array: {len(hue_array)}")
    def annotate(data, **kws):
        pearson_correlation = stats.pearsonr(x_array, y_array)
        #r2 = r2_score(x_array, y_array)
        r2_text = f"$R$ = {pearson_correlation[0]:.2f}"
        #r2_text = f"$R^2$ = {r2:.2f}"
        ax = plt.gca()
        ax.text(.05, .8, r2_text,transform=ax.transAxes)
    # Create a pandas dataframe for the data
    data = pd.DataFrame({x_label: x_array, y_label: y_array})
    print(f"Done creating dataframe with shape {data.shape}")
    # Plot the data    
    g = sns.lmplot(x=x_label, y=y_label, data=data, scatter=scatter, legend=False)
    print("Done creating lmplot")
    g.map_dataframe(annotate)
    print("Done annotating")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        # return the figure object
        return fig
    