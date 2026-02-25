import SimpleITK as sitk
import numpy as np
from radiomics.featureextractor import RadiomicsFeatureExtractor
import os
import sys
sys.path.append("/home/abharadwaj1/.conda/envs/locscale/lib/python3.7/site-packages/")
from locscale.include.emmer.ndimage.map_utils import load_map
sys.path.insert(1,'/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/notebooks/extra')
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
def calculate_radiomic_features(mask_array, apix, settings_file='/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/get_map_statistics/params.yml'):
    mask = sitk.GetImageFromArray(mask_array.astype(int))      # Ensure to convert to int

    # Create PyRadiomics feature extractor

    extractor = RadiomicsFeatureExtractor(settings_file)
    extractor.settings['resampledPixelSpacing'] = [apix, apix, apix]
    # Get surface area to volume ratio
    extractor.settings['enableCExtensions'] = True


    # Extract features
    featurevector = extractor.execute(mask, mask)

    return featurevector

def gatherStats(filepath):
    micelle, apix  = load_map(filepath)

    featuredict = calculate_radiomic_features((micelle>0.5).astype(np.float32), apix)
    length = featuredict['original_shape_MajorAxisLength']
    width  = featuredict['original_shape_MinorAxisLength']
    height = featuredict['original_shape_LeastAxisLength']
    elongation = featuredict['original_shape_Elongation']
    surface_area = featuredict['original_shape_SurfaceArea']
    sphericity = featuredict['original_shape_Sphericity']
    surface_volume_ratio = featuredict['original_shape_SurfaceVolumeRatio']



    features = {}
    features["length"] = length
    features["width"]  = width
    features["height"] = height
    features["COM"] = np.array(featuredict['diagnostics_Mask-original_CenterOfMass']) / micelle.shape[0]
    features["elongation"] = elongation
    features["surface_area"] = surface_area
    features["surface_volume_ratio"] = surface_volume_ratio
    features["sphericity"] = sphericity
    




    return features

def main():
    import pickle
    import json
    from tqdm import tqdm
    import os
    import joblib

    collected_files_json = "/home/abharadwaj1/scratch/dev/emmernet_training/training_cubes/segmentation_with_curated_micelle/collection_directory/collected_file_names.json"
    with open(collected_files_json, 'r') as f:
        collected_files = json.load(f)

    n_jobs = 10

    training_targets_dictionary = collected_files["Y_locscale_paths"]
    emdb_ids = list(training_targets_dictionary.keys())
    features_dictionary = {}
    
    # parallelize the process
    features = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(gatherStats)(filepath) for filepath in tqdm(training_targets_dictionary.values()))
    for i, emdb_id in enumerate(emdb_ids):
        features_dictionary[emdb_id] = {
            "features" : features[i],
            "filepath" : training_targets_dictionary[emdb_id],
            "emdb_id" : emdb_id,
        }
    
    
    
    collected_feature_info = os.path.join(os.path.dirname(collected_files_json), "training_targets_feature_info.pickle")
    with open(collected_feature_info, 'wb') as f:
        pickle.dump(features_dictionary, f)

if __name__ == "__main__":
    main()