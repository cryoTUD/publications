import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np
import h5py
import json
import pickle
import yaml
import atexit
import argparse
import random
from datetime import datetime

from emmernet_utils import collect_all_data, prepare_dataset_for_all_emdbs_parallel
from emmernet_models import define_model, define_model_large, define_model_dropout

random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser(description="produces neural network sharpened cryo-EM maps, trained on LocScale sharpened maps")

## MACRO VARIABLES
# type of run
parser.add_argument("-run", "--run_configuration", nargs='+', help="run configuration, options: 'data_preparation' or 'neural_network' or both, this argument is required", default=None, required=False)

# directory names
parser.add_argument("--training_targets_json", "-training_targets_json", type=str, help="Path to json file with input and training targets", required=False)
parser.add_argument("--num_maps_training", "-num_maps_training", type=int, help="Number of maps to use for training", default=None)
parser.add_argument("--num_maps_validation", "-num_maps_validation", type=int, help="Number of maps to use for validation", default=None)
parser.add_argument("--parent_data_dir", "-parent_data_dir", type=str, help="Parent directory for all data related to this model, defaults to current working directory", required=False)
## DATASETS CHARACTERISTICS
# basics
parser.add_argument("-cz", "--cube_size", type=int, help="size of map cubes, options: '64', '32' or '16', defaults to 32", default=32)

## HYPERPARAMETERS: DATA PREPARATION
# basics
parser.add_argument("-num_cubes_training", "--num_cubes_training", type=int, help="Number of cubes to use for training", default=None)
parser.add_argument("-num_cubes_validation", "--num_cubes_validation", type=int, help="Number of cubes to use for validation", default=None)

## HYPERPARAMETERS: NEURAL NETWORK
# basics: train and test
parser.add_argument("-mn", "--model_name", type=str, help="model name, format: [model_][name]", required=False)
parser.add_argument("-a", "--append_text", type=str, help="Append text for model name", default=None)
parser.add_argument("-ne", "--num_epochs", type=int, help="number of epochs, defaults to '15'", default=15)
parser.add_argument("-use_dropout", "--use_dropout", action='store_true', help="use dropout, defaults to 'False'", default=False)
# basics: train
parser.add_argument("-bs", "--batch_size", type=int, help="batch size, defaults to '8'", default=8)
parser.add_argument("-lr", "--nn_learning_rate", type=float, help="learning rate parameter, defaults to '0.001'", default=0.001)
parser.add_argument("-op", "--nn_optimizer_name", type=str, help="type of optimization algorithm, options: 'SGD' or 'Adam', defaults to 'Adam'", default="Adam")
parser.add_argument("-lo", "--nn_loss_name", type=str, help="type of loss functions, options: 'MAE', 'MAE_phase' or 'MSE', defaults to 'MAE'", default="MAE")
parser.add_argument("-me", "--nn_metric_name", type=str, help="type of trainig and validation metric, options: 'MAE' or 'MSE', defaults to 'MSE'", default="MSE")
parser.add_argument("-nn_l1_reg", "--nn_l1_reg", type=float, help="L1 regularization parameter, defaults to 'None'", default=None)
parser.add_argument("-nn_l2_reg", "--nn_l2_reg", type=float, help="L2 regularization parameter, defaults to 'None'", default=None)
parser.add_argument("-training_cube_size","--training_cube_size", type=int, help="Length of training cubes", default=60000)

# GPUs
parser.add_argument("-gpus", "--GPU_nums", nargs='+', help="numbers of the selected GPUs, format: '1 2 3'", default=[None])

# CPUs
parser.add_argument("-np", "--num_processes", type=int, help="number of processes, defaults to '10'", default=10)


def create_directories_if_not_exist(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            print("Creating directory: {}".format(directory))
            os.makedirs(directory)

def create_directories(args):
    parent_data_dir = args.parent_data_dir
    model_name = args.model_name
    append_text = args.append_text

    model_data_dir = os.path.join(parent_data_dir, model_name)

    collection_dir = os.path.join(model_data_dir, "collection_directory")
    
    outputdata_dir = os.path.join(model_data_dir, "outputdata")
    append_dir = os.path.join(outputdata_dir, append_text)
    saved_models_dir = os.path.join(append_dir, "saved_models")
    training_performance_dir = os.path.join(append_dir, "training_performance")

    cubedata_dir = os.path.join(model_data_dir, "cubedata_directory")
    cubedata_training_dir = os.path.join(cubedata_dir, "cubedata_training")
    cubedata_validation_dir = os.path.join(cubedata_dir, "cubedata_validation")

    create_directories_if_not_exist(
        model_data_dir, collection_dir, cubedata_dir, outputdata_dir, \
        append_dir, saved_models_dir, training_performance_dir,
        cubedata_training_dir, cubedata_validation_dir,
    )

    folders = {
        "model_data_dir": model_data_dir,
        "collection_dir": collection_dir,
        "cubedata_dir": cubedata_dir,
        "outputdata_dir": outputdata_dir,
        "append_dir": append_dir,
        "saved_models_dir": saved_models_dir,
        "training_performance_dir": training_performance_dir,
        "cubedata_training_dir": cubedata_training_dir,
        "cubedata_validation_dir": cubedata_validation_dir,
    }
    return folders

def print_hyperparameters(args):
    print("Hyperparameters:")
    folders = create_directories(args)
    model_name_dir = folders["model_data_dir"]
    cubedata_dir = folders["cubedata_dir"]

    cubedata_training_dir = folders["cubedata_training_dir"]
    cubedata_validation_dir = folders["cubedata_validation_dir"]

    emdb_ids_used_for_training = [x for x in os.listdir(cubedata_training_dir) if os.path.isdir(os.path.join(cubedata_training_dir, x))]
    emdb_ids_used_for_validation = [x for x in os.listdir(cubedata_validation_dir) if os.path.isdir(os.path.join(cubedata_validation_dir, x))]
    
    
    hyperparameters_dictionary = {
        "run_configuration": args.run_configuration,
        "training_targets_json" : args.training_targets_json,
        "append_text" : args.append_text,
        "model_name_dir" : model_name_dir,
        "cubedata_directory" : cubedata_dir,
        "dataset_characteristics": {
            "training_id": emdb_ids_used_for_training,
            "validation_id": emdb_ids_used_for_validation,
        },
        
        "basics": {
            "cube_size": args.cube_size,
        },
        "neural_network": {
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.nn_learning_rate,
            "optimizer_name": args.nn_optimizer_name,
            "loss_name": args.nn_loss_name,
            "metric_name": args.nn_metric_name,
            "GPU_nums": args.GPU_nums,
        },
    }
    
    print(yaml.dump(hyperparameters_dictionary, default_flow_style=False))
    # dump hyperparameters to json file
    json_file_hyperparameters = os.path.join(model_name_dir, args.model_name+"_"+"hyperparameters.json")
    with open(json_file_hyperparameters, 'w') as fp:
        json.dump(hyperparameters_dictionary, fp, indent=4)
    
    return hyperparameters_dictionary
    


class save_weights_on_epoch(tf.keras.callbacks.Callback):
    def __init__(self, model_save_folder, model_name):
        super(save_weights_on_epoch, self).__init__()
        self.model_save_folder = model_save_folder
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # save model checkpoint as hdf5 file
        model_save_path = os.path.join(self.model_save_folder, f"{self.model_name}_epoch-{epoch}.hdf5") 
        self.model.save(model_save_path)
        
        
    
class HDF5CubeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, parent_h5_path, key_list, batch_size, cube_size=48):
        """
        HDF5CubeDataGenerator: Custom data generator for loading cube-shaped data from HDF5 files with external links.

        :param parent_h5_path: Path to the parent HDF5 file containing external links to the cubes.
        :param key_list: List of top-level keys to use for data generation.
        :param batch_size: Number of samples per batch.
        :param cube_size: The size of the cubes, assuming the shape is (cube_size, cube_size, cube_size, 1).
        """
        self.parent_h5_path = parent_h5_path
        self.key_list = key_list
        self.batch_size = batch_size
        self.cube_size = cube_size

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(len(self.key_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Calculate which keys to retrieve for this batch
        batch_keys = self.key_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Initialize arrays to store the batch data
        X_data = np.empty((len(batch_keys), self.cube_size, self.cube_size, self.cube_size, 1))
        Y_data = np.empty((len(batch_keys), self.cube_size, self.cube_size, self.cube_size, 1))

        # Open the parent HDF5 file and retrieve data for each key in the batch
        with h5py.File(self.parent_h5_path, 'r') as h5_file:
            for i, key in enumerate(batch_keys):
                # Retrieve the X and Y cube datasets for the current key
                x_cube_key = list(h5_file[key].keys())[0]  # Assume the first key is the X data
                y_cube_key = list(h5_file[key].keys())[1]  # Assume the second key is the Y data

                X_data[i] = h5_file[key][x_cube_key][:]
                Y_data[i] = h5_file[key][y_cube_key][:]

        return X_data, Y_data

    def on_epoch_end(self):
        # Optionally shuffle the key list at the end of each epoch if required
        np.random.shuffle(self.key_list)

def create_hdf5_datagenerators(args):
    print("\n>>> Creating datagenerators")
    folders = create_directories(args)
    cubedata_top_directory = folders["cubedata_dir"]

    cubedata_directory_training = os.path.join(cubedata_top_directory, "cubedata_training")
    cubedata_directory_validation = os.path.join(cubedata_top_directory, "cubedata_validation")

    # Fetch the HDF5 file paths for the training and validation datasets
    h5_file_training = os.path.join(cubedata_directory_training, "combined_training_dataset.h5")
    h5_file_validation = os.path.join(cubedata_directory_validation, "combined_validation_dataset.h5")

    with h5py.File(h5_file_training, 'r') as h5_file:
        key_list_training = list(h5_file.keys())
    with h5py.File(h5_file_validation, 'r') as h5_file:
        key_list_validation = list(h5_file.keys())

    with open(os.path.join(cubedata_top_directory, "keys.json"), 'w') as f:
        json.dump({"training": key_list_training, "validation": key_list_validation}, f)

    training_cubes_length = len(key_list_training)
    validation_cubes_length = len(key_list_validation)
    print(f"  Training cubes:   {training_cubes_length}")
    print(f"  Validation cubes: {validation_cubes_length}")

    training_data_generator = HDF5CubeDataGenerator(h5_file_training, key_list_training, args.batch_size, cube_size=args.cube_size)
    validation_data_generator = HDF5CubeDataGenerator(h5_file_validation, key_list_validation, args.batch_size, cube_size=args.cube_size)

    return training_data_generator, validation_data_generator, training_cubes_length, validation_cubes_length

        

def fit_UNet_model(UNet_model, training_data_generator, validation_data_generator, training_cubes_length, validation_cubes_length, args):
    """ fits UNet model, while loading the dataset dynamically with the data generators

    Args:
        UNet_model (tf.keras.Model): UNet model object
        training_data_generator (Custom_Datagenerator): training data generator object
        validation_data_generator (Custom_Datagenerator): validation data generator object
        run_type (string): specifies the run type. Options: ["train", "train_test"]
        model_epoch (int): specifies the saved models epoch number for the train_test method. Defaults to None.

    Returns:
        history (History object): contains information about the fitting process, like the loss and metric performance per epoch
    """
    
    print("\n>>> Training model")
    folders = create_directories(args)
    saved_models_dir = folders["saved_models_dir"]

    log_dir = os.path.join(saved_models_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    epochs = args.num_epochs
    nn_callbacks = [
        save_weights_on_epoch(model_save_folder=saved_models_dir, model_name=args.model_name),
        tensorboard_callback,
    ]

    history = UNet_model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=int(training_cubes_length // args.batch_size),
        validation_steps=int(validation_cubes_length // args.batch_size),
        callbacks=nn_callbacks,
    )
    
    # Save model
    models_save_path = os.path.join(saved_models_dir, f"{args.model_name}_final_epoch_{str(epochs).zfill(2)}.hdf5")
    UNet_model.save(models_save_path)
    
    return history
    
    

def train_UNet_model(UNet_model, args):
    print("\n### Training UNet model ###")
    folders = create_directories(args)
    saved_models_dir = folders["saved_models_dir"]

    # create datagenerators from cubedata
    training_data_generator, validation_data_generator, length_training, length_validation = create_hdf5_datagenerators(args)
    
    history = fit_UNet_model(
        UNet_model=UNet_model, training_data_generator=training_data_generator, validation_data_generator=validation_data_generator,\
        training_cubes_length=length_training, validation_cubes_length=length_validation, args=args
    )
    
    training_history_path = os.path.join(saved_models_dir, "training_history.pickle")
    try:
        with open(training_history_path, 'wb') as fp:
            pickle.dump(history, fp)
    except Exception as e:
        print(f"Could not save training history: {e}")



def prepare_data(args):
    from sklearn.model_selection import train_test_split

    print("\n### Collecting and preparing training and validation data ###")
    
    folders = create_directories(args)

    collection_data_dir = folders["collection_dir"]
    cubedata_training_dir = folders["cubedata_training_dir"]
    cubedata_validation_dir = folders["cubedata_validation_dir"]

    training_targets_json = args.training_targets_json

    collected_data = collect_all_data(collection_data_dir, training_targets_json=training_targets_json)

    emdb_keys = collected_data["emdb_keys"]

    # Prepare the dataset for training and validation
    emdb_training_id, emdb_validation_id = train_test_split(emdb_keys, test_size=0.15, random_state=42, shuffle=True)

    cube_size = args.cube_size
    step_size_trainval = int(cube_size / 4 * 3)
    num_processes = args.num_processes
    max_cubes_training = args.num_cubes_training
    max_cubes_validation = args.num_cubes_validation

    _ = prepare_dataset_for_all_emdbs_parallel(emdb_training_id, \
        cubedata_directory=cubedata_training_dir, \
        collection_directory=collection_data_dir,
        combined_h5_filename="combined_training_dataset.h5", \
        step_size=step_size_trainval, cube_size=cube_size, n_jobs=num_processes, max_cubes=max_cubes_training)

    _ = prepare_dataset_for_all_emdbs_parallel(emdb_validation_id, \
        cubedata_directory=cubedata_validation_dir, \
        collection_directory=collection_data_dir,
        combined_h5_filename="combined_validation_dataset.h5", \
        step_size=step_size_trainval, cube_size=cube_size, n_jobs=num_processes, max_cubes=max_cubes_validation)

    print("\n### Data preparation finished ###")
    

def run_UNet_model(args):
    create_directories(args)
    GPU_nums_str = ",".join([str(x) for x in args.GPU_nums])
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_nums_str
    GPU_nums_length = len(args.GPU_nums) 
    GPU_names = []
    for i in np.arange(GPU_nums_length):
        GPU_num = args.GPU_nums[i]
        GPU_names.append(("/gpu:"+ str(GPU_num)))

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=GPU_names)

    # Select the right type of model

    if args.use_dropout:
        model_definition_function = define_model_dropout
    else:    
        if args.cube_size == 32:
            model_definition_function = define_model
        elif args.cube_size == 64 or args.cube_size == 48:
            model_definition_function = define_model_large
        else:
            raise ValueError("Cube size {} not supported.".format(args.cube_size))

    with mirrored_strategy.scope():
        if args.nn_loss_name == "MAE":
            nn_loss = tf.keras.losses.MeanAbsoluteError()
            nn_metric = ['mae']
        else:
            nn_loss = tf.keras.losses.MeanAbsoluteError()
            nn_metric = ['mse']
    
        UNet_model = model_definition_function(args.cube_size)
        if args.nn_optimizer_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.nn_learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.nn_learning_rate)
        UNet_model.compile(optimizer=optimizer, loss=nn_loss, metrics=nn_metric)
        #UNet_model.summary()

        train_UNet_model(UNet_model, args)

    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)
    print("\n### EmmerNet finished ###")
    

def main():
    
    # parse input arguments from user
    args_default = parser.parse_args()



    # Change the default values of arguments for segmenting the data
    args_dict = vars(args_default)
    # Set configuration: 
    # "data_preparation" to only prepare the data,
    # "neural_network" to only run the neural network training,
    # "both" to run both the data preparation and the neural network training in one run

    args_dict["run_configuration"] = "both"
    # Setting output directories and files
    args_dict["parent_data_dir"] = "/home/abharadwaj1/dev/map_sharpening/emmernet/default_parking" #<-- set parent data directory, 
    args_dict["model_name"] = "locscale2_training_test" #<-- set model name, collection dir, cubedata_dir and outputdata_dir will be created inside this dir 
    args_dict["append_text"] = "dropout_0p5_lr_0p0001_l1_0p01" #<-- set append text to store the model weights and training performance for different hyperparameter settings in different folders, format: [datetime]_[append_text], if None, only datetime will be used
    # Set the input to training as a json file 
    args_dict["training_targets_json"] = "training_targets_temp.json"

    # Processing parameters (num CPU and GPU)
    args_dict["num_processes"] = 3 # Number of processes to prepare the data 
    args_dict["GPU_nums"] = [1] # GPU numbers to use for training, format: '0 1 2 3'

    # Hyperparameters: data preparation
    args_dict["cube_size"] = 32
    args_dict["batch_size"] = 6
    # Hyperparameters: neural network training 
    args_dict["num_epochs"] = 15
    args_dict["nn_learning_rate"] = 0.0001
    args_dict["use_dropout"] = True
    
    args = argparse.Namespace(**args_dict)
    
    # Create directories for outputs 
    _ = create_directories(args)

    # Run configuration decision tree
    if "data_preparation" in args.run_configuration:
        prepare_data(args)
    elif "neural_network" in args.run_configuration:
        print_hyperparameters(args)
        run_UNet_model(args)
    elif "both" in args.run_configuration:
        prepare_data(args)
        print_hyperparameters(args)
        run_UNet_model(args)
    else:
        print("Error: please specify '--run_configuration' as 'data_preparation' or 'neural_network' or 'both', this argument is required")


# run main function
if __name__ == '__main__':
    main()
