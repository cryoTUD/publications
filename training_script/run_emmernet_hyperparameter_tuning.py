import subprocess
import os 
import sys 
import argparse
from datetime import datetime

# Parse the command line arguments to check if this is dry run or not
parser = argparse.ArgumentParser(description='Run emmernet hyperparameter tuning')
parser.add_argument('--dry_run', action='store_true', default=False, help='Dry run')
parser.add_argument('--temp_emmernet_run', action='store_true', default=False, help='Temporary emmernet run')


global dry_run; dry_run = parser.parse_args().dry_run
global temp_emmernet_run; temp_emmernet_run = parser.parse_args().temp_emmernet_run

def create_emmernet_command(batch_size_per_GPU, optimizer, nn_loss_name, nn_learning_rate, l1_reg=None, l2_reg=None):
    if temp_emmernet_run:
        emmernet_path = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet/scripts/emmernet_temp_dry.py"
    else:
        emmernet_path = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet/scripts/EMmerNet.py"
    cmd = []
    cmd.append('python')
    cmd.append(emmernet_path)
    cmd.append('-run')
    cmd.append('neural_network')
    cmd.append('-mn')
    cmd.append('model_based_C')
    cmd.append('-ne')
    cmd.append('15')
    cmd.append('-gpus')
    cmd.append('4')
    cmd.append('5')
    cmd.append('6')
    cmd.append('7')
    cmd.append('-locscale_directory')
    cmd.append('/home/abharadwaj1/dev/map_sharpening/emmernet/locscale_inputs/model_based_version_C')
    
    # Now add the hyperparameters that we want to tune
    # create a short text to describe the hyperparameters for this run
    append_text = 'hyperparameter_tuning_2/bs_' + str(batch_size_per_GPU) + '_opt_' + optimizer + '_nnloss_' + nn_loss_name + '_lr_' + str(nn_learning_rate) + '_l1_' + str(l1_reg) + '_l2_' + str(l2_reg)
    cmd.append('--append')
    cmd.append(append_text)
    # batch_size_per_GPU
    cmd.append('-bs')
    cmd.append(str(int(batch_size_per_GPU*4)))
    # nn_optimizer_name
    cmd.append('--nn_optimizer_name')
    cmd.append(optimizer)
    # nn_loss_name
    cmd.append('--nn_loss_name')
    cmd.append(nn_loss_name)
    # learning_rate
    cmd.append('--nn_learning_rate')
    cmd.append(str(nn_learning_rate))
    
    if l1_reg is not None:
        cmd.append('--nn_l1_reg')
        cmd.append(str(l1_reg))
    
    if l2_reg is not None:
        cmd.append('--nn_l2_reg')
        cmd.append(str(l2_reg))
        
    main_model_output_dir = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/model_based_C/outputdata"
    run_dir = os.path.join(main_model_output_dir, append_text)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Create a dictionary of the hyperparameters with its values
    hyperparameters = {
        "batch_size_per_GPU": batch_size_per_GPU,
        "optimizer": optimizer,
        "nn_loss_name": nn_loss_name,
        "nn_learning_rate": nn_learning_rate,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "run_dir": run_dir
    }   

    return cmd, run_dir, hyperparameters

def run_command(cmd_info):
    cmd, run_dir, hyperparameters = cmd_info
    print("Running the following command:")
    print(" ".join(cmd))
    print("."*80)
    log_file = os.path.join(run_dir, "log.txt")
    with open(log_file, "w") as f:
        print("Running the following command:", file=f)
        print(" ".join(cmd), file=f)
        print("."*80, file=f)
        print("Date: {}".format(datetime.now()), file=f)
        print("."*80, file=f)
    
    log_file_open = open(log_file, "a")
    
    if dry_run:
        return 0
    try:
        # # Create an environment variable to set the CUDA_VISIBLE_DEVICES
        # env = os.environ.copy()
        # env = {'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        #p=subprocess.run(cmd, check=True, capture_output=True, env=env)
        p = subprocess.run(" ".join(cmd), check=True, shell=True, stdout=log_file_open, stderr=subprocess.STDOUT)
        # If successful, return 0
        print(p.returncode)
    except subprocess.CalledProcessError as e:
        print("Error running command:")
        print(" ".join(cmd))
        print("+-"*40)
        print("Error command:")
        print(e.cmd)
        print("Return code:")
        print(e.returncode)
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
        print("Full error:")
        print(e)
        return 1
    except Exception as e2:
        print("Unknown error running command:")
        print(" ".join(cmd))
        print("+-"*40)
        raise e2

def modify_default_parameters_for_single_parameter(default_params, param_name, param_value_list):
    commands_to_study_param = []
    params_study = default_params.copy()
    for param_value in param_value_list:
        params_study[param_name] = param_value
        commands_to_study_param.append(create_emmernet_command(**params_study))
    
    return commands_to_study_param


def run_list_of_commands(list_of_commands, study_name):
    for i, cmd in enumerate(list_of_commands):
        p = run_command(cmd)
        if p == 0:
            # upload the logs to tensorboard
            run_dir = cmd[1]
            hyperparameters = cmd[2]
            log_file_path = os.path.join(run_dir, "saved_models", "logs")
            
            name = f"EMmerNet (MB) hyperparameters {study_name}_{i}"
            description = " | ".join([x+"="+str(y) for x,y in hyperparameters.items()])
            
            # Upload the logs to tensorboard
            upload_command = f"tensorboard dev upload --one_shot --logdir {log_file_path} --name '{name}' --description '{description}'"
            try:
                print("Uploading logs to tensorboard")
                os.system(upload_command)
            except Exception as e:
                print("Error uploading logs to tensorboard")
                print(e)
                
        print("="*80)

            
# Hyperparameters options 

batch_size_per_GPU_list = [2,4]
optimizer_list = ['SGD']
nn_loss_name_list = ['MSE']
nn_learning_rate_list = [1e-5, 1e-6]
regularised_model_l1 = [0.1, 0.2]
regularised_model_l2 = [0.1, 0.2]

default_params = {'batch_size_per_GPU': 8, 'optimizer': 'Adam', 'nn_loss_name': 'MAE', 'nn_learning_rate': 0.001, 'l1_reg': None, 'l2_reg': None}

# Create a list of all the commands that we want to run
default_parameters_command = [create_emmernet_command(**default_params)]
## Impact of batch size
commands_to_study_batch_size = modify_default_parameters_for_single_parameter(default_params, 'batch_size_per_GPU', batch_size_per_GPU_list)    
## Impact of optimizer
commands_to_study_optimizer = modify_default_parameters_for_single_parameter(default_params, 'optimizer', optimizer_list)
## Impact of nn_loss_name
commands_to_study_nn_loss_name = modify_default_parameters_for_single_parameter(default_params, 'nn_loss_name', nn_loss_name_list)
## Impact of nn_learning_rate
commands_to_study_nn_learning_rate = modify_default_parameters_for_single_parameter(default_params, 'nn_learning_rate', nn_learning_rate_list)


# Create default params for the regularised model
default_params_regularised = {'batch_size_per_GPU': 4, 'optimizer': 'Adam', 'nn_loss_name': 'MAE', 'nn_learning_rate': 1e-5, 'l1_reg': 0.01, 'l2_reg': 0.01}

commands_to_study_regularised = [create_emmernet_command(**default_params_regularised)]
## Impact of l1_reg
commands_to_study_l1_reg = modify_default_parameters_for_single_parameter(default_params_regularised, 'l1_reg', regularised_model_l1)
## Impact of l2_reg
commands_to_study_l2_reg = modify_default_parameters_for_single_parameter(default_params_regularised, 'l2_reg', regularised_model_l2)

# Run all the commands
print("Running all the commands")
print("="*80)
# list all the commands
for cmd in default_parameters_command + commands_to_study_batch_size + commands_to_study_nn_learning_rate + commands_to_study_regularised + commands_to_study_l1_reg + commands_to_study_l2_reg:
    print(" ".join(cmd[0]))
    print("-"*80)

run_list_of_commands(default_parameters_command, "default")
run_list_of_commands(commands_to_study_batch_size, "study_batch_size")
# run_list_of_commands(commands_to_study_optimizer, "study_optimizer")
#run_list_of_commands(commands_to_study_nn_loss_name, "study_nn_loss_name")
run_list_of_commands(commands_to_study_nn_learning_rate, "study_nn_learning_rate")
run_list_of_commands(commands_to_study_regularised, "study_regularised")
run_list_of_commands(commands_to_study_l1_reg, "study_l1_reg")
run_list_of_commands(commands_to_study_l2_reg, "study_l2_reg")

    
    
    
    
    