# Setup the environment
def setup_environment():
    import os 
    data_archive_path = os.environ['THESIS_DATA_ARCHIVE_PATH']
    print("-" * 80)
    print(f"Sourcing data from: {data_archive_path}")
    print("-" * 80)
    return data_archive_path

def copy_file_to_folder(file, folder):
    """Copy a file to a specified folder."""
    import shutil
    return shutil.copy(file, folder)


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

        if value_is_dict:
            new_value = jsonify_dictionary(value)
        elif value_is_iterable:
            new_value = [str(x) for x in value]
        elif not value_is_string:
            new_value = str(value)
        
        new_dict[key] = new_value
        #print("key: {}, value_is_iterable: {}, value_is_dict: {}, value_is_float: {}, value_is_int: {}, value_is_string: {}".format(key, value_is_iterable, value_is_dict, value_is_float, value_is_int, value_is_string))
    
    return new_dict 

def create_folders_if_they_do_not_exist(*args):
    import os
    for folder in args:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

def assert_paths_exist(*args):
    import os
    for path in args:
        assert os.path.exists(path), f"Path does not exist: {path}"

def any_files_are_missing(*args):
    import os
    some_files_are_missing = False
    for path in args:
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            some_files_are_missing = True
            
    return some_files_are_missing