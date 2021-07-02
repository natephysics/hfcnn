# tools for handleing raw files. 
import os
import hickle as hkl
import pandas as pd

def check_if_file_exists(filename, port=False, cache_folder_path='./data'):
    """Checks to see if the hickle binary for an image at a given timestamp is already cached to disk.
    
    Parameters
    ----------
    filename : string with the unix absolute timestamp

    port: int with the number of the port

    cache_folder_path : string with folder path

    Returns
    -------
    True if the file exists in the provided folder. False otherwise. 
    """
    # mode 
    mode = 0o666

    # construct the file path
    if port:
        port_dir = os.path.join(cache_folder_path, port)
        file_path = os.path.join(port_dir, filename)
        # check to see if the port directory exists and make one if not
        if not os.path.isdir(port_dir):
            os.mkdir(port_dir, mode)
    else:
        file_path = os.path.join(cache_folder_path, filename)
  
    # check to see if the file exists in that directory
    if os.path.isfile(file_path):
        return True
    else:
        return False


def export_data_to_local_cache(data, path: str):
    """Exports data to the local drive cache in the hlk format.
    
    Parameters
    ----------
    data : string with the label of the data

    path : string with path
    """
    # export the data
    hkl.dump(data, path)
    return

def generate_file_path(timestamp: int, port: int, directory: str = './data'):
    """Gerenates a string with the correct file path.
    """
    return os.path.join(directory, str(port), str(timestamp) + '.hkl')
    

def import_file_from_local_cache(file_path):
    """Imports the file from the local drive cache.
    
    Parameters
    ----------
    file_path : string with the file path

    Returns
    -------
    file from cache
    """
    return hkl.load(file_path)


def export_and_merge_data_frame(data_frame, path='./data/df.hkl', return_merged=False):
    """Merges the data_frame into the local cached version of the data frame.
    Compares each timestamp and does a sorted merge of the dataframes, replacing
    the dataframe on disk with the merged dataframe.
    
    Parameters
    ----------
    data_frame : pandas dataframe created by extract_heat_load_by_time function.

    path : path to local cache.

    Returns
    -------
    Merge Completed Successfully
    """
    # Check to see if there is an existing file
    if os.path.isfile(path):
        # import the cached dataframe
        cached_df = import_file_from_local_cache(path)

        # concat dataframes
        temp_df = pd.concat([data_frame, cached_df], ignore_index=True)

        # sort df
        temp_df = temp_df.sort_values(by='times', ignore_index=True)

        # remove duplicates
        temp_df = temp_df.drop_duplicates()

        # export df to disk
        export_data_to_local_cache(temp_df, path)
        
        print('Export Successful')
    
    #if not, just use the existing one
    else:
        # sort df
        temp_df = data_frame.sort_values(by='times', ignore_index=True)

        # export df to disk
        export_data_to_local_cache(temp_df, path)

        print('Export Successful')

    if return_merged:
        return temp_df
    else: 
        return 