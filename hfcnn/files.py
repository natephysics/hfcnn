# tools for handleing raw files.
import os
import pickle as pkl
import pandas as pd

def export_data_to_local_cache(data, path: str):
    """Exports data to the local drive cache in the plk format.

    Parameters
    ----------
    data : string with the label of the data

    path : string with path
    """
    # export the data
    if isinstance(data, pd.DataFrame):
        data.to_pickle(path)
    else:
        with open(path, 'wb') as handle:
            pkl.dump(data, handle)
    return

def generate_file_path(timestamp: int, port: int, directory: str = './data'):
    """Gerenates a string with the correct file path.
    """
    return os.path.join(directory, str(port), str(timestamp) + '.pkl')

def import_file_from_local_cache(file_path):
    """Imports the file from the local drive cache.

    Parameters
    ----------
    file_path : string with the file path

    Returns
    -------
    file from cache
    """
    with open(file_path, 'rb') as handle:
        data = pkl.load(handle)
    return data


def export_and_merge_data_frame(data_frame, path='./data/df.pkl', return_merged=False):
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

    #  if not, just use the existing one
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