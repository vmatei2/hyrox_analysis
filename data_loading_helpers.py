import os
import constants as _constants
def list_files_in_directory(directory):
    """
    Function to list all the files in an existing directory - expected to be the data folder in this case
    :param directory: path to the file
    :return: list of strings, each string being the name of one of the files
    """
    try:
        items = os.listdir(directory)

        # filter out directories, only keep files, remove .csv from filename
        files = [item.removesuffix(".csv") for item in items if os.path.isfile(os.path.join(directory, item))]
        files.append(_constants.ALL_RACES)
        return files
    except Exception as e:
        return f"Error loading files in the directory: {e}"
