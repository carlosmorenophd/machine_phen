"""Function to use in other class"""
import os
import json

import pandas as pd


from src.helpers.key_env import IS_DEBUG, FolderCache, FOLDER_DATA


def get_file(file_name: str, folder: FolderCache) -> str:
    """Get file from cache"""
    file = os.path.join(FOLDER_DATA, folder.value, file_name)
    return file


def get_file_to_data_frame(file_name: str, folder: FolderCache) -> pd.DataFrame:
    """Util - load csv file with pandas """
    file = os.path.join(FOLDER_DATA, folder.value, file_name)
    if IS_DEBUG:
        print(os.getcwd())
        print(FOLDER_DATA)
        print(file)
    if os.path.isfile(file):
        return pd.read_csv(file)
    raise FileNotFoundError(f"file not found - {file}")


def get_file_to_json(file_name: str, folder: FolderCache) -> pd.DataFrame:
    """Util - load json str to json dictionary """
    file = os.path.join(FOLDER_DATA, folder.value, file_name)
    if IS_DEBUG:
        print(os.getcwd())
        print(FOLDER_DATA)
        print(file)
    if os.path.isfile(file):
        json_dict = None
        with open(file, 'r', encoding="utf-8") as f:
            json_dict = json.loads(f)
        return json_dict
    raise FileNotFoundError(f"file not found - {file}")


def get_absolute_file_to_data_frame(path_to_file: str) -> pd.DataFrame:
    """file access - load csv file with pandas from absolute file name """
    if IS_DEBUG:
        print(os.getcwd())
        print(FOLDER_DATA)
        print(path_to_file)
    if os.path.isfile(path_to_file):
        return pd.read_csv(path_to_file)
    raise FileNotFoundError(f"file not found - {path_to_file}")


def save_to_csv(data_frame: pd.DataFrame, file_save: str, is_index: bool = False) -> None:
    """Util -  save any data frame on file """
    path_to_save = get_file(file_name=file_save, folder=FolderCache.UPLOAD)
    data_frame.to_csv(path_to_save, index=is_index)


def get_name_file_without_extension(file_name: str, folder: FolderCache) -> str:
    """Return only name of file

    Args:
        file_name (str): file name
        folder (FolderCache): Folder location

    Returns:
        str: _description_
    """
    file = os.path.join(FOLDER_DATA, folder.value, file_name)
    complete_name = os.path.basename(file)
    name = os.path.splitext(complete_name)
    return name[0]
