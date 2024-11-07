"""Function to use in other class"""
import os
from enum import Enum

import pandas as pd
from dotenv import load_dotenv


load_dotenv()

FOLDER_DATA = os.getenv('FOLDER_DATA', '../cache/')
REDIS_BROKEN = os.getenv("REDIS_URL")
IS_DEBUG=bool(os.getenv('DEBUG_MODE', 'false'))
FolderCache = Enum('FolderCache', [("UPLOAD", "upload_files")])


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


def save_to_csv(data_frame: pd.DataFrame, file_save: str, is_index: bool = False) -> None:
    """Util -  save any data frame on file """
    path_to_save = get_file(file_name=file_save, folder=FolderCache.UPLOAD)
    data_frame.to_csv(path_to_save, index=is_index)
