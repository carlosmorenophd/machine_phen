"""All env and static variable for system"""
import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

FOLDER_DATA = os.getenv('FOLDER_DATA', '../data/')
REDIS_BROKEN = os.getenv("REDIS_URL")
IS_DEBUG = bool(os.getenv('DEBUG_MODE', 'false'))
FolderCache = Enum('FolderCache', [("UPLOAD", "upload_files")])
