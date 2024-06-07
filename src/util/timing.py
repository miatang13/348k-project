import os
from datetime import datetime


def get_last_modified_timestamp(file_path):
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        return "File not found"


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
