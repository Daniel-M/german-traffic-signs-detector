import os
import sys


def create_path(path):
    try:
        os.mkdir(path)
        return os.path.join(os.path.dirname(path),path)

    except Exception as err:
        if os.path.exists:
            msg = ("The path '{}' exists!, skipping path creation.".format(path))
            print(msg)
            return os.path.join(os.path.dirname(path),path)
        else:
            msg = ("Something wrong happened. "
                   "The error caught was: '{}'").format(err)
            print(msg)
            return ""


def populate_tree(path_list):
    for path in path_list:
        create_path(path)
