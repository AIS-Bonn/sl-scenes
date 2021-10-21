"""
Utils methods
"""

import os
import datetime


def clear_cmd():
    """Clearning command line window"""
    os.system("cls" if os.name == "nt" else "clear")
    return


def timestamp():
    """ Obtaining the current timestamp in an human-readable way """

    timestamp = (
        str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    )

    return timestamp
