import os

import reframe.utility.sanity as sn


@sn.deferrable
def path_join(*path):
    """Deferable version of os.path.join"""
    return os.path.join(*path)
