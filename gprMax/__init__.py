"""
======
gprMax
======

Electromagnetic wave propagation simulation software.

"""

from ._version import __version__
# lazily import the top-level API to avoid pulling in heavy dependencies
# (e.g. psutil, pycuda) when only submodules such as :mod:`fractals` are used.
from importlib import import_module

def run(*args, **kwargs):
    """Entry point equivalent to :func:`gprMax.gprMax.api`.

    The actual module is imported the first time ``run`` is invoked.  This
    allows other parts of the library to be imported for testing without
    triggering the full solver import tree.
    """
    mod = import_module('.gprMax', __name__)
    return mod.api(*args, **kwargs)

__name__ = 'gprMax'
