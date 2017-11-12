"""
optosim
=======

Package of functions for the Matter-Wave Interferometry 
group for simulating experimental data.


"""

# init file

import os

_mypackage_root_dir = os.path.dirname(__file__)
_version_file = open(os.path.join(_mypackage_root_dir, 'VERSION'))
__version__ = _version_file.read().strip()

# import cython extensions
import optosim.solveRK
import optosim.solveRK_kalman

# import sub-modules
import optosim.sde_solver
import optosim.sde_solver_kalman

# the following line imports all the functions from optosim.py
from .optosim import *

