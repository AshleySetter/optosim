from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

ext = [Extension('solveRK_kalman',
                sources=["solveRK.pyx"],
                include_dirs = [numpy.get_include()])]

setup(name = 'solveRK_kalman', ext_modules=ext, cmdclass = { 'build_ext': build_pyx })
