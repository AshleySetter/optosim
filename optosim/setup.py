from sys import version
print(version)
if version[0] != '3':
    raise OSError("This package requires python 3")
from setuptools import setup
from setuptools.extension import Extension
import pip
pip.main(['install', '-r', 'requirements.txt']) # installs requirements from requirements.txt file
import os
import numpy
    
mypackage_root_dir = os.path.dirname(__file__)
with open(os.path.join(mypackage_root_dir, 'requirements.txt')) as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(mypackage_root_dir, 'optosim/VERSION')) as version_file:
    version = version_file.read().strip()

extensions = [
    Extension(
        name="optosim.solveRK",
        sources=["optosim/solve_cython/solveRK.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name="optosim.solveRK_kalman",
        sources=["optosim/solve_cython_kalman/solveRK.c"],
        include_dirs=[numpy.get_include()],
    )
]

setup(name='optosim',
      version=version,
      description='Python package with functions for data analysis',
      author='Ashley Setter',
      author_email='A.Setter@soton.ac.uk',
      url="https://github.com/AshleySetter/optosim",
      download_url="https://github.com/AshleySetter/optosim/archive/{}.tar.gz".format(version),
      include_package_data=True,
      packages=['optosim',
                'optosim.sde_solver',
                'optosim.sde_solver_kalman',                
      ],
      ext_modules = extensions,
      install_requires=requirements,
)
