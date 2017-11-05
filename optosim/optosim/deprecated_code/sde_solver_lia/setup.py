from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy 

extensions = [Extension(
        name="solve",
        sources=["solve.pyx"],
        include_dirs=[numpy.get_include()],
        )
]

setup(
    ext_modules = cythonize(extensions),
)
