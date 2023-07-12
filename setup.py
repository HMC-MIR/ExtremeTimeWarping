from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Extreme Time Warp',
    ext_modules=cythonize("dtw_algorithm.pyx"),
    zip_safe=False,
)
