from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

pyx_directories = ["model/cython/"]

extensions = [
    Extension(
        '*',
        ["*.pyx"],
        extra_compile_args=["-std=c++11"])
]

pwd = os.getcwd()
for dir in pyx_directories:
    target_dir = os.path.join(pwd, dir)
    os.chdir(target_dir)
    setup(
        ext_modules=cythonize(extensions, language="c++", annotate=False),
        include_dirs=[numpy.get_include()]
    )
    os.chdir(pwd)
