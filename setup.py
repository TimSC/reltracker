from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extmods = [Extension("relutil", ["relutil.pyx"], include_dirs=[np.get_include()])]

setup(
  name = 'reltracker',
  cmdclass = {'build_ext': build_ext},
  ext_modules = extmods
)

#python setup.py build_ext --inplace
