from distutils.core import setup
from distutils.extension import Extension

import os
import numpy

path = os.getcwd()

setup(
    ext_modules = [Extension('inpaintBCT',
                     sources=[os.path.join(path,'inpaintBCT.cpp'),
                              os.path.join(path,'heap.cpp'),
                              os.path.join(path,'inpainting_func.cpp')],
                     include_dirs=[numpy.get_include()],
                     language='c++')
    ]
)
