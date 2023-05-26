from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform
import numpy 

openmm_dir = '@OPENMM_DIR@'
Polaranglesplugin_header_dir = '@PolaranglesPLUGIN_HEADER_DIR@'
Polaranglesplugin_library_dir = '@PolaranglesPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_Polaranglesplugin',
                      sources=['PolaranglesPluginWrapper.cpp'],
                      libraries=['OpenMM', 'PolaranglesPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), Polaranglesplugin_header_dir, numpy.get_include()],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), Polaranglesplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='Polaranglesplugin',
      version='1.0',
      py_modules=['Polaranglesplugin'],
      ext_modules=[extension],
     )
