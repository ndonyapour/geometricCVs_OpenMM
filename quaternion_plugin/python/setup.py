from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform
import numpy 

openmm_dir = '@OPENMM_DIR@'
Quaternionplugin_header_dir = '@QuaternionPLUGIN_HEADER_DIR@'
Quaternionplugin_library_dir = '@QuaternionPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_Quaternionplugin',
                      sources=['QuaternionPluginWrapper.cpp'],
                      libraries=['OpenMM', 'QuaternionPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), Quaternionplugin_header_dir, numpy.get_include()],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), Quaternionplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='Quaternionplugin',
      version='1.0',
      py_modules=['Quaternionplugin'],
      ext_modules=[extension],
     )
