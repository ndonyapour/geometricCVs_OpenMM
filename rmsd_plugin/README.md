OpenMM RMSDCVForce Plugin
=====================
This project is an implementation of the [RMSDCVForce][http://docs.openmm.org/7.5.0/api-c++/generated/OpenMM.RMSDCVForce.html] as a plugin for [OpenMM](https://openmm.org). The plugin utilizes the same code provided in OpenMM, but efforts have been made to optimize it for use as a plugin. The RMSDCVForce calculates the Root-Mean-Square Deviation (RMSDCV) between a set of atom positions and reference positions.


Installing OpenMM
===================
You should use conda to make a new virtual environment:

#+begin_src sh
  conda create -n omm 
  conda activate omm
#+end_src

Then you manually install the dependencies using
conda. 

#+begin_src sh
conda install -c conda-forge compilers cmake swig
conda install -c conda-forge openmm=8.0.0 cudatoolkit 
#+end_src
Then run the following command to make sure OpenMM runs on the four platforms. If it didn't run on CUDA try to install the version of `cudatoolkit` that matches your GPU.
#+begin_src sh
  python -m openmm.testInstallation
#+end_src

Building The Plugin
===================
This project uses [CMake](http://www.cmake.org) for its build system.  To build it, follow these
steps:


1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries. 
Note: You can use this method to find the openmm header file folders
    conda list | grep openmm
    it will you show something like this  `openmm                    8.0.0           py311hbc0ebcd_0    conda-forge`
    then search for the openmm build tag in your home directory
    find . -name "*py311hbc0ebcd_0*"
  
5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

6. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and
OPENCL_LIBRARY are set correctly, and that RMSDCV_BUILD_OPENCL_LIB is selected.

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that RMSDCV_BUILD_CUDA_LIB is selected.

8. Press "Configure" again if necessary, then press "Generate".

9. Use the build system you selected to build and install the plugin.  For RMSDCV, if you
selected Unix Makefiles, type `make install`.


Test Cases
==========

To run all the test cases build the "test" target, for RMSDCV by typing `make test`.

This project contains several different directories for test cases: one for each platform, and
another for serialization related code.  Each of these directories contains a CMakeLists.txt file
that automatically creates a test from every file whose name starts with "Test" and ends with
".cpp".  To create new tests, just add a new file to any of these directories.  The file should
contain a `main()` function that executes any tests in the file and returns 0 if all tests were
successful or 1 if any of them failed.

Usually plugins are loaded dynamically at runtime, but that doesn't work well for test cases:
you want to be able to run the tests before the plugin has yet been installed into the plugins
directory.  Instead, the test cases directly link against the relevant plugin libraries.  But
that creates another problem: when a plugin is dynamically loaded at runtime, its platforms and
kernels are registered automatically, but that doesn't happen for code that statically links
against it.  Therefore, the very first line of each `main()` function typically invokes a method
to do the registration that _would_ have been done if the plugin were loaded automatically:

    registerRMSDCVOpenCLKernelFactories();

The OpenCL and CUDA test directories create three tests from each source file: the program is
invoked three times while passing the strings "single", "mixed", and "double" as a command line
argument.  The `main()` function should take this value and set it as the default precision for
the platform:

    if (argc > 1)
        Platform::getPlatformByName("OpenCL").setPropertyDefaultValue("OpenCLPrecision", string(argv[1]));

This causes the plugin to be tested in all three of the supported precision modes every time you
run the test suite.


OpenCL and CUDA Kernels
=======================

The OpenCL and CUDA versions of the force are implemented with the common compute framework.
This allows us to write a single class (`CommonCalcRMSDCVForceKernel`) that provides an
implementation for both platforms at the same time.  Device code is written in a subset of
the OpenCL and CUDA languages, with a few macro and function definitions to make them
identical.

The OpenCL and CUDA platforms compile all of their kernels from source at runtime.  This
requires you to store all your kernel source in a way that makes it accessible at runtime.  That
turns out to be harder than you might think: simply storing source files on disk is brittle,
since it requires some way of locating the files, and ordinary library files cannot contain
arbitrary data along with the compiled code.  Another option is to store the kernel source as
strings in the code, but that is very inconvenient to edit and maintain, especially since C++
doesn't have a clean syntax for multi-line strings.

This project (like OpenMM itself) uses a hybrid mechanism that provides the best of both
approaches.  The source code for the kernels is found in the `platforms/common/src/kernels`
directory.  At build time, a CMake script loads every .cc file contained in the directory
and generates a class with all the file contents as strings.  For the RMSDCV plugin, the
directory contains a single file called RMSDCVForce.cc.  You can
put anything you want into this file, and then C++ code can access the content of that file
as `CommonRMSDCVKernelSources::RMSDCVForce`.  If you add more .cc files to this directory,
correspondingly named variables will automatically be added to `CommonRMSDCVKernelSources`.


Python API
==========

OpenMM uses [SWIG](http://www.swig.org) to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

When building OpenMM's Python API, the interface file is generated automatically from the C++
API.  That guarantees the C++ and Python APIs are always synchronized with each other and avoids
the potential bugs that would come from having duplicate definitions.  It takes a lot of complex
processing to do that, though, and for a single plugin it's far simpler to just write the
interface file by hand.  You will find it in the "python" directory.

To build and install the Python API, build the "PythonInstall" target, for RMSDCV by typing
"make PythonInstall".  (If you are installing into the system Python, you may need to use sudo.)
This runs SWIG to generate the C++ and Python files for the extension module
(RMSDCVPluginWrapper.cpp and RMSDCVplugin.py), then runs a setup.py script to build and
install the module.  Once you do that, you can use the plugin from your Python scripts:

    from openmm import System
    from RMSDCVplugin import RMSDCVForce
    system = System()
    force = RMSDCVForce()
    system.addForce(force)


License
=======

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2014-2021 Stanford University and the Authors.

Authors: Peter Eastman

Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

