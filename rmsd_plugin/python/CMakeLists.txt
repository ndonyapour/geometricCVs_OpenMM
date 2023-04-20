set(WRAP_FILE RMSDCVPluginWrapper.cpp)
set(MODULE_NAME RMSDCVplugin)

# Execute SWIG to generate source code for the Python module.
set(SWIG_INPUT_FILES2
    "${CMAKE_CURRENT_SOURCE_DIR}/rmsdcvplugin.i"
    "${CMAKE_CURRENT_SOURCE_DIR}/typemaps.i"
    "${CMAKE_CURRENT_SOURCE_DIR}/header.i"
)

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/rmsdcvplugin.i"
    DEPENDS 
        ${SWIG_INPUT_FILES2}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(RMSDCVPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(RMSDCVPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

