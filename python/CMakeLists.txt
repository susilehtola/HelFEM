# Python bindings for HelFEM. Built when HELFEM_PYTHON=ON is set on the
# top-level CMake configure.

# pybind11 provides a CMake config package. Prefer finding it directly
# (system / vcpkg / conda installs put it on CMAKE_PREFIX_PATH). A
# pip-installed pybind11 ships its CMake package inside the wheel rather
# than on a standard prefix, so if the direct lookup fails, ask the
# Python interpreter where pybind11 put its CMake dir and retry with that
# as a hint.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_rc
  )
  if(NOT _pybind11_rc EQUAL 0)
    message(FATAL_ERROR
      "HELFEM_PYTHON=ON but pybind11 was not found on CMAKE_PREFIX_PATH "
      "and is not importable from the Python at\n"
      "  ${Python3_EXECUTABLE}\n"
      "Install pybind11 (e.g. pip install pybind11) and reconfigure.")
  endif()
  find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_cmake_dir}")
endif()

pybind11_add_module(_helfem MODULE bindings.cpp)
target_compile_features(_helfem PRIVATE cxx_std_17)
# legendre is a separate target (HelFEM's existing build splits the
# associated-Legendre Fortran wrappers into their own library; the
# in-tree executables link to it alongside helfem-common; we do the
# same here).
target_link_libraries(_helfem PRIVATE helfem-common helfem legendre)

# Place the .so directly under build_dir/python/helfem/ so the
# package layout is usable straight from the build tree with
# `PYTHONPATH=<build>/python python -c 'import helfem'`. Also copy
# the __init__.py shim into the same dir at configure time.
set_target_properties(_helfem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/helfem")
foreach(_pyfile __init__.py pyscf_driver.py)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/helfem/${_pyfile}"
                 "${CMAKE_CURRENT_BINARY_DIR}/helfem/${_pyfile}" COPYONLY)
endforeach()

# Install the .so + the Python shim under ${CMAKE_INSTALL_PREFIX}/python/helfem/.
install(TARGETS _helfem LIBRARY DESTINATION python/helfem)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/helfem/__init__.py"
              "${CMAKE_CURRENT_SOURCE_DIR}/helfem/pyscf_driver.py"
        DESTINATION python/helfem)

# ---------------------------------------------------------------- tests
# These bindings rotted unnoticed once already: they were left on Armadillo
# straight through the Eigen migration, and because HELFEM_PYTHON defaults OFF
# nothing ever compiled them again. Register the Python tests so that a
# configure which asks for the bindings also checks them.
#
# The scripts are COPIED into the build tree rather than run where they sit.
# Python puts a script's own directory first on sys.path, so running
# <src>/python/test_smoke.py makes `import helfem` resolve to the SOURCE
# helfem/ package -- which holds no extension module -- and the import fails
# however PYTHONPATH is set. Run them from <build>/python, where helfem/ does
# contain the .so, and the import needs no environment at all.
#
# All three need numpy only: the pyscf imports in helfem.pyscf_driver are
# deliberately inside the functions that use them, so the tests stay runnable
# on a machine without pyscf.
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                RESULT_VARIABLE _helfem_numpy_rc
                OUTPUT_QUIET ERROR_QUIET)
if(_helfem_numpy_rc EQUAL 0)
  foreach(_pytest test_smoke test_radial_df_factors test_radial_df_exhaustive)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${_pytest}.py"
                   "${CMAKE_CURRENT_BINARY_DIR}/${_pytest}.py" COPYONLY)
    add_test(NAME python.${_pytest}
             COMMAND ${Python3_EXECUTABLE}
                     "${CMAKE_CURRENT_BINARY_DIR}/${_pytest}.py")
    set_tests_properties(python.${_pytest} PROPERTIES
        LABELS "python"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 600)
  endforeach()
  message(STATUS "HelFEM Python tests: 3 case(s) registered (ctest -L python).")
else()
  message(STATUS "numpy is not importable from ${Python3_EXECUTABLE}: "
                 "HelFEM Python tests not registered.")
endif()
