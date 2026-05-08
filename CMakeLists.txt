# CMake project file for HelFEM
cmake_minimum_required (VERSION 3.14)
project (helfem CXX Fortran)
# C++ standard is set per-target via target_compile_features(... cxx_std_17)
# rather than globally.

# Ensure we are building out-of-source so that the tests work (issue 120)
get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)
if("${srcdir}" STREQUAL "${bindir}")
  message(FATAL_ERROR "HelFEM should not be configured & built in the source directory.
Please create a separate build directory and rerun cmake from the build directory.
For example: mkdir objdir; cd build; cmake ..")
endif()

# Following options can be used to configure the HelFEM CMake build. They can be set by
# passing the corresponding -D flags to CMake, e.g.:
#
#     cmake -DHELFEM_FIND_DEPS=ON -DHELFEM_BINARIES=OFF ..
#
# - USE_OPENMP: if set to OFF, force-disables OpenMP. Note that if it is on, OpenMP may
#   still get disabled if CMake is not able to find OpenMP libraries.
#
# - HELFEM_FIND_DEPS: if set to ON, CMake will try to find all dependencies automatically
#   with find_package calls. If successful, no user configuration should be necessary to
#   compile HelFEM.
#
#   In case some libraries are installed in non-standard locations, the CMAKE_PREFIX_PATH
#   variable can be used to configure additional paths where CMake looks for the libraries.
#   In this situation, additional RPATH/RUNPATH or LD_LIBRARY_PATH configuration may be
#   necessary to run binaries that have been installed with `make install`.
#
# - HELFEM_BINARIES: if set to OFF, disables the building of the HelFEM binary program that
#   are located under src/. This is useful in cases where only the libhelfem library is
#   needed and you wish to avoid the HDF5 and libxc dependencies.
#
# - HELFEM_CMAKE_SYSTEM: if set to OFF, the CMake.system file is ignored. This is useful
#   when creating multiple out-of-tree build directories from the same source directory with
#   different settings, some using CMake.system and some not. Primarily useful when
#   developing the build scripts.
#
option(USE_OPENMP "Compile OpenMP enabled version (for parallel calculations)?" ON)
option(HELFEM_FIND_DEPS "Try to find dependencies automatically?" OFF)
option(HELFEM_BINARIES "Compile HelFEM binaries?" ON)
option(HELFEM_CMAKE_SYSTEM "Load the CMake.system file (if present)?" ON)

# System specific definitions
if(HELFEM_CMAKE_SYSTEM AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/CMake.system")
    message("Including user-provided CMake.system.")
    include("${CMAKE_CURRENT_SOURCE_DIR}/CMake.system")
endif()

# Try to automatically find and configure all the dependent libraries, if the user specified
# the HELFEM_FIND_DEPS option.
#
# If the dependencies are installed in non-standard paths (e.g. somewhere under /home), then
# the CMAKE_PREFIX_PATH variables can be used to specify the path to the root of the install
# prefix. E.g. if the libraries are installed under /home/username/local/usr, then CMake
# could be invoked with:
#
#     cmake -DCMAKE_PREFIX_PATH=$HOME/local/usr ..
#
if(HELFEM_FIND_DEPS)
    # Try to find the Armadillo C++ library
    find_package(Armadillo REQUIRED)
    message("Found Armadillo ${ARMADILLO_VERSION_NAME} (v${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH})")
    message("ARMADILLO_VERSION_STRING: ${ARMADILLO_VERSION_STRING}")
    message("ARMADILLO_INCLUDE_DIRS:   ${ARMADILLO_INCLUDE_DIRS}")
    message("ARMADILLO_LIBRARIES:      ${ARMADILLO_LIBRARIES}")
    if(${ARMADILLO_VERSION_MAJOR} LESS "9")
        message(FATAL_ERROR "Armadillo too old, version 9 or above required.")
    endif()
    # Globally link the Armadillo library to everything
    include_directories("${ARMADILLO_INCLUDE_DIRS}")
    link_libraries("${ARMADILLO_LIBRARIES}")
    # If we should also compile all the HelFEM binaries (under src/), we need also need to
    # set up the HDF5 and libxc dependencies:
    if(HELFEM_BINARIES)
        find_package(HDF5 REQUIRED COMPONENTS CXX)
        message("HDF5_FOUND: ${HDF5_FOUND}")
        message("HDF5_VERSION: ${HDF5_VERSION}")
        message("HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
        message("HDF5_LIBRARIES: ${HDF5_LIBRARIES}")

        find_package(LibXc REQUIRED PATHS ".")
        message("LibXc_FOUND=${LibXc_FOUND}")
        message("LIBXC_INCLUDE_DIRS=${LIBXC_INCLUDE_DIRS}")
        message("LIBXC_LIBRARIES=${LIBXC_LIBRARIES}")

        include_directories("${HDF5_INCLUDE_DIRS}" "${LIBXC_INCLUDE_DIRS}")
        link_libraries("${HDF5_LIBRARIES}" "${LIBXC_LIBRARIES}")
    endif()
endif()

# Wigner-nj symbols: prefer a system-installed wignernj >= 0.5.0, otherwise
# fetch and build that version.
find_package(wignernj 0.5.0 QUIET)
if(NOT wignernj_FOUND)
    message(STATUS "wignernj >= 0.5.0 not found via find_package; fetching v0.5.0")
    include(FetchContent)
    FetchContent_Declare(
        wignernj
        GIT_REPOSITORY https://github.com/susilehtola/wignernj.git
        GIT_TAG        v0.5.0
    )
    FetchContent_MakeAvailable(wignernj)
    # find_package would expose the library as the namespaced wignernj::wignernj
    # imported target. Provide the same alias from FetchContent so consumers
    # can always link against the same name.
    if(NOT TARGET wignernj::wignernj)
        add_library(wignernj::wignernj ALIAS wignernj)
    endif()
else()
    message(STATUS "Using system wignernj (${wignernj_VERSION})")
endif()

# Find OpenMP support
if(USE_OPENMP)
 find_package( OpenMP )
 if(OPENMP_FOUND)
  # Add flags to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 endif()
endif()

# Include libhelfem headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libhelfem/include")

# Compile code
add_subdirectory(libhelfem)
if(HELFEM_BINARIES)
    add_subdirectory(src)
endif()
