# CMake project file for HelFEM
#
# Quick start (Linux or macOS, dependencies on a standard prefix):
#
#     cmake -B build -DCMAKE_INSTALL_PREFIX=install
#     cmake --build build -j
#     cmake --install build
#
# Dependencies installed in non-standard locations are found by adding
# their install prefixes to CMAKE_PREFIX_PATH, e.g.
#
#     cmake -B build -DCMAKE_PREFIX_PATH="/opt/libxc;$(brew --prefix)"
#
# There is no machine-specific config file to maintain any more: all
# dependencies are located with find_package / pkg-config and consumed
# as imported targets. See the option documentation below for the few
# knobs that remain (BLAS integer width, external BLAS, OpenMP).
cmake_minimum_required(VERSION 3.20)
project(helfem CXX)

# All in-tree static archives are built PIC so they can be linked into a
# shared object (the pybind11 _helfem module, or a downstream shared
# libatomscf) without `ld -shared` rejecting them.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to a Release build if the user did not pick one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enforce out-of-source builds so the tests work (issue 120).
get_filename_component(_srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bindir "${CMAKE_BINARY_DIR}" REALPATH)
if("${_srcdir}" STREQUAL "${_bindir}")
    message(FATAL_ERROR "HelFEM must be configured out-of-source.\n"
                        "Run e.g.  cmake -B build  from the source root.")
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# - USE_OPENMP:  build the OpenMP-parallel version. On macOS this needs a
#   real OpenMP runtime (`brew install libomp`); AppleClang ships none by
#   default. If find_package(OpenMP) fails the build errors with a hint.
#
# - HELFEM_BINARIES: build the executables under src/ (atomic, diatomic,
#   gensap, ...). OFF builds only libhelfem, avoiding the HDF5 and libxc
#   dependencies.
#
# - HELFEM_PYTHON: build the pybind11 `_helfem` module.
#
# - HELFEM_ILP64: use 64-bit BLAS/LAPACK integers (ILP64) and tell
#   Armadillo to match (ARMA_BLAS_LONG). ON reproduces the large-matrix
#   configuration HelFEM has historically shipped and requires an ILP64
#   BLAS (e.g. flexiblas64, or MKL/OpenBLAS built ILP64). Turn OFF on
#   platforms whose only BLAS is LP64 -- notably macOS Accelerate -- to
#   link an ordinary 32-bit-integer BLAS. ARMA_64BIT_WORD (64-bit element
#   indexing) is set either way, so large matrices work in both modes;
#   only the BLAS call ABI differs. A mismatch between this and the BLAS
#   actually linked silently corrupts the heap inside LAPACK (surfaces as
#   "free(): invalid next size"), so keep it consistent with your BLAS.
#
# - HELFEM_BLAS_PROVIDED_EXTERNALLY: do not find or link BLAS/LAPACK here;
#   a parent project provides them (libatomscf embedding its own MKL, say).
#   Sets -DARMA_DONT_USE_WRAPPER so Armadillo resolves BLAS symbols at the
#   parent's final link step. The parent must match HELFEM_ILP64.
option(USE_OPENMP "Build the OpenMP-parallel version" ON)
option(HELFEM_BINARIES "Build the HelFEM executables under src/" ON)
option(HELFEM_PYTHON "Build the HelFEM Python (pybind11) bindings" OFF)
option(HELFEM_ILP64 "Use 64-bit-integer (ILP64) BLAS/LAPACK" ON)

# Quad precision (_Float128, 113-bit mantissa, ~34 decimal digits).
#
# The HelFEM sources are C++17 and stay that way -- nothing in them requires a
# newer standard. But libstdc++ only provides <cmath> overloads and
# std::numeric_limits for _Float128 under C++23, and Eigen needs both. (With
# them, Eigen works at _Float128 with no NumTraits specialisation and no math
# shims at all; note _Float128 is the standard C++23 extended-float type, and
# is NOT the legacy __float128, which collides with libstdc++'s overloads.)
#
# So quad is an OPT-IN that raises the standard on the helfem targets only.
# Default: on when the compiler advertises C++23, off otherwise -- so nobody's
# compiler floor moves unless they ask for it. Everything still builds and runs
# at double and long double either way.
include(CheckCXXCompilerFlag)
if(cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  set(HELFEM_FLOAT128_DEFAULT ON)
else()
  set(HELFEM_FLOAT128_DEFAULT OFF)
endif()
option(HELFEM_FLOAT128 "Instantiate the templated core at _Float128 (needs C++23)" ${HELFEM_FLOAT128_DEFAULT})
option(HELFEM_BLAS_PROVIDED_EXTERNALLY
       "Skip BLAS/LAPACK discovery; parent project provides them" OFF)
# Route Eigen's dense products (GEMM/GEMV/SYRK/...) through the same
# external BLAS Armadillo uses, via EIGEN_USE_BLAS. Eigen- and
# Armadillo-computed dense products then agree bit-for-bit (both hit the
# same BLAS), and large products speed up. This requires a 32-bit-integer
# (LP64) BLAS: stock Eigen's BLAS backend is int32-only, and we do NOT
# patch Eigen to change that. So it is incompatible with HELFEM_ILP64 --
# hence the default follows the integer width (on for LP64, off for ILP64),
# and enabling it under ILP64 is a hard error. Off => Eigen uses its own
# kernels (the historical behaviour).
if(HELFEM_ILP64 OR HELFEM_BLAS_PROVIDED_EXTERNALLY)
  set(HELFEM_EIGEN_BLAS_DEFAULT OFF)
else()
  set(HELFEM_EIGEN_BLAS_DEFAULT ON)
endif()
option(HELFEM_EIGEN_BLAS
       "Route Eigen dense products through the external BLAS (needs an LP64 BLAS)"
       ${HELFEM_EIGEN_BLAS_DEFAULT})
if(HELFEM_EIGEN_BLAS AND HELFEM_ILP64)
    message(FATAL_ERROR
        "HELFEM_EIGEN_BLAS needs a 32-bit-integer (LP64) BLAS, but HELFEM_ILP64 "
        "is ON. Eigen's BLAS backend is int32-only and HelFEM does not patch "
        "Eigen. Reconfigure with -DHELFEM_ILP64=OFF to use it, or leave "
        "HELFEM_EIGEN_BLAS OFF to keep Eigen's own kernels under ILP64.")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Armadillo -- headers (and, on most installs, an `armadillo` imported
# target). Version >= 9 is required.
find_package(Armadillo 9 REQUIRED)
message(STATUS "Found Armadillo ${ARMADILLO_VERSION_STRING}")

# One INTERFACE target carries everything an Armadillo-consuming HelFEM
# target needs: the headers, the BLAS/LAPACK linkage (or the
# no-wrapper macro when the parent provides BLAS), and the integer-width
# definitions. libhelfem and helfem-common link it PUBLIC so both the
# in-tree build and downstream consumers inherit a consistent ABI.
add_library(helfem_arma INTERFACE)
add_library(helfem::arma ALIAS helfem_arma)
target_include_directories(helfem_arma INTERFACE "${ARMADILLO_INCLUDE_DIRS}")
target_compile_definitions(helfem_arma INTERFACE ARMA_64BIT_WORD)
if(HELFEM_ILP64)
    target_compile_definitions(helfem_arma INTERFACE ARMA_BLAS_LONG)
endif()

if(HELFEM_BLAS_PROVIDED_EXTERNALLY)
    # Parent resolves BLAS symbols at its final link; do not wrap through
    # libarmadillo and do not look for a BLAS here. Armadillo is used
    # header-only (the wrapper2_* shims are skipped).
    target_compile_definitions(helfem_arma INTERFACE ARMA_DONT_USE_WRAPPER)
    message(STATUS "HELFEM_BLAS_PROVIDED_EXTERNALLY=ON: Armadillo headers only, no BLAS/LAPACK linked")
else()
    # Request the integer width the rest of the build assumes before
    # locating BLAS, so e.g. flexiblas64 is preferred under ILP64.
    if(HELFEM_ILP64)
        set(BLA_SIZEOF_INTEGER 8)
    endif()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    # libarmadillo provides the wrapper2_* BLAS/LAPACK shims that
    # Armadillo's default (wrapper) build calls; the actual BLAS/LAPACK
    # back them. Both must be on the link line.
    target_link_libraries(helfem_arma INTERFACE
        ${ARMADILLO_LIBRARIES} BLAS::BLAS LAPACK::LAPACK)
    message(STATUS "BLAS: ${BLAS_LIBRARIES}")
endif()

install(TARGETS helfem_arma EXPORT helfemTargets)

# Eigen3 (>= 3.4) -- header-only. Prefer a system install, else fetch
# v3.4.0. The range form `3.4...99` accepts any >= 3.4 (a bare `3.4`
# is read as the exclusive [3.4, 3.5) by Eigen's ConfigVersion and would
# reject e.g. Fedora's Eigen 5). The system Eigen is used verbatim -- no
# patching -- which is why HELFEM_EIGEN_BLAS is restricted to LP64, where
# stock Eigen's 32-bit BLAS backend already matches the linked BLAS.
find_package(Eigen3 3.4...99 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 >= 3.4 not found; fetching v3.4.0")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0)
    set(EIGEN_BUILD_DOC       OFF CACHE INTERNAL "")
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
    set(BUILD_TESTING         OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(Eigen3)
else()
    message(STATUS "Using system Eigen3 ${Eigen3_VERSION}")
endif()

# Turn on Eigen's BLAS backend for EVERY translation unit at once. This
# MUST be global: if some TUs saw EIGEN_USE_BLAS and others did not, their
# Eigen product templates would differ and silently violate the ODR. The
# BLAS/LAPACK libraries are already on the link line via helfem::arma, so
# no extra linkage is needed -- only the define. Placed before
# add_subdirectory() so all in-tree targets inherit it. Guarded above to be
# LP64-only, so stock Eigen's int32 BLAS backend matches the linked BLAS.
if(HELFEM_EIGEN_BLAS)
    add_compile_definitions(EIGEN_USE_BLAS)
    message(STATUS "Eigen dense products routed through the external BLAS (EIGEN_USE_BLAS, LP64)")
endif()

# Wigner-nj symbols (>= 0.5.0). System install preferred, else fetch.
find_package(wignernj 0.5.0 QUIET)
if(NOT wignernj_FOUND)
    message(STATUS "wignernj >= 0.5.0 not found; fetching v0.5.0")
    include(FetchContent)
    FetchContent_Declare(wignernj
        GIT_REPOSITORY https://github.com/susilehtola/wignernj.git
        GIT_TAG        v0.5.0)
    FetchContent_MakeAvailable(wignernj)
    if(NOT TARGET wignernj::wignernj)
        add_library(wignernj::wignernj ALIAS wignernj)
    endif()
else()
    message(STATUS "Using system wignernj ${wignernj_VERSION}")
endif()

# OpenOrbitalOptimizer -- header-only SCF-convergence library. System
# install preferred, else fetch.
find_package(OpenOrbitalOptimizer QUIET)
if(NOT OpenOrbitalOptimizer_FOUND)
    message(STATUS "OpenOrbitalOptimizer not found; fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(OpenOrbitalOptimizer
        GIT_REPOSITORY https://github.com/susilehtola/OpenOrbitalOptimizer.git
        GIT_TAG        master)
    FetchContent_MakeAvailable(OpenOrbitalOptimizer)
else()
    message(STATUS "Using system OpenOrbitalOptimizer")
endif()

# OpenMP -- consumed as the OpenMP::OpenMP_CXX imported target, which
# carries the right flags on every compiler (crucially AppleClang, which
# needs -Xpreprocessor -fopenmp plus libomp rather than a bare -fopenmp).
if(USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    message(STATUS "OpenMP enabled (${OpenMP_CXX_VERSION})")
endif()

# HDF5 (C++) and libxc are only needed by the executables under src/.
if(HELFEM_BINARIES)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    message(STATUS "HDF5 ${HDF5_VERSION}")
    # Wrap HDF5 in one stable helfem-owned interface so the exported
    # target set references a name we control. FindHDF5's imported
    # targets vary across versions and some do not chain the C library
    # behind the C++ one (missing H5check_version at link), so use the
    # HDF5_CXX_LIBRARIES variable, which FindHDF5 populates with the full
    # C++ + C library set.
    add_library(helfem_hdf5 INTERFACE)
    add_library(helfem::hdf5 ALIAS helfem_hdf5)
    target_include_directories(helfem_hdf5 INTERFACE ${HDF5_CXX_INCLUDE_DIRS})
    target_link_libraries(helfem_hdf5 INTERFACE ${HDF5_CXX_LIBRARIES})
    install(TARGETS helfem_hdf5 EXPORT helfemTargets)

    # libxc ships pkg-config metadata (.pc) rather than a CMake package on
    # most installs; pkg_check_modules honours CMAKE_PREFIX_PATH so a
    # non-standard libxc prefix is found the same way as everything else.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBXC REQUIRED IMPORTED_TARGET libxc)
    message(STATUS "libxc ${LIBXC_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
add_subdirectory(lib1dfem)
add_subdirectory(libhelfem)
if(HELFEM_PYTHON)
    add_subdirectory(python)
endif()
if(HELFEM_BINARIES)
    add_subdirectory(src)
endif()

# ---------------------------------------------------------------------------
# Export an installed CMake package
# ---------------------------------------------------------------------------
# Consumers then do
#     find_package(helfem CONFIG REQUIRED)
#     target_link_libraries(mytarget PRIVATE helfem::helfem)
include(CMakePackageConfigHelpers)

if(HELFEM_BINARIES)
    set(HELFEM_HAS_HDF5 TRUE)
else()
    set(HELFEM_HAS_HDF5 FALSE)
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/helfemConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/helfemConfig.cmake"
    INSTALL_DESTINATION lib/cmake/helfem)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/helfemConfigVersion.cmake"
    VERSION "0.0.1"
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/helfemConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/helfemConfigVersion.cmake"
    DESTINATION lib/cmake/helfem)

install(EXPORT helfemTargets
    NAMESPACE helfem::
    DESTINATION lib/cmake/helfem)
