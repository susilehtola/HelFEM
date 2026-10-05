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

# Is HelFEM the top-level project, or is it being consumed by another build?
# PROJECT_IS_TOP_LEVEL would say, but it arrived in CMake 3.21 and we require
# 3.20, so derive it. Used to keep choices that are reasonable for a standalone
# build from being imposed on a parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HELFEM_IS_TOP_LEVEL TRUE)
else()
    set(HELFEM_IS_TOP_LEVEL FALSE)
endif()

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
# - HELFEM_USE_OPENMP: build the OpenMP-parallel version. On macOS this
#   needs a real OpenMP runtime (`brew install libomp`); AppleClang ships
#   none by default. If find_package(OpenMP) fails the build errors with a
#   hint. (Formerly USE_OPENMP, still honoured for a standalone build.)
#
# - HELFEM_BINARIES: build the executables under src/ (atomic, diatomic,
#   gensap, ...). OFF builds only libhelfem, avoiding the HDF5 and libxc
#   dependencies.
#
# - HELFEM_PYTHON: build the pybind11 `_helfem` module.
#
# - HELFEM_BUILD_TESTS: build and register the test suite. Defaults to ON for
#   a standalone build and OFF when HelFEM is consumed, so a parent project
#   neither has enable_testing() called on its behalf nor inherits HelFEM's
#   test names in its own ctest.
#
# - HELFEM_EIGEN_BLAS: route Eigen's dense products through an external
#   BLAS (see the option definition below). ON by default, with a
#   graceful fallback: if no usable LP64 BLAS is found the build simply
#   uses Eigen's own header-only kernels and links no BLAS at all.
#   (There is no ILP64 knob any more: Eigen indexes with a 64-bit type
#   unconditionally, and the optional BLAS backend requires a 32-bit
#   LP64 BLAS.)
# USE_OPENMP was the historical spelling. An unprefixed name in the cache is a
# collision waiting to happen once HelFEM is consumed by another build -- the
# parent may well have its own USE_OPENMP, meaning something else entirely.
#
# The old name is still honoured, but ONLY for a standalone HelFEM build. Inside
# a consuming project it is deliberately ignored: honouring it there would
# reintroduce exactly the collision this rename exists to remove.
set(_helfem_openmp_default ON)
if(HELFEM_IS_TOP_LEVEL AND DEFINED USE_OPENMP AND NOT DEFINED HELFEM_USE_OPENMP)
    message(DEPRECATION
        "USE_OPENMP is deprecated and will be removed; use HELFEM_USE_OPENMP. "
        "Honouring USE_OPENMP=${USE_OPENMP} for this standalone build.")
    set(_helfem_openmp_default ${USE_OPENMP})
endif()
option(HELFEM_USE_OPENMP "Build the OpenMP-parallel version" ${_helfem_openmp_default})
unset(_helfem_openmp_default)
option(HELFEM_BINARIES "Build the HelFEM executables under src/" ON)
option(HELFEM_PYTHON "Build the HelFEM Python (pybind11) bindings" OFF)

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
include(CheckCXXSourceCompiles)

# Whether the toolchain can actually build the quad path, tested by
# compiling what the code uses rather than by asking whether the compiler
# advertises C++23.
#
# Those are not the same question, and the difference is a real bug: GCC 11
# advertises cxx_std_23 (it has partial C++23 via -std=c++2b) while its
# libstdc++ provides none of the _Float128 support. Defaulting ON there gave
# hundreds of errors inside libhelfem -- missing operator""f128, ambiguous
# std::pow/abs/erfc against the legacy __float128 overloads, no
# numeric_limits -- with nothing to tell the user that a default had enabled
# an optional feature they never asked for. The feature test arrives a
# release or two before the feature.
#
# Each line below corresponds to one of those failure modes, so a toolchain
# that passes this compiles the quad instantiation.
set(_helfem_f128_probe "
#include <cmath>
#include <limits>
int main() {
  _Float128 x = 1.0f128;               // operator\"\"f128
  int n = 2;
  x = std::pow(x, n);                  // ambiguous against __float128 overloads
  x = std::abs(x);
  x = std::erfc(x);
  x = std::exp(x);
  return std::numeric_limits<_Float128>::is_specialized ? 0 : 1;
}
")
set(HELFEM_FLOAT128_DEFAULT OFF)
if(cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  # Raise the standard for the probe by setting CMAKE_CXX_STANDARD, NOT by
  # pushing -std=c++23 into CMAKE_REQUIRED_FLAGS. try_compile() appends the
  # flag for CMAKE_CXX_STANDARD *after* CMAKE_REQUIRED_FLAGS, so the latter
  # loses to whatever standard the surrounding project has set: embedded in
  # a project that asks for C++17, the probe compiled as
  # "-std=c++23 -std=c++17", C++17 won, 1.0f128 did not exist, and quad was
  # silently disabled on a toolchain that supports it perfectly. With
  # HELFEM_FLOAT128=ON cached that turned into the FATAL_ERROR below. A
  # standalone HelFEM build never saw it, because it sets no
  # CMAKE_CXX_STANDARD of its own and there was nothing to append.
  if(DEFINED CMAKE_CXX_STANDARD)
    set(_helfem_f128_had_std TRUE)
    set(_helfem_f128_saved_std "${CMAKE_CXX_STANDARD}")
  else()
    set(_helfem_f128_had_std FALSE)
  endif()
  set(_helfem_f128_saved_req "${CMAKE_CXX_STANDARD_REQUIRED}")
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  check_cxx_source_compiles("${_helfem_f128_probe}" HELFEM_FLOAT128_WORKS)
  if(_helfem_f128_had_std)
    set(CMAKE_CXX_STANDARD "${_helfem_f128_saved_std}")
  else()
    unset(CMAKE_CXX_STANDARD)
  endif()
  set(CMAKE_CXX_STANDARD_REQUIRED "${_helfem_f128_saved_req}")
  if(HELFEM_FLOAT128_WORKS)
    set(HELFEM_FLOAT128_DEFAULT ON)
  else()
    message(STATUS
      "Quad precision default OFF: the compiler advertises C++23 but its "
      "standard library does not provide the _Float128 support the quad "
      "path needs (literal suffix, <cmath> overloads, numeric_limits). "
      "GCC 11 is the common case. Everything builds at double and long "
      "double; -DHELFEM_FLOAT128=ON to override.")
  endif()
endif()
option(HELFEM_FLOAT128 "Instantiate the templated core at _Float128 (needs C++23)" ${HELFEM_FLOAT128_DEFAULT})

# An explicit request on a toolchain that cannot do it fails here, with a
# reason, instead of a wall of errors a thousand lines into libhelfem.
if(HELFEM_FLOAT128 AND DEFINED HELFEM_FLOAT128_WORKS AND NOT HELFEM_FLOAT128_WORKS)
  message(FATAL_ERROR
    "HELFEM_FLOAT128=ON was requested, but this toolchain cannot compile the "
    "quad path: the probe using the f128 literal suffix, std::pow/abs/erfc on "
    "_Float128 and std::numeric_limits<_Float128> does not build. Advertising "
    "C++23 is not enough -- GCC 11 does, without providing any of it. "
    "Configure with -DHELFEM_FLOAT128=OFF for double and long double.")
endif()
# Route Eigen's dense products (GEMM/GEMV/SYRK/...) through an external
# BLAS via EIGEN_USE_BLAS, instead of Eigen's own kernels. Vendor BLAS
# libraries are typically faster for large dense products, and this makes
# Eigen's results reproduce the external BLAS bit-for-bit. It requires a
# 32-bit-integer (LP64) BLAS -- stock Eigen's BLAS backend is int32-only
# and we do NOT patch Eigen -- so the BLAS found below is an ordinary LP64
# one (e.g. libflexiblas, reference BLAS, LP64 OpenBLAS/MKL). ON by
# default because vendor libraries are the sane performance default; if
# no LP64 BLAS is found the configure falls back to Eigen's header-only
# kernels with a STATUS notice instead of failing, so BLAS never becomes
# a hard dependency. Set OFF to force the header-only kernels (also the
# reproducible-across-machines choice).
option(HELFEM_EIGEN_BLAS
       "Route Eigen dense products through an external LP64 BLAS (EIGEN_USE_BLAS)"
       ON)

# Route Eigen's dense DECOMPOSITIONS through LAPACK via the LAPACKE C
# interface. Separate from HELFEM_EIGEN_BLAS above and OFF by default:
# measured, Eigen's own header-only solvers beat LAPACKE by 2-2.5x at
# every matrix size tried, and the SCF spends most of its time in them.
# See the detailed numbers at the detection site below.
option(HELFEM_EIGEN_LAPACKE
       "Route Eigen dense decompositions through LAPACKE (EIGEN_USE_LAPACKE)"
       OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Optional external BLAS for Eigen's dense products. The linear algebra is
# Eigen (header-only), so the default build links no BLAS/LAPACK at all.
# When HELFEM_EIGEN_BLAS is on, this INTERFACE target puts an LP64
# BLAS/LAPACK on the link line to back EIGEN_USE_BLAS; otherwise it is
# empty. libhelfem links it PUBLIC so consumers inherit the same linkage.
add_library(helfem_blas INTERFACE)
add_library(helfem::blas ALIAS helfem_blas)
set(HELFEM_EIGEN_BLAS_ACTIVE OFF)
set(HELFEM_EIGEN_LAPACKE_ACTIVE OFF)
if(HELFEM_EIGEN_BLAS)
    # Eigen's BLAS backend is int32-only; ask FindBLAS for an LP64
    # interface explicitly. (BLA_SIZEOF_INTEGER is honoured from CMake
    # 3.22; older CMake ignores it and we get the default search.)
    set(BLA_SIZEOF_INTEGER 4)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(helfem_blas INTERFACE BLAS::BLAS LAPACK::LAPACK)
        set(HELFEM_EIGEN_BLAS_ACTIVE ON)
        message(STATUS "BLAS (for EIGEN_USE_BLAS): ${BLAS_LIBRARIES}")
        # Eigen can also route its dense decompositions -- including the
        # SelfAdjointEigenSolver used in every SCF iteration -- to LAPACK,
        # but only through the LAPACKE C interface (EIGEN_USE_LAPACKE).
        # LAPACKE is a separate library/header from LAPACK itself, so
        # detect it separately and degrade to Eigen's own decompositions
        # when absent. Same LP64 constraint: lapack_int is int32 unless
        # LAPACKE was built ILP64, which the LP64 LAPACK above rules out.
        #
        # Off by default, because measured on this workload it is a
        # pessimisation rather than an optimisation. Same Eigen call, same
        # matrix, only the backend differing (ms, one core):
        #
        #   n     real symmetric        complex Hermitian
        #         LAPACKE -> Eigen      LAPACKE -> Eigen
        #   100      4.11 -> 1.71          8.81 -> 3.70
        #   308     62.62 -> 29.17       186.45 -> 65.74
        #   600    385.59 -> 196.33      1528   -> 675
        #   1000   2032   -> 1143        7499   -> 2847
        #
        # Eigen's own solvers win by 2-2.5x at every size tried. They are
        # header-only and get inlined and vectorised into the caller,
        # while the LAPACKE route pays an out-of-line call per operation
        # into a scalar-heavy tridiagonal QR whose rotation kernels
        # (dlasr/zlasr) do not thread. Since the SCF spends ~44% of an
        # atom-in-jellium run inside exactly those kernels, the default
        # matters. Turn it on with -DHELFEM_EIGEN_LAPACKE=ON if a
        # particular LAPACK is faster on some machine -- measure first.
        #
        # This is independent of EIGEN_USE_BLAS above, which stays on:
        # the vendor GEMM is genuinely faster than Eigen's own, and the
        # two switches control different things.
        if(HELFEM_EIGEN_LAPACKE)
        find_library(LAPACKE_LIBRARY NAMES lapacke)
        find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)
        endif()
        if(HELFEM_EIGEN_LAPACKE AND LAPACKE_LIBRARY AND LAPACKE_INCLUDE_DIR)
            target_link_libraries(helfem_blas INTERFACE ${LAPACKE_LIBRARY})
            target_include_directories(helfem_blas INTERFACE ${LAPACKE_INCLUDE_DIR})
            set(HELFEM_EIGEN_LAPACKE_ACTIVE ON)
            message(STATUS "LAPACKE (for EIGEN_USE_LAPACKE): ${LAPACKE_LIBRARY}")
        elseif(HELFEM_EIGEN_LAPACKE)
            message(STATUS "LAPACKE requested but not found; Eigen's own decompositions stay in use")
        else()
            message(STATUS "Eigen's own dense decompositions in use "
                           "(2-2.5x faster than LAPACKE here; -DHELFEM_EIGEN_LAPACKE=ON to override)")
        endif()
    else()
        message(STATUS "No LP64 BLAS/LAPACK found; using Eigen's built-in kernels "
                       "(set CMAKE_PREFIX_PATH to point at a BLAS, or -DHELFEM_EIGEN_BLAS=OFF to silence)")
    endif()
endif()

install(TARGETS helfem_blas EXPORT helfemTargets)

# ---------------------------------------------------------------------------
# Fetched dependencies install into a HelFEM-private subtree
# ---------------------------------------------------------------------------
# Any dependency HelFEM fetches must ship with an installed HelFEM, because
# helfemTargets.cmake references its targets: leave one out and
# find_package(helfem) fails for every consumer of the install. But it must
# NOT install where find_package looks by default, or the next configure
# against the same prefix -- ./compile.sh, say -- finds that installed copy
# instead of fetching, and the dependency silently freezes at whatever
# snapshot was installed first, however the pin in this file moves.
#
# Before this, each fetch site traded one of those for the other: OOO and
# OpenTrustRegion were excluded from the install (so an installed HelFEM was
# unusable by any consumer), while Eigen, wignernj and libxc installed at
# their default locations (so they froze).
#
# Both are met by installing into <prefix>/${HELFEM_DEPS_SUBDIR}: none of
# find_package's default search patterns reach it, and helfemConfig.cmake
# looks there explicitly with NO_DEFAULT_PATH. Every one of these projects
# takes its install locations from GNUInstallDirs, so pointing the three
# relevant variables at the private tree for the duration of the fetch
# covers them all without knowing each one's own variable names -- which
# matters, since libxc for one sets its config directory with a plain set()
# that cannot be overridden.
set(HELFEM_DEPS_SUBDIR "lib/helfem-deps")
foreach(_dep Eigen3 wignernj OpenOrbitalOptimizer OpenTrustRegion libxc)
    set(HELFEM_FETCHED_${_dep} FALSE)
endforeach()
# Create the GNUInstallDirs cache entries now. A subproject whose own
# include(GNUInstallDirs) CREATES one would, under the pre-3.21 behaviour of
# policy CMP0126 its cmake_minimum_required may select, drop the normal
# variable set below and install to the default location after all.
include(GNUInstallDirs)
#
# Fetched dependencies are also built STATIC and position-independent. They
# are private implementation details installed off the search path, and a
# shared library there would need a runtime path managed on every node it
# runs on. More to the point, wignernj, libxc and OpenTrustRegion each
# declare BUILD_SHARED_LIBS as their own option() -- defaulting to ON, ON
# and OFF -- and option() writes a GLOBAL cache entry. Left alone, whichever
# was fetched first fixed the library type for every later one AND for the
# project including HelFEM: fetching wignernj made OpenTrustRegion shared,
# and would have made a consumer's own libraries shared too.
#
# So the macro takes, after the name, the cache variables a dependency is
# known to create under names it does not own. Any of them that did not
# exist before the fetch is removed after it -- the dependency re-creates it
# on every configure, so nothing is lost -- while one that DID exist, say an
# explicit -DBUILD_SHARED_LIBS from whoever is building, is left untouched.
#
# After the keyword INSTALL_DIRS come the dependency's CACHED install
# locations -- OpenOrbitalOptimizer_INSTALL_CMAKEDIR, Eigen's
# INCLUDE_INSTALL_DIR and friends -- which it derives from
# CMAKE_INSTALL_<dir> the first time and then reads back from the cache.
# Those are cleared BEFORE the fetch as well as after it, unconditionally.
# Otherwise a value cached once wins over the private layout set below: a
# build tree first configured before that layout existed kept installing
# OpenOrbitalOptimizer's package config to lib64/cmake, where
# helfemConfig.cmake never looks, and find_package(helfem) failed for every
# consumer of the install. Overriding one of these by hand is not supported:
# a fetched dependency's install location is HelFEM's to choose.
macro(helfem_fetch_privately _dep)
    cmake_parse_arguments(_helfem_fp "" "" "INSTALL_DIRS" ${ARGN})
    set(_helfem_fp_scrub BUILD_SHARED_LIBS ${_helfem_fp_UNPARSED_ARGUMENTS})
    foreach(_v IN LISTS _helfem_fp_INSTALL_DIRS)
        unset(${_v} CACHE)
        unset(${_v})
    endforeach()
    foreach(_v IN LISTS _helfem_fp_scrub)
        if(DEFINED CACHE{${_v}})
            set(_helfem_fp_hadcache_${_v} TRUE)
        else()
            set(_helfem_fp_hadcache_${_v} FALSE)
        endif()
    endforeach()
    # Normal variables, which a project including HelFEM may have set in the
    # scope it handed us: saved exactly and put back exactly.
    foreach(_v BUILD_SHARED_LIBS CMAKE_POSITION_INDEPENDENT_CODE
               CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_DATADIR CMAKE_INSTALL_INCLUDEDIR
               CMAKE_EXPORT_NO_PACKAGE_REGISTRY CMAKE_EXPORT_PACKAGE_REGISTRY)
        if(DEFINED ${_v})
            set(_helfem_fp_had_${_v} TRUE)
            set(_helfem_fp_saved_${_v} "${${_v}}")
        else()
            set(_helfem_fp_had_${_v} FALSE)
        endif()
    endforeach()
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INSTALL_LIBDIR     "${HELFEM_DEPS_SUBDIR}/lib")
    set(CMAKE_INSTALL_DATADIR    "${HELFEM_DEPS_SUBDIR}/share")
    set(CMAKE_INSTALL_INCLUDEDIR "${HELFEM_DEPS_SUBDIR}/include")
    # No export(PACKAGE) into the user package registry (~/.cmake/packages).
    # Eigen's does, under its cmake_minimum_required(3.10) policies, and the
    # entry names THIS build tree: every later configure on the machine, of
    # any project, then finds the fetched Eigen there instead of fetching --
    # the same freeze as installing it to a default location, only harder to
    # see, and dangling once the build tree is deleted. Both spellings, since
    # which one applies depends on the subproject's policy CMP0090.
    set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
    set(CMAKE_EXPORT_PACKAGE_REGISTRY OFF)
    FetchContent_MakeAvailable(${_dep})
    foreach(_v BUILD_SHARED_LIBS CMAKE_POSITION_INDEPENDENT_CODE
               CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_DATADIR CMAKE_INSTALL_INCLUDEDIR
               CMAKE_EXPORT_NO_PACKAGE_REGISTRY CMAKE_EXPORT_PACKAGE_REGISTRY)
        if(_helfem_fp_had_${_v})
            set(${_v} "${_helfem_fp_saved_${_v}}")
        else()
            unset(${_v})
        endif()
        unset(_helfem_fp_had_${_v})
        unset(_helfem_fp_saved_${_v})
    endforeach()
    foreach(_v IN LISTS _helfem_fp_scrub)
        if(NOT _helfem_fp_hadcache_${_v})
            unset(${_v} CACHE)
        endif()
        unset(_helfem_fp_hadcache_${_v})
    endforeach()
    foreach(_v IN LISTS _helfem_fp_INSTALL_DIRS)
        unset(${_v} CACHE)
    endforeach()
    unset(_helfem_fp_scrub)
    unset(_helfem_fp_INSTALL_DIRS)
    unset(_helfem_fp_UNPARSED_ARGUMENTS)
    set(HELFEM_FETCHED_${_dep} TRUE)
endmacro()

# Eigen3 (>= 3.4) -- header-only. Prefer a system install, else fetch the
# latest stable release (5.0.1). The system Eigen is used verbatim -- no
# patching -- which is why HELFEM_EIGEN_BLAS is restricted to LP64, where
# stock Eigen's 32-bit BLAS backend already matches the linked BLAS.
#
# TWO REQUESTS, because no single version argument admits both series.
# Eigen 3.4's ConfigVersion declares SameMajorVersion, under which a bare
# `3.4` accepts every 3.x >= 3.4, but a range is accepted only if BOTH ends
# have major version 3 -- so `3.4...99` rejects every Eigen 3, and on RHEL
# 10 (EPEL eigen3 3.4.0) the configure reported "Eigen3 >= 3.4 not found"
# and fetched 5.0.1 over the installed package. Eigen 5's ConfigVersion
# reads a bare `3.4` as the exclusive [3.4, 3.5) and so needs the range.
# A version-rejected find_package loads no targets, so the second request
# starts clean.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 3.4...99 QUIET NO_MODULE)
endif()
if(NOT Eigen3_FOUND)
    # One place for the fetched version: it names the tag, the status
    # line and the version the shim config below reports.
    set(_eigen_fetch_version 5.0.1)
    message(STATUS "Eigen3 >= 3.4 not found; fetching v${_eigen_fetch_version}")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        ${_eigen_fetch_version})
    set(EIGEN_BUILD_DOC       OFF CACHE INTERNAL "")
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
    # BUILD_TESTING is CTest's global switch, not Eigen's: it has to be a
    # cache entry for Eigen's option() to honour it, but it must not outlive
    # Eigen's configure, or an including project's include(CTest) finds it
    # already OFF and never registers its own tests. Removed below unless
    # whoever is building had set it.
    if(DEFINED CACHE{BUILD_TESTING})
        set(_helfem_had_build_testing TRUE)
    else()
        set(_helfem_had_build_testing FALSE)
    endif()
    set(BUILD_TESTING         OFF CACHE INTERNAL "")
    # Eigen assigns its target to the Eigen3Targets export set
    # unconditionally but only *installs* that set when
    # EIGEN_BUILD_CMAKE_PACKAGE is on, which defaults to
    # PROJECT_IS_TOP_LEVEL and so is off when it is fetched. Left off, the
    # generate step fails -- install(EXPORT helfemTargets) includes helfem,
    # which requires eigen, "that is not in any export set" -- and an
    # installed HelFEM would in any case be unusable, since its public
    # headers include <Eigen/...> and nothing would have installed them.
    # If we vendor Eigen we have to ship it.
    set(EIGEN_BUILD_CMAKE_PACKAGE ON CACHE INTERNAL "")
    helfem_fetch_privately(Eigen3
        INSTALL_DIRS INCLUDE_INSTALL_DIR CMAKEPACKAGE_INSTALL_DIR PKGCONFIG_INSTALL_DIR)
    if(NOT _helfem_had_build_testing)
        unset(BUILD_TESTING CACHE)
    endif()
    unset(_helfem_had_build_testing)

    # Eigen's *build* tree exports no config package -- only its install
    # tree does -- so a nested find_package(Eigen3 NO_MODULE) fails even
    # though Eigen3::Eigen is by now a perfectly good target.
    # OpenOrbitalOptimizer's CMakeLists does exactly that, so with neither
    # Eigen3 nor OOO installed the configure died in OOO with "Could not
    # find a package configuration file provided by Eigen3" -- while
    # HelFEM had just fetched it. Stand up a shim config pointing at the
    # copy we fetched and aim Eigen3_DIR at it, so any dependency that
    # asks for Eigen3 the normal way gets that one rather than failing or,
    # worse, picking up an unrelated system copy.
    set(_eigen_shim "${CMAKE_CURRENT_BINARY_DIR}/eigen3-config-shim")
    file(WRITE "${_eigen_shim}/Eigen3Config.cmake" "\
# Generated by HelFEM: redirects find_package(Eigen3) to the copy fetched
# by the top-level project. Do not edit; regenerated on every configure.
if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR \"Eigen3::Eigen missing: the shim is only valid \"
                      \"inside the build that fetched Eigen.\")
endif()
set(EIGEN3_INCLUDE_DIR \"${eigen3_SOURCE_DIR}\")
set(EIGEN3_INCLUDE_DIRS \"${eigen3_SOURCE_DIR}\")
set(EIGEN3_VERSION_STRING \"${_eigen_fetch_version}\")
set(Eigen3_FOUND TRUE)
")
    # Version file so that a nested find_package(Eigen3 <version>) -- with
    # or without a range -- is answered rather than silently rejected.
    file(WRITE "${_eigen_shim}/Eigen3ConfigVersion.cmake" "\
set(PACKAGE_VERSION \"${_eigen_fetch_version}\")
if(PACKAGE_FIND_VERSION_RANGE)
  if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION_MIN AND
     ((PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL \"INCLUDE\" AND
       PACKAGE_VERSION VERSION_LESS_EQUAL PACKAGE_FIND_VERSION_MAX) OR
      (PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL \"EXCLUDE\" AND
       PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MAX)))
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()
elseif(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
")
    set(Eigen3_DIR "${_eigen_shim}" CACHE PATH
        "Eigen3 config directory (shim for the FetchContent copy)" FORCE)
    unset(_eigen_shim)
    unset(_eigen_fetch_version)
else()
    message(STATUS "Using system Eigen3 ${Eigen3_VERSION}")
endif()

# Turn on Eigen's BLAS backend for EVERY translation unit at once. This
# MUST be global: if some TUs saw EIGEN_USE_BLAS and others did not, their
# Eigen product templates would differ and silently violate the ODR. The
# BLAS/LAPACK come in via the helfem::blas target above; here we only add
# the define. Placed before add_subdirectory() so all in-tree targets
# inherit it. LP64-only, so stock Eigen's int32 BLAS backend matches.
if(HELFEM_EIGEN_BLAS_ACTIVE)
    add_compile_definitions(EIGEN_USE_BLAS)
    message(STATUS "Eigen dense products routed through the external BLAS (EIGEN_USE_BLAS, LP64)")
endif()
if(HELFEM_EIGEN_LAPACKE_ACTIVE)
    add_compile_definitions(EIGEN_USE_LAPACKE)
    message(STATUS "Eigen dense decompositions routed through LAPACKE (EIGEN_USE_LAPACKE, LP64)")
endif()

# Wigner-nj symbols (>= 0.7.0). System install preferred, else fetch.
# 0.7.0 is the minimum for the templated / quadmath gaunt<T> API the
# arbitrary-precision (quad) Gaunt path uses (gaunt.cpp:
# wignernj::gaunt<long double>, wignernj_quadmath.h). The fetch fallback
# pulls the current newest release (v0.8.0).
find_package(wignernj 0.7.0 QUIET)
if(NOT wignernj_FOUND)
    message(STATUS "wignernj >= 0.7.0 not found; fetching v0.8.0")
    include(FetchContent)
    FetchContent_Declare(wignernj
        GIT_REPOSITORY https://github.com/susilehtola/wignernj.git
        GIT_TAG        v0.8.0)
    # gaunt.cpp calls the __float128 C entry point ::gaunt_q, which wignernj
    # only compiles when its quadmath build is enabled. In wignernj 0.8.0 that
    # option was renamed BUILD_QUADMATH -> WIGNERNJ_BUILD_QUADMATH and defaults
    # OFF, so it must be set explicitly here or ::gaunt_q fails to link.
    set(WIGNERNJ_BUILD_QUADMATH ON CACHE INTERNAL "")
    helfem_fetch_privately(wignernj)
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
    # Installed with HelFEM, but into the private subtree; see
    # helfem_fetch_privately. It used to be excluded from the install
    # altogether (EXCLUDE_FROM_ALL), which kept the snapshot from freezing
    # but left every installed HelFEM referencing a target no consumer could
    # find.
    # Pinned to an explicit commit, NOT to master. With a floating branch two
    # builds of identical HelFEM source can link different OOO and produce
    # different numbers, with nothing in the HelFEM history to explain it --
    # which makes `git bisect` meaningless, since every step samples a moving
    # dependency. Bump this deliberately, in its own commit, so the bump is
    # bisectable like any other change.
    #
    # bef468d is 105 commits past v0.4.0 and carries the Aufbau/occupation
    # fixes; pinning to the v0.4.0 tag instead would silently revert them.
    # It also fixes the cleanup() segfault on a single-entry orbital
    # history, which a one-electron ion reaches through the default method
    # set as soon as the first DIIS error exceeds 1 and DIIS is dropped.
    FetchContent_Declare(OpenOrbitalOptimizer
        GIT_REPOSITORY https://github.com/susilehtola/OpenOrbitalOptimizer.git
        GIT_TAG        bef468dfb24241ea6e44d73e2d749182d45dce16)
    helfem_fetch_privately(OpenOrbitalOptimizer
        INSTALL_DIRS OpenOrbitalOptimizer_INSTALL_CMAKEDIR)
else()
    message(STATUS "Using system OpenOrbitalOptimizer")
endif()

# OpenMP -- consumed as the OpenMP::OpenMP_CXX imported target, which
# carries the right flags on every compiler (crucially AppleClang, which
# needs -Xpreprocessor -fopenmp plus libomp rather than a bare -fopenmp).
if(HELFEM_USE_OPENMP)
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

    # An HDF5 outside the system prefixes -- copied into a home directory
    # so compute nodes can see it, say -- brings its own dependencies with
    # it, typically libaec/libsz for szip-compressed datasets. FindHDF5
    # names the HDF5 libraries by full path but does not always name those
    # transitive ones, so the linker is told what to link and never where
    # to look, and the build fails on a library sitting right next to one
    # it just found. Publishing the directories HDF5 itself was found in
    # covers that without anyone having to hand-edit link_directories().
    set(_h5dirs "")
    foreach(_h5lib IN LISTS HDF5_CXX_LIBRARIES)
        if(IS_ABSOLUTE "${_h5lib}")
            get_filename_component(_h5dir "${_h5lib}" DIRECTORY)
            list(APPEND _h5dirs "${_h5dir}")
        endif()
    endforeach()
    if(_h5dirs)
        list(REMOVE_DUPLICATES _h5dirs)
        target_link_directories(helfem_hdf5 INTERFACE ${_h5dirs})
        message(STATUS "HDF5 link directories: ${_h5dirs}")
    endif()
    unset(_h5dirs)
    unset(_h5dir)
    unset(_h5lib)

    install(TARGETS helfem_hdf5 EXPORT helfemTargets)

    # libxc, tried in order:
    #
    #   1. libxc's own CMake package (LibxcConfig.cmake, target Libxc::xc),
    #      which every CMake-built libxc installs;
    #   2. pkg-config metadata (libxc.pc), which the Autotools build and most
    #      distribution packages ship;
    #   3. with HELFEM_FETCH_LIBXC (default ON), fetching and building libxc
    #      HELFEM_LIBXC_FETCH_TAG from gitlab.com/libxc/libxc.
    #
    # Only (2) used to exist, and both of its steps were REQUIRED, so a
    # machine without a pkg-config binary -- common on HPC module systems --
    # failed at the first line even with libxc installed, and one without
    # libxc.pc failed at the second even with a CMake-built libxc on
    # CMAKE_PREFIX_PATH.
    #
    # Whichever supplies it is wrapped in helfem::libxc, so the exported
    # target set references a name HelFEM controls; helfemConfig.cmake
    # re-finds it the same way (HELFEM_LIBXC_PROVIDER records which).
    option(HELFEM_FETCH_LIBXC
           "Fetch and build libxc if no installed copy is found" ON)
    set(HELFEM_LIBXC_FETCH_TAG "7.1.2" CACHE STRING
        "libxc git tag to fetch when HELFEM_FETCH_LIBXC falls back to it")
    add_library(helfem_libxc INTERFACE)
    add_library(helfem::libxc ALIAS helfem_libxc)
    set(HELFEM_LIBXC_PROVIDER "")

    find_package(Libxc CONFIG QUIET)
    if(Libxc_FOUND AND TARGET Libxc::xc)
        target_link_libraries(helfem_libxc INTERFACE Libxc::xc)
        set(HELFEM_LIBXC_PROVIDER "config")
        message(STATUS "libxc ${Libxc_VERSION} (CMake package in ${Libxc_DIR})")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LIBXC QUIET IMPORTED_TARGET libxc)
        endif()
        if(LIBXC_FOUND)
            target_link_libraries(helfem_libxc INTERFACE PkgConfig::LIBXC)
            set(HELFEM_LIBXC_PROVIDER "pkgconfig")
            message(STATUS "libxc ${LIBXC_VERSION} (pkg-config)")
        elseif(HELFEM_FETCH_LIBXC)
            # libxc's own CMakeLists requires 3.21; HelFEM requires only 3.20.
            if(CMAKE_VERSION VERSION_LESS 3.21)
                message(FATAL_ERROR
                    "libxc was not found, and building it from source needs "
                    "CMake >= 3.21 (this is ${CMAKE_VERSION}). Install libxc "
                    "and put its prefix on CMAKE_PREFIX_PATH, or use a newer "
                    "CMake.")
            endif()
            message(STATUS "libxc not found; fetching ${HELFEM_LIBXC_FETCH_TAG} "
                           "from gitlab.com/libxc/libxc")
            include(FetchContent)
            # By git tag, not by archive: GitLab's generated tarballs are not
            # guaranteed byte-stable, so a pinned hash on one can break with
            # no upstream change. On a compute node without network access,
            # point FETCHCONTENT_SOURCE_DIR_LIBXC at a checkout instead.
            FetchContent_Declare(libxc
                GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
                GIT_TAG        ${HELFEM_LIBXC_FETCH_TAG}
                GIT_SHALLOW    TRUE)
            # libxc reads these as options/cache defaults; set them as plain
            # variables for its configure only, and restore ours after, so
            # nothing leaks into the including project.
            foreach(_v MAXORDER)
                if(DEFINED ${_v})
                    set(_helfem_saved_${_v} "${${_v}}")
                    set(_helfem_had_${_v} TRUE)
                else()
                    set(_helfem_had_${_v} FALSE)
                endif()
            endforeach()
            # Energy, potential and kernel: HelFEM calls the *_fxc family for
            # the exact XC response. 2 is libxc 7's default; pinned so a stray
            # cache value cannot drop the kernel.
            set(MAXORDER 2)
            # libxc's own options live under generic names -- ENABLE_FORTRAN
            # in particular would silently pre-empt an including project's
            # option(ENABLE_FORTRAN ... ON) -- so they are scoped to its
            # configure like BUILD_SHARED_LIBS.
            helfem_fetch_privately(libxc
                MAXORDER ENABLE_FORTRAN ENABLE_PYTHON BUILD_FPIC DISABLE_FHC
                XC_CHECK_NUMERICS XC_PROFILING NAMESPACE_INSTALL_INCLUDEDIR
                INSTALL_DIRS FORTRAN_MODULE_INSTALL_DIR)
            foreach(_v MAXORDER)
                if(_helfem_had_${_v})
                    set(${_v} "${_helfem_saved_${_v}}")
                else()
                    unset(${_v})
                endif()
                unset(_helfem_saved_${_v})
                unset(_helfem_had_${_v})
            endforeach()
            # libxc publishes its include directory only as INSTALL_INTERFACE,
            # so an in-tree consumer of `xc` gets no path to its headers.
            # Supply the build-tree locations: xc.h is rewritten into the
            # binary src/, xc_version.h is generated at the binary top, and
            # xc_funcs.h stays in the source src/.
            target_include_directories(helfem_libxc INTERFACE
                $<BUILD_INTERFACE:${libxc_BINARY_DIR}/src>
                $<BUILD_INTERFACE:${libxc_BINARY_DIR}>
                $<BUILD_INTERFACE:${libxc_SOURCE_DIR}/src>)
            # `xc` is installed by libxc's own rules, into the private
            # subtree (helfem_fetch_privately), and exported as Libxc::xc;
            # helfemConfig.cmake finds it there.
            target_link_libraries(helfem_libxc INTERFACE xc)
            set(HELFEM_LIBXC_PROVIDER "fetched")
        else()
            message(FATAL_ERROR
                "libxc not found: neither a CMake package (LibxcConfig.cmake) "
                "nor pkg-config metadata (libxc.pc). Put its prefix on "
                "CMAKE_PREFIX_PATH, or set -DHELFEM_FETCH_LIBXC=ON to build "
                "${HELFEM_LIBXC_FETCH_TAG} from source.")
        endif()
    endif()
    install(TARGETS helfem_libxc EXPORT helfemTargets)
endif()

# OpenTrustRegion -- second-order trust-region orbital optimizer, used by
# the atom-in-jellium driver where the fractional occupations at the Fermi
# level leave the first-order methods crawling. It is a Fortran library
# with a C interface, so it is strictly optional: without it the drivers
# build exactly as before, minus the --secondorder path.
#
# Only the LP64 (32-bit integer) build is usable. The C header picks its
# integer type from USE_ILP64 at OUR compile time, and we do not define
# that, so a 64-bit-integer library would disagree with the settings
# struct we compile against -- silently, since the symbol names are the
# same either way. helfem::otr therefore carries the LP64 library or
# nothing at all, and otr_solver.cpp additionally checks the library's
# own `ilp64` flag at run time.
option(HELFEM_OPENTRUSTREGION
       "Use OpenTrustRegion for second-order orbital optimization" ON)
add_library(helfem_otr INTERFACE)
add_library(helfem::otr ALIAS helfem_otr)
set(HELFEM_OPENTRUSTREGION_ACTIVE OFF)
if(HELFEM_OPENTRUSTREGION)
    find_package(OpenTrustRegion QUIET COMPONENTS int32)
    if(OpenTrustRegion_FOUND)
        target_link_libraries(helfem_otr INTERFACE OpenTrustRegion::opentrustregion)
        set(HELFEM_OPENTRUSTREGION_ACTIVE ON)
        message(STATUS "Using system OpenTrustRegion ${OpenTrustRegion_VERSION}")
    else()
        # Fetching means compiling Fortran, which HelFEM otherwise does not
        # need, and linking an LP64 BLAS/LAPACK, which OpenTrustRegion marks
        # REQUIRED. Both are checked here rather than left to fail inside the
        # subproject, so a machine without them degrades to a build without
        # the second-order path instead of a configure error.
        include(CheckLanguage)
        check_language(Fortran)
        set(BLA_SIZEOF_INTEGER 4)
        find_package(BLAS QUIET)
        find_package(LAPACK QUIET)
        if(CMAKE_Fortran_COMPILER AND BLAS_FOUND AND LAPACK_FOUND)
            message(STATUS "OpenTrustRegion not found; fetching from GitHub")
            enable_language(Fortran)
            enable_language(C)
            include(FetchContent)
            # Pinned, so two builds of identical HelFEM source cannot link
            # different optimizers, and installed privately with HelFEM; see
            # helfem_fetch_privately.
            # Force the LP64 build; OpenTrustRegion's own autodetection falls
            # back to ILP64 when it cannot find a 32-bit BLAS, which is the
            # one outcome we cannot consume.
            # INTEGER_SIZE is OpenTrustRegion's option, not ours, and it is
            # unprefixed -- so FORCEing it writes a name we do not own into
            # the consumer's cache, permanently and over any explicit -D they
            # set. HelFEM genuinely does need LP64 here (see above), so save
            # whatever was there and put it back once OpenTrustRegion has been
            # configured with the value it needs.
            if(DEFINED CACHE{INTEGER_SIZE})
                set(_helfem_saved_intsize "${INTEGER_SIZE}")
                set(_helfem_had_intsize TRUE)
            else()
                set(_helfem_had_intsize FALSE)
            endif()
            set(INTEGER_SIZE 4 CACHE STRING "OpenTrustRegion integer size" FORCE)
            set(OpenTrustRegion_BUILD_TESTING OFF CACHE INTERNAL "")
            FetchContent_Declare(OpenTrustRegion
                GIT_REPOSITORY https://github.com/eriksen-lab/opentrustregion.git
                GIT_TAG        v2.0.0)
            helfem_fetch_privately(OpenTrustRegion)
            # OpenTrustRegion's targets are configured by the call above, so the
            # integer width is already baked into them; restoring the cache now
            # cannot change what was built.
            if(_helfem_had_intsize)
                set(INTEGER_SIZE "${_helfem_saved_intsize}" CACHE STRING
                    "OpenTrustRegion integer size" FORCE)
            else()
                unset(INTEGER_SIZE CACHE)
            endif()
            unset(_helfem_saved_intsize)
            unset(_helfem_had_intsize)
            target_link_libraries(helfem_otr INTERFACE opentrustregion)
            set(HELFEM_OPENTRUSTREGION_ACTIVE ON)
        else()
            message(STATUS "OpenTrustRegion unavailable "
                           "(needs a Fortran compiler and an LP64 BLAS/LAPACK); "
                           "the second-order optimizer will not be built")
        endif()
    endif()
endif()
if(HELFEM_OPENTRUSTREGION_ACTIVE)
    target_compile_definitions(helfem_otr INTERFACE HELFEM_HAVE_OPENTRUSTREGION)
endif()
# Exported only alongside helfem-common, the sole consumer (otr_solver.cpp).
# helfem_otr's INTERFACE carries OpenTrustRegion::opentrustregion, which the
# export set does not define, so exporting it in a HELFEM_BINARIES=OFF install
# leaves find_package(helfem) referencing a target that does not exist.
if(HELFEM_BINARIES)
    install(TARGETS helfem_otr EXPORT helfemTargets)
endif()

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
add_subdirectory(libhelfem)
add_subdirectory(libhelfemqc)
if(HELFEM_PYTHON)
    add_subdirectory(python)
endif()
# src/ is entered unconditionally: it defines the dependency-light
# helfem-fem / legendre targets as well as the binaries, and gates the
# HDF5/libxc-dependent half on HELFEM_BINARIES itself.
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# enable_testing() has to be called from the top level for `ctest` and
# `make test` to find anything. Registration itself lives in tests/, which
# adds both the self-checking unit binaries and the Python integration
# harness. Plain `ctest` stays in the minutes range; the slow tier is
# opt-in via -DHELFEM_SLOW_TESTS=ON.
#
# Both of these are gated, because neither is HelFEM's to do unasked when it
# is consumed by another build. enable_testing() acts on the TOP-LEVEL project,
# not on us, so calling it from inside a consumer turns on test infrastructure
# they did not ask for; and add_subdirectory(tests) would register some two
# hundred HelFEM cases into their `ctest`, where the names would be
# indistinguishable from their own. Defaulting to HELFEM_IS_TOP_LEVEL leaves a
# standalone build exactly as it was.
option(HELFEM_BUILD_TESTS
       "Build and register HelFEM's own test suite"
       ${HELFEM_IS_TOP_LEVEL})
if(HELFEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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
