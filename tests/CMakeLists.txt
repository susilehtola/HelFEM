# CTest registration.
#
# Two kinds of test live here. The self-checking unit binaries built under
# src/ exit non-zero on failure and need no arguments, so they register
# directly. The integration suite is the Python harness in this directory,
# which compares converged energies against tests/refs/ci.json.
#
# --bindir is taken from $<TARGET_FILE_DIR:diatomic> rather than being
# spelled out, so it follows the build tree instead of being a path the
# caller has to remember (objdir/src vs install/bin vs a multi-config
# generator's per-config subdirectory).
#
# Plain `ctest` runs the unit tests and the quick integration tier -- the
# same set CI runs, minutes rather than hours. The slow tier is registered
# only under HELFEM_SLOW_TESTS, because CTest runs every registered test by
# default and a `make test` that silently takes hours is a `make test`
# nobody runs.

find_package(Python3 COMPONENTS Interpreter)

# ---------------------------------------------------------------- unit
# Self-contained checks with no reference data, exercising layers the
# integration suite reaches only indirectly.
#
# ONLY binaries that can actually fail are listed. Deliberately excluded:
#
#   atomic_itest          takes "nquad R" arguments and, whatever it
#                         measures, ends in `return 0` -- it prints
#                         "Error in inner integral is %e" and exits
#                         successfully no matter how large that error is.
#   atomdb_test           print-only, no failure path.
#   diatomic_purem_test   print-only, no failure path -- and 149 s of it.
#
# Registering those would report a pass that means nothing, which is
# worse than not testing them; turning them into real checks is a
# separate job (they need a tolerance and a non-zero exit).
foreach(t legendre_test legendre_unit_test gaunt_test sphtest over_r_test
          checkpoint_precision_test)
  if(TARGET ${t})
    add_test(NAME unit.${t} COMMAND ${t})
    set_tests_properties(unit.${t} PROPERTIES LABELS "unit")
  endif()
endforeach()

# --------------------------------------------------------- integration
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "Python3 not found: integration tests not registered")
  return()
endif()
if(NOT TARGET diatomic)
  message(STATUS "HELFEM_BINARIES=OFF: integration tests not registered")
  return()
endif()

set(HELFEM_TEST_DRIVER "${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py")
set(HELFEM_TEST_REFS   "${CMAKE_CURRENT_SOURCE_DIR}/refs/ci.json")

# One test per tag rather than one giant test, so a failure names the area
# and `ctest -R` can select it.
#
# --skip-slow is not optional here. Tags are orthogonal to cost: the
# symmetry tag covers both the seconds-long Ne/H2 cases and the CH/NH ones
# that run minutes each, so without it `ctest` would quietly inherit the
# slow tier through a tag that says nothing about runtime.
function(helfem_add_integration_test name tag)
  add_test(NAME ${name}
           COMMAND ${Python3_EXECUTABLE} ${HELFEM_TEST_DRIVER}
                   --check ${HELFEM_TEST_REFS}
                   --tag ${tag} --skip-slow
                   --bindir $<TARGET_FILE_DIR:diatomic>
                   --jobs ${HELFEM_TEST_JOBS})
  # The harness runs each case with OMP_NUM_THREADS=1 and parallelises
  # across cases, so converged energies do not depend on the job count --
  # see tests/README.md on thread-count reproducibility.
  set_tests_properties(${name} PROPERTIES
    LABELS "${ARGN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT ${HELFEM_TEST_TIMEOUT})
endfunction()

set(HELFEM_TEST_JOBS 4 CACHE STRING "Parallel cases for the integration tests")
set(HELFEM_TEST_TIMEOUT 5400 CACHE STRING "Per-tier timeout in seconds")

helfem_add_integration_test(integration.quick    ci       "integration;quick")
helfem_add_integration_test(integration.symmetry symmetry "integration;symmetry")

option(HELFEM_SLOW_TESTS "Register the slow integration tier (hours)" OFF)
if(HELFEM_SLOW_TESTS)
  # Everything, including the cases that run minutes to hours each. This is
  # what the weekly workflow covers.
  add_test(NAME integration.full
           COMMAND ${Python3_EXECUTABLE} ${HELFEM_TEST_DRIVER}
                   --check ${HELFEM_TEST_REFS}
                   --bindir $<TARGET_FILE_DIR:diatomic>
                   --jobs ${HELFEM_TEST_JOBS} --timeout 7200)
  set_tests_properties(integration.full PROPERTIES
    LABELS "integration;slow"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 0)
endif()
