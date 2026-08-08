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

set(HELFEM_TEST_JOBS 4 CACHE STRING "Parallel cases for the aggregate test")
set(HELFEM_TEST_TIMEOUT 5400 CACHE STRING "Per-test timeout in seconds")
option(HELFEM_SLOW_TESTS "Enable the slow integration cases (hours)" OFF)


# ONE CTEST TEST PER CASE.
#
# This used to be one test per tag, which meant `integration.full` was a
# single opaque entry: 6412 s, "***Failed", and no indication of which of
# ~180 cases failed or where the time went. CTest can only report what it
# is given, so it is given each case separately -- a failure now names the
# case, and `ctest` prints a per-case time.
#
# The case list comes from cases.json at configure time rather than being
# duplicated here, so adding a case to the suite registers it
# automatically. Re-run cmake after editing cases.json.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import json,sys; s=json.load(open(sys.argv[1]));\
print('\\n'.join(c['name']+'|'+','.join(c.get('tags',[])) for c in s['cases']))"
    ${CMAKE_CURRENT_SOURCE_DIR}/cases.json
  OUTPUT_VARIABLE HELFEM_CASE_LIST
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE HELFEM_CASE_LIST_RC)

if(NOT HELFEM_CASE_LIST_RC EQUAL 0)
  message(WARNING "Could not read tests/cases.json; integration cases not registered")
  return()
endif()

string(REPLACE "\n" ";" HELFEM_CASE_LINES "${HELFEM_CASE_LIST}")
set(_n_fast 0)
set(_n_slow 0)
foreach(line ${HELFEM_CASE_LINES})
  string(REPLACE "|" ";" parts "${line}")
  list(GET parts 0 case_name)
  list(LENGTH parts nparts)
  set(case_tags "")
  if(nparts GREATER 1)
    list(GET parts 1 case_tags_csv)
    string(REPLACE "," ";" case_tags "${case_tags_csv}")
  endif()

  # A case is registered under its own name; --filter anchors on it so no
  # other case is dragged in by a shared prefix.
  add_test(NAME case.${case_name}
           COMMAND ${Python3_EXECUTABLE} ${HELFEM_TEST_DRIVER}
                   --check ${HELFEM_TEST_REFS}
                   --filter "^${case_name}$"
                   --bindir $<TARGET_FILE_DIR:diatomic>
                   --jobs 1)
  set_tests_properties(case.${case_name} PROPERTIES
    LABELS "integration;${case_tags}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT ${HELFEM_TEST_TIMEOUT})

  # A plain `ctest` runs the quick tier and nothing else. CTest offers no
  # "default subset", so everything outside it is registered-but-disabled:
  # visible in `ctest -N`, skipped unless asked for. Gating on the ci tag
  # rather than on "not slow" matters -- only ~49 of the ~180 cases carry
  # ci, while ~130 are merely not-slow, and a default run of those would
  # be hours.
  if(NOT "ci" IN_LIST case_tags AND NOT HELFEM_SLOW_TESTS)
    set_tests_properties(case.${case_name} PROPERTIES DISABLED TRUE)
    math(EXPR _n_slow "${_n_slow}+1")
  else()
    math(EXPR _n_fast "${_n_fast}+1")
  endif()
endforeach()

# Invariants compare cases against EACH OTHER, so they cannot live in a
# per-case test -- each of those runs a single case in its own process.
# One extra test covers them for the quick tier; --smoke needs no
# reference file and fails on a crash or a violated invariant.
add_test(NAME integration.invariants
         COMMAND ${Python3_EXECUTABLE} ${HELFEM_TEST_DRIVER}
                 --smoke --tag ci --skip-slow
                 --bindir $<TARGET_FILE_DIR:diatomic>
                 --jobs ${HELFEM_TEST_JOBS})
set_tests_properties(integration.invariants PROPERTIES
  LABELS "integration;invariants"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  TIMEOUT ${HELFEM_TEST_TIMEOUT})

message(STATUS "HelFEM tests: ${_n_fast} quick case(s) enabled, ${_n_slow} "
               "other case(s) registered but disabled "
               "(-DHELFEM_SLOW_TESTS=ON enables all). Run them in parallel "
               "with e.g. ctest -j6.")
