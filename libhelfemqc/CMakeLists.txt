# libhelfemqc -- the radial quantum-chemistry layer split out of libhelfem.
#
# Phase 3 of the library restructuring (docs/library-refactor.md): libhelfem
# is the reusable, domain-agnostic finite-element core; libhelfemqc is the
# chemistry that sits on top of it -- concrete nuclear charge distributions
# (the model-potential f(r) implementations), the FE Coulomb/exchange
# assembler, and the atom-centred orbital bases (NAO/GTO/STO). It links
# PUBLIC to helfem and knows about electrons; helfem does not.

# Public headers -- the moved quantum-chemistry surface. Copied into the
# build tree's include/ (mirroring libhelfem) so both in-tree consumers and
# the installed package see them on the same include path.
list(APPEND LIBHELFEMQC_PUBLIC_HEADERS
    # Concrete nucleus models -- the electron-nuclear f(r) implementations
    # behind modelpotential::ModelPotentialT<T> (the abstract base stays in
    # libhelfem).
    "PointNucleus.h"
    "GaussianNucleus.h"
    "HollowNucleus.h"
    "SphericalNucleus.h"
    "RegularizedNucleus.h"
    # One-multipole J/K assembly over an FE-backed atomic basis.
    "CoulombExchangeFE.h"
    # Atom-centred numerical-orbital bases built via libhelfem's
    # FE-projection factories.
    "NAORadialBasis.h"
    "GTO.h"
    "STO.h"
)
foreach(header IN LISTS LIBHELFEMQC_PUBLIC_HEADERS)
    configure_file("include/${header}" "include/${header}" COPYONLY)
endforeach()

add_library(helfemqc
    src/PointNucleus.cpp
    src/GaussianNucleus.cpp
    src/HollowNucleus.cpp
    src/SphericalNucleus.cpp
    src/RegularizedNucleus.cpp
)
target_include_directories(helfemqc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # The moved nucleus .cpp files include libhelfem's private headers
        # (utils.h) by quote; keep libhelfem/src on the private path so they
        # resolve exactly as they did when compiled inside libhelfem.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../libhelfem/src>
)
# PUBLIC helfem: the QC layer consumes RadialBasis / FiniteElementBasis /
# PolynomialBasis / ModelPotential (the abstract base) and the folded-in FEM
# quadrature, and its headers expose them, so consumers inherit helfem.
target_link_libraries(helfemqc PUBLIC helfem)
target_compile_features(helfemqc PUBLIC cxx_std_17)

# Quad precision: the moved nucleus sources carry guarded _Float128 explicit
# instantiations, so mirror libhelfem's opt-in (raise the standard and define
# HELFEM_HAVE_FLOAT128) where it is enabled.
if(HELFEM_FLOAT128)
  target_compile_features(helfemqc PUBLIC cxx_std_23)
  target_compile_definitions(helfemqc PUBLIC HELFEM_HAVE_FLOAT128)
endif()

# In-tree alias mirroring the exported name.
add_library(helfem::qc ALIAS helfemqc)

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include)
install(TARGETS helfemqc
    EXPORT helfemTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
