#!/bin/bash

# Installation directory
export target=$(pwd)/install # this installs the binaries under install/bin/ in the present directory

if [[ ! -d objdir ]]; then
    mkdir objdir
fi
cd objdir

ARMAFLAGS="-DARMA_NO_DEBUG -DARMA_DONT_USE_WRAPPER -llapack -lblas"
export CXXFLAGS="-g -O2 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation ${ARMAFLAGS}"
cmake ..  \
          -DUSE_OPENMP=ON \
          -DHELFEM_FIND_DEPS=ON \
          -DCMAKE_INSTALL_PREFIX=${target} \
          -DCMAKE_BUILD_TYPE=Release

make -j4 VERBOSE=1 install
