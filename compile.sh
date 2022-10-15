#!/bin/bash

# Installation directory
export target=$(pwd) # this installs the binaries under bin/ in the present directory

if [[ ! -d objdir ]]; then
    mkdir objdir
fi
cd objdir

if(( 1 )); then
    export CXXFLAGS="-g -O2 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation -DARMA_NO_DEBUG -DARMA_DONT_USE_WRAPPER"
    cmake ..  \
          -DUSE_OPENMP=ON \
          -DCMAKE_INSTALL_PREFIX=${target} \
          -DCMAKE_BUILD_TYPE=Release
else
    export CXXFLAGS="-g -O0 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation -Wextra -Wshadow -DARMA_DONT_USE_WRAPPER"
    cmake ..  \
          -DUSE_OPENMP=ON \
          -DCMAKE_INSTALL_PREFIX=${target} \
          -DCMAKE_BUILD_TYPE=Debug
fi

make -j9 VERBOSE=1 install
