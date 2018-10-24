#!/bin/bash

# Installation directory
export target=$(pwd) # this installs the binaries under bin/ in the present directory

if [[ ! -d objdir ]]; then
    mkdir objdir
fi
cd objdir

#export CXXFLAGS="-g -O2 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation -DARMA_NO_DEBUG"
export CXXFLAGS="-g -O2 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation -Wshadow"
#export CXXFLAGS="-g -O0 -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation"
#export CXXFLAGS="-g -Og -Wall -Wno-implicit-fallthrough -Wno-misleading-indentation"

cmake ..  \
      -DUSE_OPENMP=ON \
      -DCMAKE_INSTALL_PREFIX=${target} \
      -DCMAKE_BUILD_TYPE=Release \

make -j9 VERBOSE=1 install
