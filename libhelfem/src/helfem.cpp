/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#include <helfem.h>
#include <iostream>

using namespace std;

bool helfem::verbose = false;

void helfem::set_verbosity(bool new_verbosity) {
  if (verbose && new_verbosity)
    printf("HelFEM library already in verbose mode.");
  else if (!verbose && new_verbosity)
    printf("HelFEM library set to verbose mode.");
  verbose = new_verbosity;
}

std::string helfem::version() { return __HELFEM_VERSION__; }
