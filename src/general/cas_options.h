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
#ifndef HELFEM_CAS_OPTIONS_H
#define HELFEM_CAS_OPTIONS_H

#include "cmdline.h"
#include "cas_optimizer.h"

namespace helfem {
  namespace cas {

    /// The CASSCF options every driver offers, spelled the same everywhere.
    /// The trust-region knobs are the second-order SCF's (--somacro,
    /// --somicro, --soredfac, --sosolver), which the driver defines.
    inline void add_options(cmdline::parser & parser) {
      parser.add<int>("casact", 0, "CASSCF after the SCF: number of active orbitals (0 for none)", false, 0);
      parser.add<int>("casinact", 0, "CASSCF: number of doubly occupied inactive orbitals", false, 0);
      parser.add<int>("casM", 0, "CASSCF: spin multiplicity of the target state", false, 1);
      parser.add<bool>("cascoupled", 0, "CASSCF: optimize orbitals and CI coefficients together (else re-solve the CI at every orbital step)", false, true);
      parser.add<double>("casconvthr", 0, "CASSCF: RMS gradient to converge to", false, 1e-6);
    }

    inline CASRequest request(const cmdline::parser & parser, int verbosity) {
      CASRequest r;
      r.ninact = parser.get<int>("casinact");
      r.nact = parser.get<int>("casact");
      r.twoS = parser.get<int>("casM") - 1;
      r.coupling = parser.get<bool>("cascoupled") ? Coupling::Coupled : Coupling::TwoStep;
      r.options.otr.conv_tol = parser.get<double>("casconvthr");
      r.options.otr.n_macro = parser.get<int>("somacro");
      r.options.otr.n_micro = parser.get<int>("somicro");
      r.options.otr.global_red_factor = parser.get<double>("soredfac");
      r.options.otr.local_red_factor = 0.1 * r.options.otr.global_red_factor;
      r.options.otr.subsystem_solver = parser.get<std::string>("sosolver");
      r.options.verbosity = verbosity;
      return r;
    }

  } // namespace cas
} // namespace helfem

#endif // HELFEM_CAS_OPTIONS_H
