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
#ifndef HELFEM_EIGEN_IO_H
#define HELFEM_EIGEN_IO_H

#include "Matrix.h"
#include <string>

// Diagnostic dump helpers for helfem::Vector / helfem::Matrix. Used
// by the FE test drivers (harmonic, inttest, ...) that want the same
// arma-style raw-ASCII / print output the codebase had before the
// arma migration.

namespace helfem {
  namespace io {
    /// Write a Vector to `path` one entry per line (matches
    /// arma::mat::save("...", arma::raw_ascii) semantics for a
    /// column vector).
    void write_raw_ascii(const std::string & path, const helfem::Vector & v);

    /// Write a Matrix to `path` in row-major raw ASCII: entries in a
    /// row separated by single spaces, rows separated by newlines.
    void write_raw_ascii(const std::string & path, const helfem::Matrix & m);

    /// Print a Matrix to stdout in the same fixed-width 8.4f grid the
    /// arma::mat::print(name) layout used.
    void print_matrix(const std::string & name, const helfem::Matrix & m);
  }
}

#endif
