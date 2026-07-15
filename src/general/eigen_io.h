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
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

// Diagnostic dump helpers for helfem::Vec<T> / helfem::Mat<T>. Used by the
// FE test drivers (harmonic, inttest, ...) and by gensap for the same
// arma-style raw-ASCII / print output the codebase had before the arma
// migration.

namespace helfem {
  namespace io {

    namespace detail {
      /// One matrix row, as arma::raw_ascii writes it: each entry
      /// right-justified in a fixed-width field with a single leading space.
      template <typename T>
      void write_raw_row(std::ofstream & f, const helfem::Mat<T> & m,
                         Eigen::Index i, int width) {
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
          f.put(' ');
          f.width(width);
          f << m(i, j);
        }
        f.put('\n');
      }
    }

    /// Write a Mat<T> / Vec<T> to `path` in the arma::save(path, raw_ascii)
    /// layout: scientific notation, one row per line, each entry
    /// right-justified in a fixed-width field with a single leading space.
    ///
    /// The precision is NOT hardcoded -- it follows the scalar type. A
    /// scientific field with precision p prints p+1 significant figures, and
    /// max_digits10 is the count needed to round-trip T, so precision =
    /// max_digits10(T)-1 dumps exactly enough to reconstruct the value and no
    /// more: 16 for double (byte-identical to the old arma output), 20 for
    /// long double, 35 for _Float128. The field width scales with it.
    template <typename T>
    void write_raw_ascii(const std::string & path, const helfem::Mat<T> & m) {
      const int prec  = std::numeric_limits<T>::max_digits10 - 1;
      const int width = std::numeric_limits<T>::max_digits10 + 7;  // sign, '.', 'e', exp; 24 at double
      std::ofstream out(path);
      out.setf(std::ios::scientific);
      out.fill(' ');
      out.precision(prec);
      for (Eigen::Index i = 0; i < m.rows(); ++i)
        detail::write_raw_row<T>(out, m, i, width);
    }

    template <typename T>
    void write_raw_ascii(const std::string & path, const helfem::Vec<T> & v) {
      write_raw_ascii<T>(path, helfem::Mat<T>(v));   // N x 1: one entry per line
    }

    /// Print a Mat<T> to stdout under a name header, one row per line, in
    /// scientific notation. Fixed %8.4f (the old layout) is useless for the
    /// wide dynamic range of the matrices this prints -- two-electron
    /// integrals, densities -- and is not arma's default either. The
    /// precision follows the scalar type, like write_raw_ascii above.
    template <typename T>
    void print_matrix(const std::string & name, const helfem::Mat<T> & m) {
      const int prec  = std::numeric_limits<T>::max_digits10 - 1;
      const int width = std::numeric_limits<T>::max_digits10 + 7;
      std::cout << name << "\n";
      std::ios_base::fmtflags flags(std::cout.flags());
      const std::streamsize oldprec(std::cout.precision());
      std::cout.setf(std::ios::scientific);
      std::cout.fill(' ');
      std::cout.precision(prec);
      for (Eigen::Index i = 0; i < m.rows(); ++i) {
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
          std::cout.put(' ');
          std::cout.width(width);
          std::cout << m(i, j);
        }
        std::cout.put('\n');
      }
      std::cout.flags(flags);
      std::cout.precision(oldprec);
    }
  }
}

#endif
