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
#include "eigen_io.h"
#include <cstdio>
#include <fstream>

namespace helfem {
  namespace io {



    // Match arma::mat::save(path, arma::raw_ascii) byte-for-byte: each entry
    // written in scientific notation, 16 significant figures, right-justified
    // in a field of width 24 with a single leading space. This is what the
    // suite's *.dat outputs (SAP tables, completeness profiles, quadrature
    // dumps) have always looked like; the previous default-ostream formatting
    // silently truncated to 6 figures despite the docstring promising arma
    // semantics.
    static void write_row(std::ofstream & f, const helfem::Matrix & m, Eigen::Index i) {
      for (Eigen::Index j = 0; j < m.cols(); ++j) {
        f.put(' ');
        f.width(24);
        f << m(i, j);
      }
      f.put('\n');
    }

    void write_raw_ascii(const std::string & path, const helfem::Vector & v) {
      std::ofstream out(path);
      out.setf(std::ios::scientific);
      out.fill(' ');
      out.precision(16);
      const helfem::Matrix col(v);          // N x 1: one entry per line
      for (Eigen::Index i = 0; i < col.rows(); ++i)
        write_row(out, col, i);
    }

    void write_raw_ascii(const std::string & path, const helfem::Matrix & m) {
      std::ofstream out(path);
      out.setf(std::ios::scientific);
      out.fill(' ');
      out.precision(16);
      for (Eigen::Index i = 0; i < m.rows(); ++i)
        write_row(out, m, i);
    }

    void print_matrix(const std::string & name, const helfem::Matrix & m) {
      printf("%s\n", name.c_str());
      for (Eigen::Index i = 0; i < m.rows(); ++i) {
        for (Eigen::Index j = 0; j < m.cols(); ++j)
          printf(" %8.4f", m(i, j));
        printf("\n");
      }
    }

  }
}
