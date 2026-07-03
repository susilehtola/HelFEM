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

    void write_raw_ascii(const std::string & path, const helfem::Vector & v) {
      std::ofstream out(path);
      for (Eigen::Index i = 0; i < v.size(); ++i)
        out << v(i) << "\n";
    }

    void write_raw_ascii(const std::string & path, const helfem::Matrix & m) {
      std::ofstream out(path);
      for (Eigen::Index i = 0; i < m.rows(); ++i) {
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
          if (j) out << " ";
          out << m(i, j);
        }
        out << "\n";
      }
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
