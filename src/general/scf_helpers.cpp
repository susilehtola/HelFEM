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
#include "scf_helpers.h"
#include "timer.h"
#include <ArmaEigen.h>
#include <Eigen/Eigenvalues>
#include <cfloat>
#include <fstream>
#include <random>
#include <sstream>

namespace helfem {
  namespace scf {
    // Phase 5.10: Eigen-typed.

    namespace {
      // Sort indices ascending by E values.
      inline std::vector<Eigen::Index> sort_index_ascending(const helfem::Vector & E) {
        std::vector<Eigen::Index> idx(E.size());
        for (Eigen::Index i = 0; i < E.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&E](Eigen::Index a, Eigen::Index b) { return E(a) < E(b); });
        return idx;
      }
    }

    // Phase 5.13: Eigen-typed matrices; m_idx kept arma::uvec.

    // Phase 5.11: Eigen-typed.
    void eig_gsym(helfem::Vector & E, helfem::Matrix & C, const helfem::Matrix & F, const helfem::Matrix & Sinvh) {
      // Form matrix in orthonormal basis: Forth = Sinvh^T * F * Sinvh.
      const helfem::Matrix Forth = Sinvh.transpose() * F * Sinvh;
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Forth);
      if (es.info() != Eigen::Success)
        throw std::logic_error("Eigendecomposition failed!\n");
      E = es.eigenvalues();
      // Return to non-orthonormal basis: C = Sinvh * V.
      C = Sinvh * es.eigenvectors();
    }

    // Eigen matrices; m_idx is a per-symmetry list of Eigen index lists.
    void eig_gsym_sub(helfem::Vector & E, helfem::Matrix & C,
                      const helfem::Matrix & F, const helfem::Matrix & Sinvh,
                      const std::vector<std::vector<Eigen::Index>> & m_idx, bool verbose) {
      (void) verbose;
      E = helfem::Vector::Zero(F.rows());
      C = helfem::Matrix::Zero(F.rows(), F.rows());

      Eigen::Index iidx = 0;
      for (size_t isym = 0; isym < m_idx.size(); ++isym) {
        const std::vector<Eigen::Index> & rows = m_idx[isym];
        // Scmp = Sinvh(rows, :).
        const helfem::Matrix Scmp = Sinvh(rows, Eigen::all);
        // Snrm(j) = ||Scmp.col(j)||.
        helfem::Vector Snrm(Scmp.cols());
        for (Eigen::Index j = 0; j < Scmp.cols(); ++j)
          Snrm(j) = Scmp.col(j).norm();
        // Sind = indices of columns with nonzero norm.
        std::vector<Eigen::Index> Sind;
        Sind.reserve(Snrm.size());
        for (Eigen::Index j = 0; j < Snrm.size(); ++j)
          if (Snrm(j) != 0.0) Sind.push_back(j);

        helfem::Vector Esub;
        helfem::Matrix Csub;
        const helfem::Matrix SinvhSub = Sinvh(Eigen::all, Sind);
        eig_gsym(Esub, Csub, F, SinvhSub);

        E.segment(iidx, Esub.size()) = Esub;
        C.middleCols(iidx, Csub.cols()) = Csub;
        iidx += Esub.size();
      }
      if (iidx != F.rows()) {
        std::ostringstream oss;
        oss << "Symmetry mismatch: expected " << F.rows() << " vectors but got " << iidx << "!\n";
        throw std::logic_error(oss.str());
      }

      // Sort energies ascending.
      const auto Eord = sort_index_ascending(E);
      const helfem::Vector Eold = E;
      const helfem::Matrix Cold = C;
      for (Eigen::Index i = 0; i < E.size(); ++i) {
        E(i) = Eold(Eord[i]);
        C.col(i) = Cold.col(Eord[i]);
      }
    }


    // Phase 5.14: Eigen-typed.

    // Phase 5.13: Eigen-typed.

    helfem::Matrix fock_symmetry_average(const helfem::Matrix & Fin,
                                          const std::vector< std::vector<std::vector<Eigen::Index>> > & sym_idx) {
      helfem::Matrix Fout = Fin;
      for (size_t isym = 0; isym < sym_idx.size(); ++isym) {
        const Eigen::Index n = static_cast<Eigen::Index>(sym_idx[isym][0].size());
        helfem::Matrix Fmean = helfem::Matrix::Zero(n, n);
        for (size_t ic = 0; ic < sym_idx[isym].size(); ++ic) {
          const std::vector<Eigen::Index> & idx = sym_idx[isym][ic];
          Fmean += Fin(idx, idx);
        }
        Fmean /= static_cast<double>(sym_idx[isym].size());
        for (size_t ic = 0; ic < sym_idx[isym].size(); ++ic) {
          const std::vector<Eigen::Index> & idx = sym_idx[isym][ic];
          Fout(idx, idx) = Fmean;
        }
      }
      return Fout;
    }

    // Phase 5.14: Eigen-typed.
    // (sort_eig covers the work) and elided the dead code.
    // needs the iterative path, swap in Spectra (header-only) here.
    // Phase 5.15: Eigen-typed.
    // Phase 5.16: Eigen-typed.
    // J. Chem. Phys. 134, 064101 (2011).


    void parse_nela_nelb(int & nela, int & nelb, int & Q, int & M, int Z) {
      if(nela==0 && nelb==0) {
        // Use Q and M. Number of electrons is
        int nel=Z-Q;
        if(M<1)
          throw std::runtime_error("Invalid value for multiplicity, which must be >=1.\n");
        else if(nel%2==0 && M%2!=1) {
          std::ostringstream oss;
          oss << "Requested multiplicity " << M << " with " << nel << " electrons.\n";
          throw std::runtime_error(oss.str());
        } else if(nel%2==1 && M%2!=0) {
          std::ostringstream oss;
          oss << "Requested multiplicity " << M << " with " << nel << " electrons.\n";
          throw std::runtime_error(oss.str());
        }

        if(nel%2==0)
          // Even number of electrons, the amount of spin up is
          nela=nel/2+(M-1)/2;
        else
          // Odd number of electrons, the amount of spin up is
          nela=nel/2+M/2;
        // The rest are spin down
        nelb=nel-nela;

        if(nela<0) {
          std::ostringstream oss;
          oss << "A multiplicity of " << M << " would mean " << nela << " alpha electrons!\n";
          throw std::runtime_error(oss.str());
        } else if(nelb<0) {
          std::ostringstream oss;
          oss << "A multiplicity of " << M << " would mean " << nelb << " beta electrons!\n";
          throw std::runtime_error(oss.str());
        }

      } else {
        Q=Z-nela-nelb;
        M=1+nela-nelb;

        if(M<1) {
          std::ostringstream oss;
          oss << "nela=" << nela << ", nelb=" << nelb << " would mean multiplicity " << M << " which is not allowed!\n";
          throw std::runtime_error(oss.str());
        }
      }
    }

    // Phase 5.15: Eigen-typed. Parses either a whitespace-separated
    // ASCII file or a string of doubles.
    helfem::Vector parse_xc_params(const std::string & input) {
      helfem::Vector r;
      if (!input.size()) return r;

      bool isfile;
      {
        std::ifstream probe(input.c_str());
        isfile = probe.good();
      }

      std::vector<double> vals;
      if (isfile) {
        std::ifstream f(input.c_str());
        double x;
        while (f >> x) vals.push_back(x);
      } else {
        std::istringstream iss(input);
        double x;
        while (iss >> x) vals.push_back(x);
      }

      r.resize(static_cast<Eigen::Index>(vals.size()));
      for (size_t i = 0; i < vals.size(); ++i)
        r(static_cast<Eigen::Index>(i)) = vals[i];
      return r;
    }
  }
}
