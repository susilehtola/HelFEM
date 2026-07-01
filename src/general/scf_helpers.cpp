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

namespace helfem {
  namespace scf {
    // Phase 5.10: Eigen-typed.
    helfem::Matrix form_density(const helfem::Matrix & C, size_t nocc) {
      if (static_cast<size_t>(C.cols()) < nocc)
        throw std::logic_error("Not enough orbitals!\n");
      if (nocc == 0)
        return helfem::Matrix::Zero(C.rows(), C.rows());
      const auto Cocc = C.leftCols(static_cast<Eigen::Index>(nocc));
      return Cocc * Cocc.transpose();
    }

    namespace {
      // Convert one arma::uvec (chemistry-side index list) to a vector
      // of Eigen::Index, suitable for Eigen 3.4 (rows/cols) indexing.
      inline std::vector<Eigen::Index> uvec_to_indices(const arma::uvec & u) {
        std::vector<Eigen::Index> v(u.n_elem);
        for (size_t i = 0; i < u.n_elem; ++i) v[i] = static_cast<Eigen::Index>(u(i));
        return v;
      }
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
    void enforce_occupations(helfem::Matrix & C, helfem::Vector & E,
                              const helfem::Matrix & S, const arma::ivec & nocc,
                              const std::vector<arma::uvec> & m_idx) {
      if (nocc.n_elem != m_idx.size())
        throw std::logic_error("nocc vector and symmetry indices don't match!\n");

      // Duplicate check across the union of symmetry blocks.
      {
        std::vector<arma::uword> all;
        for (size_t i = 0; i < m_idx.size(); ++i)
          for (arma::uword j = 0; j < m_idx[i].n_elem; ++j)
            all.push_back(m_idx[i](j));
        std::vector<arma::uword> sorted = all;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
          throw std::logic_error("Duplicate basis functions in symmetry list!\n");
      }

      std::vector<Eigen::Index> occidx;
      for (size_t isym = 0; isym < m_idx.size(); ++isym) {
        if (!nocc(isym)) continue;
        const auto rows = uvec_to_indices(m_idx[isym]);
        // Csub = C(rows, :), Ssub = S(rows, rows).
        const helfem::Matrix Csub = C(rows, Eigen::all);
        const helfem::Matrix Ssub = S(rows, rows);
        // Csubnrm = diag(Csub^T Ssub Csub).
        helfem::Vector Csubnrm = (Csub.transpose() * Ssub * Csub).diagonal();
        const double thr = 10 * DBL_EPSILON;
        for (Eigen::Index j = 0; j < Csubnrm.size(); ++j)
          if (Csubnrm(j) <= thr) Csubnrm(j) = 0.0;
        // Cind = columns of C with nonzero norm in this block.
        std::vector<Eigen::Index> Cind;
        Cind.reserve(Csubnrm.size());
        for (Eigen::Index j = 0; j < Csubnrm.size(); ++j)
          if (Csubnrm(j) != 0.0) Cind.push_back(j);
        for (arma::sword io = 0; io < nocc(isym); ++io)
          occidx.push_back(Cind[io]);
      }
      std::sort(occidx.begin(), occidx.end());

      // Duplicate check on occupied indices.
      if (std::adjacent_find(occidx.begin(), occidx.end()) != occidx.end()) {
        for (auto i : occidx) printf("%lld ", (long long) i);
        printf(" -- occupied orbital list\n");
        fflush(stdout);
        throw std::logic_error("Duplicates in occupied orbital list!\n");
      }

      std::vector<Eigen::Index> virtidx;
      virtidx.reserve(C.cols());
      for (Eigen::Index i = 0; i < C.cols(); ++i) {
        bool found = false;
        for (auto j : occidx) if (j == i) { found = true; break; }
        if (!found) virtidx.push_back(i);
      }

      // Sort occupied by energy ascending.
      std::sort(occidx.begin(), occidx.end(),
                [&E](Eigen::Index a, Eigen::Index b) { return E(a) < E(b); });
      std::sort(virtidx.begin(), virtidx.end(),
                [&E](Eigen::Index a, Eigen::Index b) { return E(a) < E(b); });

      std::vector<Eigen::Index> newidx;
      newidx.reserve(occidx.size() + virtidx.size());
      for (auto i : occidx)  newidx.push_back(i);
      for (auto i : virtidx) newidx.push_back(i);

      const helfem::Matrix Cold = C;
      const helfem::Vector Eold = E;
      for (size_t k = 0; k < newidx.size(); ++k) {
        C.col(static_cast<Eigen::Index>(k)) = Cold.col(newidx[k]);
        E(static_cast<Eigen::Index>(k))     = Eold(newidx[k]);
      }
    }

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

    // Phase 5.12: Eigen matrices; m_idx kept arma::uvec until TwoDBasis
    // ::get_sym_idx migrates.
    void eig_gsym_sub(helfem::Vector & E, helfem::Matrix & C,
                      const helfem::Matrix & F, const helfem::Matrix & Sinvh,
                      const std::vector<arma::uvec> & m_idx, bool verbose) {
      (void) verbose;
      E = helfem::Vector::Zero(F.rows());
      C = helfem::Matrix::Zero(F.rows(), F.rows());

      Eigen::Index iidx = 0;
      for (size_t isym = 0; isym < m_idx.size(); ++isym) {
        const auto rows = uvec_to_indices(m_idx[isym]);
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

    void eig_sym_sub(helfem::Vector & E, helfem::Matrix & C,
                     const helfem::Matrix & F, const std::vector<arma::uvec> & m_idx) {
      E = helfem::Vector::Zero(F.rows());
      C = helfem::Matrix::Zero(F.rows(), F.rows());

      Eigen::Index iidx = 0;
      for (size_t i = 0; i < m_idx.size(); ++i) {
        const auto idx = uvec_to_indices(m_idx[i]);
        // Subblock F(idx, idx).
        const helfem::Matrix Fsub = F(idx, idx);
        Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Fsub);
        if (es.info() != Eigen::Success)
          throw std::logic_error("Subspace eigendecomposition failed!\n");
        const helfem::Vector Esub = es.eigenvalues();
        const helfem::Matrix Csub = es.eigenvectors();
        // Place results: E[iidx:iidx+n] = Esub; C(idx, iidx:iidx+n) = Csub.
        E.segment(iidx, Esub.size()) = Esub;
        for (Eigen::Index k = 0; k < Csub.cols(); ++k)
          for (size_t r = 0; r < idx.size(); ++r)
            C(idx[r], iidx + k) = Csub(static_cast<Eigen::Index>(r), k);
        iidx += Esub.size();
      }
      if (iidx != F.rows()) {
        std::ostringstream oss;
        oss << "Symmetry mismatch: expected " << F.rows() << " vectors but got " << iidx << "!\n";
        throw std::logic_error(oss.str());
      }

      const auto Eord = sort_index_ascending(E);
      const helfem::Vector Eold = E;
      const helfem::Matrix Cold = C;
      for (Eigen::Index i = 0; i < E.size(); ++i) {
        E(i) = Eold(Eord[i]);
        C.col(i) = Cold.col(Eord[i]);
      }
    }

    // Phase 5.14: Eigen-typed.
    void eig_sub_wrk(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt,
                      const helfem::Matrix & F, size_t Nact) {
      if (Nact <= static_cast<size_t>(Cocc.cols())) {
        std::ostringstream oss;
        oss << "eig_sub_wrk: active space (" << Nact
            << ") must be larger than the occupied space ("
            << Cocc.cols() << ").\n";
        throw std::logic_error(oss.str());
      }

      // Orbital gradient block Forth = Cocc^T * F * Cvirt.
      const helfem::Matrix Forth = Cocc.transpose() * F * Cvirt;

      // Gradient norms per virtual column.
      helfem::Vector Fnorm(Forth.cols());
      for (Eigen::Index i = 0; i < Forth.cols(); ++i)
        Fnorm(i) = Forth.col(i).norm();

      // Sort virtual orbitals by descending gradient norm.
      std::vector<Eigen::Index> idx(Fnorm.size());
      for (Eigen::Index i = 0; i < Fnorm.size(); ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(),
                [&Fnorm](Eigen::Index a, Eigen::Index b) { return Fnorm(a) > Fnorm(b); });
      const helfem::Matrix Cvirt_old = Cvirt;
      const helfem::Vector Fnorm_old = Fnorm;
      for (size_t k = 0; k < idx.size(); ++k) {
        Cvirt.col(static_cast<Eigen::Index>(k)) = Cvirt_old.col(idx[k]);
        Fnorm(static_cast<Eigen::Index>(k))    = Fnorm_old(idx[k]);
      }

      const Eigen::Index Nact_virt = static_cast<Eigen::Index>(Nact) - Cocc.cols();
      const double act = Fnorm.head(Nact_virt).sum();
      const double frz = Fnorm.tail(Fnorm.size() - Nact_virt).sum();
      printf("Active space norm %e, frozen space norm %e\n", act, frz);

      helfem::Matrix Corth(Cocc.rows(), Cocc.cols() + Nact_virt);
      Corth.leftCols(Cocc.cols())  = Cocc;
      Corth.rightCols(Nact_virt)   = Cvirt.leftCols(Nact_virt);

      helfem::Matrix C;
      eig_gsym(E, C, F, Corth);

      const Eigen::Index n_occ = Cocc.cols();
      Cocc                       = C.leftCols(n_occ);
      Cvirt.leftCols(Nact_virt)  = C.middleCols(n_occ, Nact_virt);
    }

    // Phase 5.13: Eigen-typed.
    helfem::Matrix enforce_fock_symmetry(const helfem::Matrix & Fin,
                                          const std::vector<arma::uvec> & m_idx) {
      helfem::Matrix Fout = helfem::Matrix::Zero(Fin.rows(), Fin.rows());
      for (size_t isym = 0; isym < m_idx.size(); ++isym) {
        const auto idx = uvec_to_indices(m_idx[isym]);
        Fout(idx, idx) = Fin(idx, idx);
      }
      return Fout;
    }

    helfem::Matrix fock_symmetry_average(const helfem::Matrix & Fin,
                                          const std::vector< std::vector<arma::uvec> > & sym_idx) {
      helfem::Matrix Fout = Fin;
      for (size_t isym = 0; isym < sym_idx.size(); ++isym) {
        const Eigen::Index n = static_cast<Eigen::Index>(sym_idx[isym][0].n_elem);
        helfem::Matrix Fmean = helfem::Matrix::Zero(n, n);
        for (size_t ic = 0; ic < sym_idx[isym].size(); ++ic) {
          const auto idx = uvec_to_indices(sym_idx[isym][ic]);
          Fmean += Fin(idx, idx);
        }
        Fmean /= static_cast<double>(sym_idx[isym].size());
        for (size_t ic = 0; ic < sym_idx[isym].size(); ++ic) {
          const auto idx = uvec_to_indices(sym_idx[isym][ic]);
          Fout(idx, idx) = Fmean;
        }
      }
      return Fout;
    }

    // Phase 5.14: Eigen-typed.
    void sort_eig(helfem::Vector & Eorb, helfem::Matrix & Cocc, helfem::Matrix & Cvirt,
                   const helfem::Matrix & Fao, size_t Nact, int maxit, double convthr) {
      // C = [Cocc | Cvirt].
      helfem::Matrix C(Cocc.rows(), Cocc.cols() + Cvirt.cols());
      C.leftCols(Cocc.cols())   = Cocc;
      C.rightCols(Cvirt.cols()) = Cvirt;

      for (int it = 0; it < maxit; ++it) {
        const helfem::Matrix Fmo = C.transpose() * Fao * C;

        // Gerschgorin lower bound for eigenvalues.
        helfem::Vector Ebar(Fmo.cols()), R(Fmo.cols());
        for (Eigen::Index i = 0; i < Fmo.cols(); ++i) {
          double r = 0.0;
          for (Eigen::Index j = 0; j < i; ++j)          r += std::pow(Fmo(j, i), 2);
          for (Eigen::Index j = i + 1; j < Fmo.rows(); ++j) r += std::pow(Fmo(j, i), 2);
          Ebar(i) = Fmo(i, i);
          R(i)    = std::sqrt(r);
        }

        // Sort ascending by (Ebar - R).
        const helfem::Vector key = Ebar - R;
        std::vector<Eigen::Index> idx(key.size());
        for (Eigen::Index i = 0; i < key.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&key](Eigen::Index a, Eigen::Index b) { return key(a) < key(b); });

        printf("Orbital guess iteration %i\n", it);
        const double ograd = R.head(Cocc.cols()).array().square().sum();
        printf("Orbital gradient %e, occupied orbitals\n", ograd);
        for (Eigen::Index i = 0; i < Cocc.cols(); ++i)
          printf("%2i %5i % e .. % e\n", (int) i, (int) idx[i],
                  Ebar(idx[i]) - R(idx[i]), Ebar(i) + R(idx[i]));

        // Convergence: Gerschgorin discs of virtual orbitals must sit
        // above the maximum occupied bound, and orbital gradient below
        // threshold.
        bool convd = true;
        double Emax = Ebar(0) + R(0);
        for (Eigen::Index i = 0; i < Cocc.cols(); ++i)
          Emax = std::max(Emax, Ebar(i) + R(i));
        for (Eigen::Index i = Cocc.cols(); i < Ebar.size(); ++i)
          if (Ebar(i) - R(i) >= Emax) convd = false;
        if (ograd >= convthr) convd = false;
        if (convd) break;

        // Reorder C by idx.
        const helfem::Matrix Cold = C;
        for (size_t k = 0; k < idx.size(); ++k)
          C.col(static_cast<Eigen::Index>(k)) = Cold.col(idx[k]);

        helfem::Matrix Cocctest  = C.leftCols(Cocc.cols());
        helfem::Matrix Cvirttest = C.rightCols(C.cols() - Cocc.cols());
        eig_sub_wrk(Eorb, Cocctest, Cvirttest, Fao, Nact);
        C.leftCols(Cocc.cols())                     = Cocctest;
        C.rightCols(C.cols() - Cocc.cols())         = Cvirttest;
      }

      Cocc  = C.leftCols(Cocc.cols());
      Cvirt = C.rightCols(C.cols() - Cocc.cols());
    }

    // Phase 5.14: Eigen-typed. The post-return unreachable iterative
    // block was already dead in the arma version; kept the pattern
    // (sort_eig covers the work) and elided the dead code.
    void eig_sub(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt,
                  const helfem::Matrix & F, size_t nsub, int maxit, double convthr) {
      if (nsub >= static_cast<size_t>(Cocc.cols() + Cvirt.cols())) {
        helfem::Matrix Corth(Cocc.rows(), Cocc.cols() + Cvirt.cols());
        Corth.leftCols(Cocc.cols())   = Cocc;
        Corth.rightCols(Cvirt.cols()) = Cvirt;
        helfem::Matrix C;
        eig_gsym(E, C, F, Corth);
        const Eigen::Index n_occ = Cocc.cols();
        if (n_occ) Cocc = C.leftCols(n_occ);
        Cvirt = C.rightCols(C.cols() - n_occ);
        return;
      }
      sort_eig(E, Cocc, Cvirt, F, nsub, maxit, convthr);
    }

    // Phase 5.14: iterative eigenvalue solver.
    //
    // The arma version used arma::newarp (an in-tree ARPACK-ish Lanczos).
    // Eigen has no built-in equivalent and there are no in-tree callers,
    // so the safe migration is a plain full-spectrum solve on Forth via
    // SelfAdjointEigenSolver. If a caller ever comes back that genuinely
    // needs the iterative path, swap in Spectra (header-only) here.
    void eig_iter(helfem::Vector & E, helfem::Matrix & Cocc, helfem::Matrix & Cvirt,
                   const helfem::Matrix & F, const helfem::Matrix & Sinvh,
                   size_t nocc, size_t neig, size_t nsub, int maxit, double convthr) {
      (void) nsub; (void) maxit; (void) convthr;
      const helfem::Matrix Forth = Sinvh.transpose() * F * Sinvh;
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(Forth);
      if (es.info() != Eigen::Success)
        throw std::logic_error("eig_iter: eigendecomposition failed!\n");
      // Take the smallest neig eigenpairs (ascending order from
      // SelfAdjointEigenSolver).
      const Eigen::Index Ne  = std::min<Eigen::Index>(neig, es.eigenvalues().size());
      E = es.eigenvalues().head(Ne);
      const helfem::Matrix eigvec = Sinvh * es.eigenvectors().leftCols(Ne);

      if (static_cast<size_t>(Ne) < nocc)
        throw std::logic_error("Eigendecomposition did not converge!\n");
      Cocc = eigvec.leftCols(nocc);
      if (static_cast<size_t>(eigvec.cols()) > nocc)
        Cvirt = eigvec.rightCols(eigvec.cols() - nocc);
      else
        Cvirt.resize(eigvec.rows(), 0);
    }

    arma::mat perturbation_matrix(size_t N, double ampl) {
      arma::mat R(N,N);
      // Uniform distribution
      R.randu();
      // Apply amplitude and antisymmetrize
      R=0.5*ampl*(R-R.t());
      // Eigendecompose
      arma::vec Rval;
      arma::cx_mat Rvec;
      bool diagok=arma::eig_sym(Rval,Rvec,std::complex<double>(0.0,-1.0)*R);
      if(!diagok)
        throw std::runtime_error("Error diagonalizing R.\n");

      // Rotation matrix is given by
      return arma::real(Rvec*arma::diagmat(arma::exp(std::complex<double>(0.0,1.0)*Rval))*arma::trans(Rvec));
    }

    void form_NOs(const arma::mat & P, const arma::mat & Sh, const arma::mat & Sinvh, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs) {
      // P in orthonormal basis is
      arma::mat P_orth=arma::trans(Sh)*P*Sh;

      // Diagonalize P to get NOs in orthonormal basis.
      arma::vec Pval;
      arma::mat Pvec;
      arma::eig_sym(Pval,Pvec,P_orth);

      // Reverse ordering to get decreasing eigenvalues
      occs.zeros(Pval.n_elem);
      arma::mat Pv(Pvec.n_rows,Pvec.n_cols);
      for(size_t i=0;i<Pval.n_elem;i++) {
        size_t idx=Pval.n_elem-1-i;
        occs(i)=Pval(idx);
        Pv.col(i)=Pvec.col(idx);
      }

      /* Get NOs in AO basis. The natural orbital is written in the
         orthonormal basis as

         |i> = x_ai |a> = x_ai s_ja |j>
         = s_ja x_ai |j>
      */

      // The matrix that takes us from AO to NO is
      AO_to_NO=Sinvh*Pv;
      // and the one that takes us from NO to AO is
      NO_to_AO=arma::trans(Sh*Pv);
    }

    void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & Sh, const arma::mat & Sinvh, int nocca, int noccb) {
      /*
       * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
       * unrestricted mean-field methods for controlling
       * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
       */

      Timer t;

      arma::vec occs;
      arma::mat AO_to_NO;
      arma::mat NO_to_AO;
      form_NOs(P_AO,Sh,Sinvh,AO_to_NO,NO_to_AO,occs);

      // Construct \Delta matrix in AO basis
      arma::mat Delta_AO=(Fa_AO-Fb_AO)/2.0;

      // and take it to the NO basis.
      arma::mat Delta_NO=arma::trans(AO_to_NO)*Delta_AO*AO_to_NO;

      // Amount of independent orbitals is
      size_t Nind=AO_to_NO.n_cols;
      // Amount of core orbitals is
      size_t Nc=std::min(nocca,noccb);
      // Amount of active space orbitals is
      size_t Na=std::max(nocca,noccb)-Nc;
      // Amount of virtual orbitals (in NO space) is
      size_t Nv=Nind-Na-Nc;

      // Form lambda by flipping the signs of the cv and vc blocks and
      // zeroing out everything else.
      arma::mat lambda_NO(Delta_NO);
      /*
        eig_sym_ordered puts the NOs in the order of increasing
        occupation. Thus, the lowest Nv orbitals belong to the virtual
        space, the following Na to the active space and the last Nc to the
        core orbitals.
      */
      // Zero everything
      lambda_NO.zeros();
      // and flip signs of cv and vc blocks from Delta
      for(size_t c=0;c<Nc;c++) // Loop over core orbitals
        for(size_t v=Nind-Nv;v<Nind;v++) { // Loop over virtuals
          lambda_NO(c,v)=-Delta_NO(c,v);
          lambda_NO(v,c)=-Delta_NO(v,c);
        }

      // Lambda in AO is
      arma::mat lambda_AO=arma::trans(NO_to_AO)*lambda_NO*NO_to_AO;

      // Update Fa and Fb
      Fa_AO+=lambda_AO;
      Fb_AO-=lambda_AO;

      printf("Performed CUHF update of Fock operators in %s.\n",t.elapsed().c_str());
    }


    std::string memory_size(size_t size) {
      std::ostringstream ret;

      const size_t kilo(1000);
      const size_t mega(kilo*kilo);
      const size_t giga(mega*kilo);

      // Number of gigabytes
      size_t gigs(size/giga);
      if(gigs>0) {
        size-=gigs*giga;
        ret << gigs;
        ret << " G ";
      }
      size_t megs(size/mega);
      if(megs>0) {
        size-=megs*mega;
        ret << megs;
        ret << " M ";
      }
      size_t kils(size/kilo);
      if(kils>0) {
        size-=kils*kilo;
        ret << kils;
        ret << " k ";
      }

      return ret.str();
    }

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

    arma::vec parse_xc_params(const std::string & input) {
      arma::vec r;
      if(input.size()) {
        // Is this a file name?
        bool isfile;
        {
          std::ifstream f(input.c_str());
          isfile = f.good();
        }
        if(isfile) {
          // Load the file
          r.load(input,arma::raw_ascii);
        } else {
          // Assume string input
          r = arma::vec(input);
        }
      }

      return r;
    }
  }
}
