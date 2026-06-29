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

    void enforce_occupations(arma::mat & C, arma::vec & E, const arma::mat & S, const arma::ivec & nocc, const std::vector<arma::uvec> & m_idx) {
      if(nocc.n_elem != m_idx.size())
        throw std::logic_error("nocc vector and symmetry indices don't match!\n");

      // Make sure index vector doesn't have duplicates
      {
        // Collect all indices
        arma::uvec fullidx;
        for(size_t i=0;i<m_idx.size();i++) {
          arma::uvec fidx(fullidx.n_elem+m_idx[i].n_elem);
          if(fullidx.n_elem)
            fidx.subvec(0,fullidx.n_elem-1)=fullidx;
          fidx.subvec(fullidx.n_elem,fidx.n_elem-1)=m_idx[i];
          fullidx=fidx;
        }
        // Get indices of unique elements
        arma::uvec iunq(arma::find_unique(fullidx));
        if(iunq.n_elem != fullidx.n_elem)
          throw std::logic_error("Duplicate basis functions in symmetry list!\n");
      }

      // Indices of occupied orbitals
      std::vector<arma::uword> occidx;

      // Loop over symmetries
      for(size_t isym=0;isym<m_idx.size();isym++) {
        // Check for occupation
        if(!nocc(isym))
          continue;

        // C submatrix
        arma::mat Csub(C.rows(m_idx[isym]));
        // S submatrix
        arma::mat Ssub(S.submat(m_idx[isym],m_idx[isym]));

        // Find basis vectors that belong to this symmetry: compute their norm
        arma::vec Csubnrm(arma::diagvec(Csub.t()*Ssub*Csub));
        // Clean up
        Csubnrm(arma::find(Csubnrm <= 10*DBL_EPSILON)).zeros();

        // Column indices of C that have non-zero elements
        arma::uvec Cind(arma::find(Csubnrm));

        // Add to list of occupied orbitals
        for(arma::sword io=0;io<nocc(isym);io++)
          occidx.push_back(Cind(io));
      }

      // Sort list
      std::sort(occidx.begin(),occidx.end());

      // Make sure orbital vector doesn't have duplicates
      {
        arma::uvec iunq(arma::find_unique(arma::conv_to<arma::uvec>::from(occidx)));
        if(iunq.n_elem != occidx.size()) {
          arma::conv_to<arma::uvec>::from(occidx).print("Occupied orbital list");
          fflush(stdout);
          throw std::logic_error("Duplicates in occupied orbital list!\n");
        }
      }

      // Add in the rest of the orbitals
      std::vector<arma::uword> virtidx;
      for(arma::uword i=0;i<C.n_cols;i++) {
        bool found=false;
        for(size_t j=0;j<occidx.size();j++)
          if(occidx[j]==i) {
            found=true;
            break;
          }
        if(!found)
          virtidx.push_back(i);
      }

      // Make sure orbital energies are in order
      arma::uvec occorder;
      if(occidx.size()) {
	occorder=arma::conv_to<arma::uvec>::from(occidx);
	arma::vec Eocc(E(occorder));
	arma::uvec occsort(arma::sort_index(Eocc,"ascend"));
	occorder=occorder(occsort);
      }

      arma::uvec virtorder;
      if(virtidx.size()) {
	virtorder=arma::conv_to<arma::uvec>::from(virtidx);
	arma::vec Evirt(E(virtorder));
	arma::uvec virtsort(arma::sort_index(Evirt,"ascend"));
	virtorder=virtorder(virtsort);
      }

      arma::uvec newidx(occorder.n_elem+virtorder.n_elem);
      if(occorder.n_elem)
	newidx.subvec(0,occorder.n_elem-1)=occorder;
      if(virtorder.n_elem)
	newidx.subvec(occorder.n_elem,newidx.n_elem-1)=virtorder;
      C=C.cols(newidx);
      E=E(newidx);
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

    namespace {
      // Convert one arma::uvec (chemistry-side index list) to a vector
      // of Eigen::Index, suitable for Eigen 3.4 (rows/cols) indexing.
      inline std::vector<Eigen::Index> uvec_to_indices(const arma::uvec & u) {
        std::vector<Eigen::Index> v(u.n_elem);
        for (size_t i = 0; i < u.n_elem; ++i) v[i] = static_cast<Eigen::Index>(u(i));
        return v;
      }
      // Pivot-sort indices ascending by E values (stable not required).
      inline std::vector<Eigen::Index> sort_index_ascending(const helfem::Vector & E) {
        std::vector<Eigen::Index> idx(E.size());
        for (Eigen::Index i = 0; i < E.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&E](Eigen::Index a, Eigen::Index b) { return E(a) < E(b); });
        return idx;
      }
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

    void eig_sub_wrk(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, size_t Nact) {
      // Active space must be at least as large as the occupied space, otherwise
      // the subvec/cols indices below underflow (everything here is unsigned).
      if(Nact <= Cocc.n_cols) {
        std::ostringstream oss;
        oss << "eig_sub_wrk: active space (" << Nact
            << ") must be larger than the occupied space ("
            << Cocc.n_cols << ").\n";
        throw std::logic_error(oss.str());
      }

      // Form orbital gradient
      arma::mat Forth(Cocc.t()*F*Cvirt);

      // Compute gradient norms
      arma::vec Fnorm(Forth.n_cols);
      for(size_t i=0;i<Forth.n_cols;i++)
        Fnorm(i)=arma::norm(Forth.col(i),"fro");

      // Sort in decreasing value
      arma::uvec idx(arma::sort_index(Fnorm,"descend"));

      // Update order
      Cvirt=Cvirt.cols(idx);
      Fnorm=Fnorm(idx);

      // Calculate norms
      double act(arma::sum(Fnorm.subvec(0,Nact-Cocc.n_cols-1)));
      double frz(arma::sum(Fnorm.subvec(Nact-Cocc.n_cols-1,Fnorm.n_elem-1)));
      printf("Active space norm %e, frozen space norm %e\n",act,frz);

      // Form subspace solution (Phase 5.11 bridge).
      arma::mat C;
      arma::mat Corth(arma::join_rows(Cocc,Cvirt.cols(0,Nact-Cocc.n_cols-1)));
      helfem::Vector E_e; helfem::Matrix C_e;
      eig_gsym(E_e, C_e, helfem::to_eigen(F), helfem::to_eigen(Corth));
      E = helfem::to_arma(E_e);
      C = helfem::to_arma(C_e);

      // Update occupied and virtual orbitals
      Cocc=C.cols(0,Cocc.n_cols-1);
      Cvirt.cols(0,Nact-Cocc.n_cols-1)=C.cols(Cocc.n_cols,Nact-1);
    }

    arma::mat enforce_fock_symmetry(const arma::mat & Fin, const std::vector<arma::uvec> & m_idx) {
      arma::mat Fout;
      Fout.zeros(Fin.n_rows,Fin.n_rows);

      // Loop over symmetries
      for(size_t isym=0;isym<m_idx.size();isym++) {
        // Find basis vectors that belong to this symmetry
        arma::uvec idx(m_idx[isym]);
        Fout(idx,idx)=Fin(idx,idx);
      }

      return Fout;
    }

    arma::mat fock_symmetry_average(const arma::mat & Fin, const std::vector< std::vector<arma::uvec> > & sym_idx) {
      arma::mat Fout(Fin);

      // Loop over symmetries
      for(size_t isym=0;isym<sym_idx.size();isym++) {
        // Form averaged Fock matrix
        arma::mat Fmean(sym_idx[isym][0].n_elem, sym_idx[isym][0].n_elem, arma::fill::zeros);
        for(size_t ic=0;ic<sym_idx[isym].size();ic++) {
          const arma::uvec & idx(sym_idx[isym][ic]);
          Fmean += Fin(idx,idx);
        }
        Fmean /= sym_idx[isym].size();

        // Write out the averaged matrix
        for(size_t ic=0;ic<sym_idx[isym].size();ic++) {
          const arma::uvec & idx(sym_idx[isym][ic]);
          Fout(idx,idx)=Fmean;
        }
      }

      return Fout;
    }

    void sort_eig(arma::vec & Eorb, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & Fao, size_t Nact, int maxit, double convthr) {
      // Initialize vector
      arma::mat C(Cocc.n_rows,Cocc.n_cols+Cvirt.n_cols);
      C.cols(0,Cocc.n_cols-1)=Cocc;
      C.cols(Cocc.n_cols,Cocc.n_cols+Cvirt.n_cols-1)=Cvirt;

      // Compute lower bounds of eigenvalues
      for(int it=0;it<maxit;it++) {
        // Fock matrix
        arma::mat Fmo(arma::trans(C)*Fao*C);

        // Gerschgorin lower bound for eigenvalues
        arma::vec Ebar(Fao.n_cols), R(Fao.n_cols);
        for(size_t i=0;i<Fmo.n_cols;i++) {
          double r=0.0;
          for(size_t j=0;j<i;j++)
            r+=std::pow(Fmo(j,i),2);
          for(size_t j=i+1;j<Fmo.n_rows;j++)
            r+=std::pow(Fmo(j,i),2);
          r=sqrt(r);

          Ebar(i)=Fmo(i,i);
          R(i)=r;
        }

        // Sort by lowest possible eigenvalue
        arma::uvec idx(arma::sort_index(Ebar-R,"ascend"));
        printf("Orbital guess iteration %i\n",(int) it);
        double ograd(arma::sum(arma::square(R.subvec(0,Cocc.n_cols-1))));
        printf("Orbital gradient %e, occupied orbitals\n",ograd);
        for(size_t i=0;i<Cocc.n_cols;i++)
          printf("%2i %5i % e .. % e\n",(int) i, (int) idx(i), Ebar(idx(i))-R(idx(i)), Ebar(i)+R(idx(i)));

        // Has sort converged?
        bool convd=true;
        // Check if circles overlap. Maximum occupied orbital energy is
        double Emax=Ebar(0)+R(0);
        for(size_t i=0;i<Cocc.n_cols;i++)
          Emax=std::max(Emax,Ebar(i)+R(i));
        for(size_t i=Cocc.n_cols;i<Ebar.n_elem;i++)
          if(Ebar(i)-R(i) >= Emax)
            // Circles overlap!
            convd=false;
        // Check if gradient has converged
        if(ograd>=convthr)
          convd=false;
        if(convd)
          break;

        // Change orbital ordering
        C=C.cols(idx);

        // Occupy orbitals with lowest estimated eigenvalues
        arma::mat Cocctest(C.cols(0,Cocc.n_cols-1));
        arma::mat Cvirttest(C.cols(Cocc.n_cols,C.n_cols-1));

        // Improve Gerschgorin estimate
        eig_sub_wrk(Eorb,Cocctest,Cvirttest,Fao,Nact);

        // Update C
        C.cols(0,Cocc.n_cols-1)=Cocctest;
        C.cols(Cocc.n_cols,C.n_cols-1)=Cvirttest;
      }

      Cocc=C.cols(0,Cocc.n_cols-1);
      Cvirt=C.cols(Cocc.n_cols,C.n_cols-1);
    }

    void eig_sub(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, size_t nsub, int maxit, double convthr) {
      if(nsub >= Cocc.n_cols+Cvirt.n_cols) {
        arma::mat Corth(arma::join_rows(Cocc,Cvirt));

        // Phase 5.11 bridge.
        arma::mat C;
        helfem::Vector E_e; helfem::Matrix C_e;
        eig_gsym(E_e, C_e, helfem::to_eigen(F), helfem::to_eigen(Corth));
        E = helfem::to_arma(E_e);
        C = helfem::to_arma(C_e);
        if(Cocc.n_cols)
          Cocc=C.cols(0,Cocc.n_cols-1);
        Cvirt=C.cols(Cocc.n_cols,C.n_cols-1);
        return;
      }

      // Initialization: make sure we're occupying the lowest eigenstates
      sort_eig(E, Cocc, Cvirt, F, nsub, maxit, convthr);
      // The above already does everything
      return;

      // Perform initial solution
      eig_sub_wrk(E,Cocc,Cvirt,F,nsub);

      // Iterative improvement
      int iit;
      arma::vec Eold;
      for(iit=0;iit<maxit;iit++) {
        // New subspace solution
        eig_sub_wrk(E,Cocc,Cvirt,F,nsub);
        // Frozen subspace gradient
        arma::mat Cfrz(Cvirt.cols(nsub-Cocc.n_cols,Cvirt.n_cols-1));
        // Orbital gradient
        arma::mat G(arma::trans(Cocc)*F*Cfrz);
        double ograd(arma::norm(G,"fro"));
        printf("Eigeniteration %i orbital gradient %e\n",iit,ograd);
        arma::vec Ecur(E.subvec(0,Cocc.n_cols-1));
        Ecur.t().print("Occupied energies");
        if(iit>0)
          (Ecur-Eold).t().print("Energy change");
        Eold=Ecur;
        if(ograd<convthr)
          break;
      }
      if(iit==maxit) throw std::runtime_error("Eigensolver did not converge!\n");
    }

    void eig_iter(arma::vec & E, arma::mat & Cocc, arma::mat & Cvirt, const arma::mat & F, const arma::mat & Sinvh, size_t nocc, size_t neig, size_t nsub, int maxit, double convthr) {
      arma::mat Forth(Sinvh.t()*F*Sinvh);

      const arma::newarp::DenseGenMatProd<double> op(Forth);

      arma::newarp::SymEigsSolver< double, arma::newarp::EigsSelect::SMALLEST_ALGE, arma::newarp::DenseGenMatProd<double> > eigs(op, neig, nsub);
      eigs.init();

      arma::uword nconv = eigs.compute(maxit, convthr);
      printf("%i eigenvalues converged in %i iterations\n",(int) nconv, (int) eigs.num_iterations());
      E = eigs.eigenvalues();
      if(nconv < nocc)
        throw std::logic_error("Eigendecomposition did not convege!\n");

      arma::mat eigvec;
      // Go back to non-orthogonal basis
      eigvec = Sinvh*eigs.eigenvectors();

      Cocc=eigvec.cols(0,nocc-1);
      if(eigvec.n_cols>nocc)
        Cvirt=eigvec.cols(nocc,eigvec.n_cols-1);
      else
        Cvirt.clear();
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
