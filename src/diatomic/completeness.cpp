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
#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/timer.h"
#include "../general/lcao.h"
#include "basis.h"
#include "twodquadrature.h"
#include "Matrix.h"
#include "../general/eigen_io.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cfloat>
#include <climits>

using namespace helfem;

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("ldft", 0, "theta rule for quadrature (0 for auto)", false, 0);
  parser.add<std::string>("load", 0, "load guess from checkpoint", false, "");
  parser.add<int>("completeness", 0, "perform completeness scan up to lmax", false, -1);
  parser.add<double>("minexp", 0, "minimum exponent", false, 1e-5);
  parser.add<double>("maxexp", 0, "minimum exponent", false, 1e10);
  parser.add<int>("nexp", 0, "number of points in exponent scan", false, 501);
  parser.add<int>("iprobe", 0, "probe to use: 0 for gto, 1 for sto", false, 0);
  parser.parse_check(argc, argv);

  // Get parameters
  int ldft(parser.get<int>("ldft"));
  int completeness=parser.get<int>("completeness");
  std::string load(parser.get<std::string>("load"));
  double minexp(parser.get<double>("minexp"));
  double maxexp(parser.get<double>("maxexp"));
  int nexp(parser.get<int>("nexp"));
  int iprobe(parser.get<int>("iprobe"));

  // Load checkpoint
  Checkpoint loadchk(load,false);
  // Basis set
  diatomic::basis::TwoDBasis basis;
  loadchk.read(basis);
  // Sinvh
  helfem::Matrix Sinvh;
  loadchk.read("Sinvh",Sinvh);
  helfem::Matrix Sinv(Sinvh*Sinvh.transpose());
  // Orbitals
  helfem::Matrix Ca, Cb;
  loadchk.read("Ca",Ca);
  loadchk.read("Cb",Cb);
  // Number of occupied orbitals
  int nela, nelb;
  loadchk.read("nela",nela);
  loadchk.read("nelb",nelb);
  // Nuclear charges
  int Z1, Z2;
  loadchk.read("Z1",Z1);
  loadchk.read("Z2",Z2);

  // Completeness probe
  int lquad = (ldft>0) ? ldft : 4*basis.get_lval().maxCoeff()+12;
  helfem::diatomic::twodquad::TwoDGrid qgrid;
  qgrid=helfem::diatomic::twodquad::TwoDGrid(&basis,lquad);
  qgrid.compute_atoms(Z1,Z2);

  // Unique m values, sorted ascending
  Eigen::VectorXi mval_all(basis.get_mval());
  std::vector<int> mtmp(mval_all.data(), mval_all.data()+mval_all.size());
  std::sort(mtmp.begin(), mtmp.end());
  mtmp.erase(std::unique(mtmp.begin(), mtmp.end()), mtmp.end());
  Eigen::VectorXi muni = Eigen::Map<Eigen::VectorXi>(mtmp.data(), mtmp.size());

  // Exponents
  helfem::Vector expn = helfem::Vector::LinSpaced(nexp, log10(minexp), log10(maxexp));
  expn = (expn.array()*std::log(10.0)).exp().matrix();

  for(Eigen::Index im=0;im<muni.size();im++) {
    int m=muni(im);

    // Compute the projection of the atomic orbitals on the grid
    helfem::Matrix minimal_basis_orbitals;
    size_t nminimal = 0;
    {
      std::vector<helfem::Matrix> minimal_basis;
      for(int l=std::abs(m);l<=completeness;l++) {
        helfem::Matrix proj = qgrid.atomic_projection(l, m, helfem::diatomic::twodquad::PROBE_LEFT);
        if(proj.size()) {
          minimal_basis.push_back(proj);
          nminimal += proj.rows();
          printf("Left-hand atom has %i orbitals for m = %i from l = %i\n",(int) proj.rows(),m,l);
          fflush(stdout);
        }
      }
      for(int l=std::abs(m);l<=completeness;l++) {
        helfem::Matrix proj = qgrid.atomic_projection(l, m, helfem::diatomic::twodquad::PROBE_RIGHT);
        if(proj.size()) {
          minimal_basis.push_back(proj);
          nminimal += proj.rows();
          printf("Right-hand atom has %i orbitals for m = %i from l = %i\n",(int) proj.rows(),m,l);
          fflush(stdout);
        }
      }

      printf("We have %i minimal basis functions for m = %i\n",(int) nminimal, m);
      fflush(stdout);
      if(nminimal>0) {
        minimal_basis_orbitals = helfem::Matrix::Zero(nminimal, minimal_basis[0].cols());
        Eigen::Index ioff=0;
        for(size_t i = 0; i < minimal_basis.size();i++) {
          minimal_basis_orbitals.middleRows(ioff, minimal_basis[i].rows()) = minimal_basis[i];
          ioff += minimal_basis[i].rows();
        }
      }
    }

    // Inverse overlap matrix. Diagonalise the (real symmetric) overlap and
    // rebuild from the well-conditioned eigenpairs: Sinv = sum_i (1/l_i) v_i v_i^T.
    // This form is invariant under eigenvector sign, so no sign convention matters.
    std::function<helfem::Matrix(const helfem::Matrix & S)> form_Sinv = [](const helfem::Matrix & S) {
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(S);
      const helfem::Vector & Sval = es.eigenvalues();
      const helfem::Matrix & Svec = es.eigenvectors();
      // Find well-conditioned part
      helfem::Matrix result = helfem::Matrix::Zero(S.rows(), S.cols());
      for(Eigen::Index i=0;i<Sval.size();i++) {
        if(Sval(i)>=1e-6)
          result += (1.0/Sval(i)) * Svec.col(i) * Svec.col(i).transpose();
      }
      return result;
    };

    // Minimal-basis overlap and inverse overlap matrices
    helfem::Matrix Smin, Smin_inv;
    if(nminimal>0) {
      Smin = minimal_basis_orbitals*Sinv*minimal_basis_orbitals.transpose();

      helfem::Matrix dSmin = Smin - helfem::Matrix::Identity(Smin.rows(), Smin.cols());
      printf("Minimal basis overlap matrix for m = %i, difference from orthonormality\n",m);
      helfem::io::print_matrix("", dSmin);
      fflush(stdout);

      Smin_inv = form_Sinv(Smin);
    }

    // Compute the projection of the wave function on the minimal basis
    double alpha_proj = 0.0, beta_proj = 0.0;

    helfem::Matrix alpha_minbas;
    if(nminimal>0) {
      alpha_minbas = (minimal_basis_orbitals * Ca.leftCols(nela)).transpose();
      alpha_proj = (alpha_minbas * Smin_inv * alpha_minbas.transpose()).trace();
      printf("Projection of m = %i alpha orbitals on minimal basis is % .6f\n", m, alpha_proj);
      fflush(stdout);
    }
    helfem::Matrix beta_minbas;
    if(nelb>0 && nminimal>0) {
      beta_minbas = (minimal_basis_orbitals * Cb.leftCols(nelb)).transpose();
      beta_proj = (beta_minbas * Smin_inv * beta_minbas.transpose()).trace();
      printf("Projection of m = %i beta  orbitals on minimal basis is % .6f\n", m, beta_proj);
      fflush(stdout);
    }

    for(int l=std::abs(m);l<=completeness;l++) {
      static const std::string indices[]={"lh", "mid", "rh"};
      static const helfem::diatomic::twodquad::probe_t probes[]={helfem::diatomic::twodquad::PROBE_LEFT, helfem::diatomic::twodquad::PROBE_MIDDLE, helfem::diatomic::twodquad::PROBE_RIGHT};
      for(int icen=0;icen<3;icen++) {
        // Importance profile (no minimal basis): expn, alpha, beta
        helfem::Matrix I0 = helfem::Matrix::Zero(3, expn.size());
        // Importance profile: expn, alpha, beta
        helfem::Matrix I = helfem::Matrix::Zero(3, expn.size());
        // Finite element basis set completeness: expn, Y
        helfem::Matrix Y = helfem::Matrix::Zero(2, expn.size());

        std::string lcao;
        if(iprobe==0) {
          lcao="gto";
        } else if(iprobe==1) {
          lcao="sto";
        } else
          throw std::logic_error("Unknown probe\n");

        // Loop over batches of exponents
        Eigen::Index exponent_batch_size = 200;
        for(Eigen::Index exponent_batch = 0; exponent_batch <= expn.size()/exponent_batch_size; exponent_batch++) {
          Eigen::Index istart = exponent_batch_size*exponent_batch;
          Eigen::Index iend = std::min(exponent_batch_size*(exponent_batch+1), expn.size()) - 1;
          if(istart>iend)
            break;
          helfem::Vector expbatch(expn.segment(istart, iend-istart+1));

          // LCAO projection <\alpha|FEM>
          helfem::Matrix Plcao;
          if(iprobe==0) {
            Plcao=qgrid.gto_projection(l, m, expbatch, probes[icen]);
          } else if(iprobe==1) {
            Plcao=qgrid.sto_projection(l, m, expbatch, probes[icen]);
          } else
            throw std::logic_error("Unknown probe\n");

          // Completeness profile
          helfem::Vector Ysub = (Plcao*Sinv*Plcao.transpose()).diagonal();

          helfem::Matrix Plcao_t(Plcao.transpose());

          // Projections onto occupied orbitals
          helfem::Matrix Pa(Ca.leftCols(nela).transpose()*Plcao_t);
          helfem::Matrix Pb;
          if(nelb)
            Pb = Cb.leftCols(nelb).transpose()*Plcao_t;

          // Compute <\alpha|minimal basis>
          helfem::Matrix lcao_overlap_minbas;
          if(nminimal>0)
            lcao_overlap_minbas = minimal_basis_orbitals*Sinv*Plcao.transpose();

          // Compute the importance profile
#pragma omp parallel for
          for(Eigen::Index ix=0;ix<expbatch.size(); ix++) {
            // Output index
            Eigen::Index iout = istart+ix;

            // Importance profile without minimal basis
            I0(0, iout) = expbatch(ix);
            I0(1, iout) = Pa.col(ix).squaredNorm();
            if(nelb)
              I0(2, iout) = Pb.col(ix).squaredNorm();

            // Inverse of padded overlap matrix
            helfem::Matrix Spad_inv;
            if(nminimal>0) {
              // Construct padded overlap matrix:
              helfem::Matrix Spad = helfem::Matrix::Zero(Smin.rows()+1, Smin.cols()+1);
              // minimal basis
              Spad.topLeftCorner(Smin.rows(), Smin.cols()) = Smin;
              // minimal basis - added function
              Spad.block(0, Smin.cols(), Smin.rows(), 1) = lcao_overlap_minbas.col(ix);
              Spad.block(Smin.cols(), 0, 1, Smin.rows()) = lcao_overlap_minbas.col(ix).transpose();
              // added function - added function
              Spad(Smin.rows(), Smin.cols()) = Plcao_t.col(ix).dot(Sinv * Plcao_t.col(ix));
              // Inverse
              Spad_inv = form_Sinv(Spad);

              // Overlap of the orbitals with the scanning function
              helfem::Matrix alpha_pad(alpha_minbas.rows(), alpha_minbas.cols()+1);
              alpha_pad.leftCols(alpha_minbas.cols()) = alpha_minbas;
              alpha_pad.col(alpha_minbas.cols()) = Pa.col(ix);

              double alpha_padproj = (alpha_pad * Spad_inv * alpha_pad.transpose()).trace();
              I(0, iout) = expbatch(ix);
              I(1, iout) = alpha_padproj - alpha_proj;
            } else {
              for(Eigen::Index ir=0;ir<I.rows();ir++)
                I(ir, iout) = I0(ir, iout);
            }

            //printf("Projection of alpha orbitals on padded basis %e is % .6f\n", expbatch(ix), alpha_padproj);
            //printf("Projection increment is  % .e\n", alpha_padproj - alpha_proj);
            if(nelb>0 && nminimal>0) {
              helfem::Matrix beta_pad(beta_minbas.rows(), beta_minbas.cols()+1);
              beta_pad.leftCols(beta_minbas.cols()) = beta_minbas;
              beta_pad.col(beta_minbas.cols()) = Pb.col(ix);

              double beta_padproj = (beta_pad * Spad_inv * beta_pad.transpose()).trace();
              I(2, iout) = beta_padproj - beta_proj;
              //printf("Projection of beta orbitals on padded basis %e is % .6f\n", expbatch(ix), beta_padproj);
              //printf("Projection increment is  %e\n", beta_padproj-beta_proj);
            }

            // Save completeness profile
            Y(0, iout) = expbatch(ix);
            Y(1, iout) = Ysub(ix);
          }
        }

        // Save importance profile
        {
          std::ostringstream oss;
          oss << "importance0_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          helfem::io::write_raw_ascii(oss.str(), helfem::Matrix(I0.transpose()));
        }
        {
          std::ostringstream oss;
          oss << "importance_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          helfem::io::write_raw_ascii(oss.str(), helfem::Matrix(I.transpose()));
        }

        // Save completeness profile
        {
          std::ostringstream oss;
          oss << "completeness_" << lcao << "_" << indices[icen] << "_" << l << "_" << m << ".dat";
          helfem::io::write_raw_ascii(oss.str(), helfem::Matrix(Y.transpose()));
        }

        /*
          printf("*** l=%i m=%i ***\n",l,m);
          Slcao=qgrid.gto_overlap(l, m, expn, diatomic::twodquad::PROBE_LEFT);
          Slcao.print("lh gto overlap");
          Slcao=qgrid.gto_overlap(l, m, expn, diatomic::twodquad::PROBE_RIGHT);
          Slcao.print("rh gto overlap");

          Slcao=qgrid.sto_overlap(l, m, expn, diatomic::twodquad::PROBE_LEFT);
          Slcao.print("lh sto overlap");
          Slcao=qgrid.sto_overlap(l, m, expn, diatomic::twodquad::PROBE_RIGHT);
          Slcao.print("rh sto overlap");
          printf("\n");
        */
      }
    }
  }

  return 0;
}
