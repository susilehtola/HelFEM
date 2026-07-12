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
#include "dftgrid_purem.h"
#include "dftgrid.h"     // increment_gga
#include "chebyshev.h"
#include <ArmaEigen.h>
#include <algorithm>
#include <cmath>

namespace helfem {
  namespace diatomic {
    namespace dftgrid_purem {

      PureMDFTGridWorker::PureMDFTGridWorker() : basp(nullptr) {
      }

      PureMDFTGridWorker::~PureMDFTGridWorker() {
      }

      PureMDFTGridWorker::PureMDFTGridWorker(const helfem::diatomic::basis::TwoDBasis * basp_, int lang) : basp(basp_) {
        // nu (angular) grid only -- there is no phi grid.
        arma::vec cth_a, wang_a;
        chebyshev::chebyshev(lang, cth_a, wang_a);
        cth  = helfem::to_eigen(cth_a);
        wang = helfem::to_eigen(wang_a);

        // Distinct m values present in the basis
        const arma::ivec mv(basp->get_mval());
        for (size_t i = 0; i < mv.n_elem; i++) {
          const int m = (int) mv(i);
          if (std::find(mlist.begin(), mlist.end(), m) == mlist.end())
            mlist.push_back(m);
        }
        std::sort(mlist.begin(), mlist.end());
      }

      void PureMDFTGridWorker::compute_bf(size_t iel, size_t irad) {
        const Eigen::Index nang = cth.size();
        const bool need_df = do_grad || do_tau;

        const double r0    = basp->get_r(iel)(irad);
        const double wr    = basp->get_wrad(iel)(irad);
        const double Rhalf = basp->get_Rhalf();
        const double shmu  = std::sinh(r0);

        // Scale factors and the quadrature weight. Same measure as
        // twodquadrature: the phi integral is done analytically and yields
        // the explicit factor 2*pi.
        scale_r.resize(nang);
        inv_scale_r2.resize(nang);
        inv_scale_phi2.resize(nang);
        wtot = helfem::Vector::Zero(nang);
        for (Eigen::Index ia = 0; ia < nang; ia++) {
          // sin(nu) written to avoid cancellation near |cth| = 1
          const double sth  = std::sqrt(std::max((1.0 - cth(ia)) * (1.0 + cth(ia)), 0.0));
          // h_mu = h_nu = Rhalf sqrt(sinh^2 mu + sin^2 nu)
          const double hmu  = Rhalf * std::sqrt(shmu * shmu + sth * sth);
          // h_phi = Rhalf sinh(mu) sin(nu)
          const double hphi = Rhalf * shmu * sth;
          scale_r(ia)        = hmu;
          inv_scale_r2(ia)   = (hmu  > 0.0) ? 1.0 / (hmu * hmu)   : 0.0;
          inv_scale_phi2(ia) = (hphi > 0.0) ? 1.0 / (hphi * hphi) : 0.0;
          // sin(nu) is already contained in wang; the 2*pi is the analytic
          // phi integral.
          wtot(ia) = 2.0 * M_PI * wang(ia) * wr * std::pow(Rhalf, 3) * shmu
                      * (shmu * shmu + sth * sth);
        }

        // Per-m real basis functions (and in-plane derivatives if needed)
        const size_t nm = mlist.size();
        bf_ind_m.assign(nm, std::vector<Eigen::Index>());
        bf_m.assign(nm, helfem::Matrix());
        dr_m.assign(nm, helfem::Matrix());
        dth_m.assign(nm, helfem::Matrix());

        for (size_t im = 0; im < nm; im++) {
          const int m = mlist[im];
          const arma::uvec ind(basp->bf_list_dummy(iel, m));
          bf_ind_m[im].resize(ind.n_elem);
          for (size_t k = 0; k < ind.n_elem; k++)
            bf_ind_m[im][k] = (Eigen::Index) ind(k);
          const Eigen::Index nbf = (Eigen::Index) ind.n_elem;
          if (!nbf) continue;

          bf_m[im] = helfem::Matrix::Zero(nbf, nang);
          if (need_df) {
            dr_m[im]  = helfem::Matrix::Zero(nbf, nang);
            dth_m[im] = helfem::Matrix::Zero(nbf, nang);
          }

          for (Eigen::Index ia = 0; ia < nang; ia++) {
            const helfem::Matrix abf(helfem::to_eigen(basp->eval_bf(iel, irad, cth(ia), m)));
            bf_m[im].col(ia) = abf.transpose();
            if (need_df) {
              arma::mat dr_a, dth_a;
              basp->eval_df(iel, irad, cth(ia), m, dr_a, dth_a);
              dr_m[im].col(ia)  = helfem::to_eigen(dr_a).transpose();
              dth_m[im].col(ia) = helfem::to_eigen(dth_a).transpose();
            }
          }
        }
      }

      /// Gather the (idx, idx) sub-block of a full AO matrix.
      static helfem::Matrix gather_block(const helfem::Matrix & P, const std::vector<Eigen::Index> & idx) {
        const Eigen::Index n = (Eigen::Index) idx.size();
        helfem::Matrix out(n, n);
        for (Eigen::Index i = 0; i < n; i++)
          for (Eigen::Index j = 0; j < n; j++)
            out(i, j) = P(idx[i], idx[j]);
        return out;
      }

      void PureMDFTGridWorker::update_density(const helfem::Matrix & P) {
        if (!P.size())
          throw std::runtime_error("Error - density matrix is empty!\n");
        polarized = false;

        const Eigen::Index nang = cth.size();
        const size_t nm = mlist.size();

        rho = helfem::Matrix::Zero(1, nang);
        rho_m.assign(nm, helfem::Vector::Zero(nang));
        Pv_m.assign(nm, helfem::Matrix());

        for (size_t im = 0; im < nm; im++) {
          if (bf_ind_m[im].empty()) continue;
          const helfem::Matrix Pblk = gather_block(P, bf_ind_m[im]);
          Pv_m[im] = Pblk * bf_m[im];
          // rho_m(ia) = sum_i (P bf)(i,ia) bf(i,ia)
          rho_m[im] = (Pv_m[im].array() * bf_m[im].array()).colwise().sum().transpose();
          rho.row(0) += rho_m[im].transpose();
        }

        if (do_grad) {
          // Only the (mu, nu) components exist: d rho / d phi == 0 identically,
          // because |e^{i m phi}| = 1 makes rho phi-independent.
          grho  = helfem::Matrix::Zero(2, nang);
          sigma = helfem::Matrix::Zero(1, nang);
          for (size_t im = 0; im < nm; im++) {
            if (bf_ind_m[im].empty()) continue;
            grho.row(0) += 2.0 * (Pv_m[im].array() * dr_m[im].array()).colwise().sum().matrix();
            grho.row(1) += 2.0 * (Pv_m[im].array() * dth_m[im].array()).colwise().sum().matrix();
          }
          for (Eigen::Index ia = 0; ia < nang; ia++) {
            // h_mu == h_nu == scale_r
            grho(0, ia) /= scale_r(ia);
            grho(1, ia) /= scale_r(ia);
            sigma(0, ia) = grho(0, ia) * grho(0, ia) + grho(1, ia) * grho(1, ia);
          }
        }

        if (do_tau) {
          tau = helfem::Matrix::Zero(1, nang);
          Pv_dr_m.assign(nm, helfem::Matrix());
          Pv_dth_m.assign(nm, helfem::Matrix());
          for (size_t im = 0; im < nm; im++) {
            if (bf_ind_m[im].empty()) continue;
            const helfem::Matrix Pblk = gather_block(P, bf_ind_m[im]);
            Pv_dr_m[im]  = Pblk * dr_m[im];
            Pv_dth_m[im] = Pblk * dth_m[im];
            const double m2 = (double) (mlist[im] * mlist[im]);
            for (Eigen::Index ia = 0; ia < nang; ia++) {
              const double kr = (Pv_dr_m[im].col(ia).array()  * dr_m[im].col(ia).array()).sum()  * inv_scale_r2(ia);
              const double kt = (Pv_dth_m[im].col(ia).array() * dth_m[im].col(ia).array()).sum() * inv_scale_r2(ia);
              // Analytic phi contribution: d psi / d phi = i m psi, so
              // |d psi / d phi|^2 = m^2 |psi|^2 -> m^2 rho_m / h_phi^2.
              const double kp = m2 * rho_m[im](ia) * inv_scale_phi2(ia);
              tau(0, ia) += 0.5 * (kr + kt + kp);
            }
          }
        }

        if (do_lapl)
          throw std::logic_error("Laplacian not implemented.\n");
      }

      void PureMDFTGridWorker::update_density(const helfem::Matrix & Pa, const helfem::Matrix & Pb) {
        if (!Pa.size() || !Pb.size())
          throw std::runtime_error("Error - density matrix is empty!\n");
        polarized = true;

        const Eigen::Index nang = cth.size();
        const size_t nm = mlist.size();

        rho = helfem::Matrix::Zero(2, nang);
        rhoa_m.assign(nm, helfem::Vector::Zero(nang));
        rhob_m.assign(nm, helfem::Vector::Zero(nang));
        Pav_m.assign(nm, helfem::Matrix());
        Pbv_m.assign(nm, helfem::Matrix());

        for (size_t im = 0; im < nm; im++) {
          if (bf_ind_m[im].empty()) continue;
          const helfem::Matrix Pablk = gather_block(Pa, bf_ind_m[im]);
          const helfem::Matrix Pbblk = gather_block(Pb, bf_ind_m[im]);
          Pav_m[im] = Pablk * bf_m[im];
          Pbv_m[im] = Pbblk * bf_m[im];
          rhoa_m[im] = (Pav_m[im].array() * bf_m[im].array()).colwise().sum().transpose();
          rhob_m[im] = (Pbv_m[im].array() * bf_m[im].array()).colwise().sum().transpose();
          rho.row(0) += rhoa_m[im].transpose();
          rho.row(1) += rhob_m[im].transpose();
        }

        if (do_grad) {
          // rows 0,1: alpha (mu, nu); rows 2,3: beta (mu, nu)
          grho  = helfem::Matrix::Zero(4, nang);
          sigma = helfem::Matrix::Zero(3, nang);
          for (size_t im = 0; im < nm; im++) {
            if (bf_ind_m[im].empty()) continue;
            grho.row(0) += 2.0 * (Pav_m[im].array() * dr_m[im].array()).colwise().sum().matrix();
            grho.row(1) += 2.0 * (Pav_m[im].array() * dth_m[im].array()).colwise().sum().matrix();
            grho.row(2) += 2.0 * (Pbv_m[im].array() * dr_m[im].array()).colwise().sum().matrix();
            grho.row(3) += 2.0 * (Pbv_m[im].array() * dth_m[im].array()).colwise().sum().matrix();
          }
          for (Eigen::Index ia = 0; ia < nang; ia++) {
            for (int k = 0; k < 4; k++)
              grho(k, ia) /= scale_r(ia);
            const double gam = grho(0, ia), gan = grho(1, ia);
            const double gbm = grho(2, ia), gbn = grho(3, ia);
            sigma(0, ia) = gam * gam + gan * gan;
            sigma(1, ia) = gam * gbm + gan * gbn;
            sigma(2, ia) = gbm * gbm + gbn * gbn;
          }
        }

        if (do_tau) {
          tau = helfem::Matrix::Zero(2, nang);
          Pav_dr_m.assign(nm, helfem::Matrix());
          Pav_dth_m.assign(nm, helfem::Matrix());
          Pbv_dr_m.assign(nm, helfem::Matrix());
          Pbv_dth_m.assign(nm, helfem::Matrix());
          for (size_t im = 0; im < nm; im++) {
            if (bf_ind_m[im].empty()) continue;
            const helfem::Matrix Pablk = gather_block(Pa, bf_ind_m[im]);
            const helfem::Matrix Pbblk = gather_block(Pb, bf_ind_m[im]);
            Pav_dr_m[im]  = Pablk * dr_m[im];
            Pav_dth_m[im] = Pablk * dth_m[im];
            Pbv_dr_m[im]  = Pbblk * dr_m[im];
            Pbv_dth_m[im] = Pbblk * dth_m[im];
            const double m2 = (double) (mlist[im] * mlist[im]);
            for (Eigen::Index ia = 0; ia < nang; ia++) {
              const double kar = (Pav_dr_m[im].col(ia).array()  * dr_m[im].col(ia).array()).sum()  * inv_scale_r2(ia);
              const double kat = (Pav_dth_m[im].col(ia).array() * dth_m[im].col(ia).array()).sum() * inv_scale_r2(ia);
              const double kap = m2 * rhoa_m[im](ia) * inv_scale_phi2(ia);
              const double kbr = (Pbv_dr_m[im].col(ia).array()  * dr_m[im].col(ia).array()).sum()  * inv_scale_r2(ia);
              const double kbt = (Pbv_dth_m[im].col(ia).array() * dth_m[im].col(ia).array()).sum() * inv_scale_r2(ia);
              const double kbp = m2 * rhob_m[im](ia) * inv_scale_phi2(ia);
              tau(0, ia) += 0.5 * (kar + kat + kap);
              tau(1, ia) += 0.5 * (kbr + kbt + kbp);
            }
          }
        }

        if (do_lapl)
          throw std::logic_error("Laplacian not implemented.\n");
      }

      double PureMDFTGridWorker::compute_Ekin() const {
        double ekin = 0.0;
        if (!do_tau) return ekin;
        for (Eigen::Index ip = 0; ip < wtot.size(); ip++) {
          if (!polarized)
            ekin += wtot(ip) * tau(0, ip);
          else
            ekin += wtot(ip) * (tau(0, ip) + tau(1, ip));
        }
        return ekin;
      }

      /// Accumulate one m block's XC contribution into the AO matrix H.
      /// `spin` selects the vxc / vsigma rows (0 = restricted or alpha,
      /// 1 = beta); `gr_rows` gives the (mu, nu) gradient rows for this spin.
      void PureMDFTGridWorker::eval_Fxc(helfem::Matrix & H) const {
        if (polarized)
          throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
        const Eigen::Index nang = cth.size();

        for (size_t im = 0; im < mlist.size(); im++) {
          const std::vector<Eigen::Index> & idx = bf_ind_m[im];
          if (idx.empty()) continue;
          const Eigen::Index nbf = (Eigen::Index) idx.size();
          helfem::Matrix Hm = helfem::Matrix::Zero(nbf, nbf);

          // LDA
          {
            helfem::Vector v = vxc.row(0).transpose();
            v.array() *= wtot.array();
            helfem::dftgrid_common::increment_lda<double>(Hm, v, bf_m[im]);
          }

          if (do_gga) {
            const helfem::Vector vs = vsigma.row(0).transpose();
            // increment_gga wants (npts x 3); the phi column is identically
            // zero, so its derivative matrix never contributes.
            helfem::Matrix gr = helfem::Matrix::Zero(nang, 3);
            for (Eigen::Index ia = 0; ia < nang; ia++) {
              const double f = 2.0 * wtot(ia) * vs(ia) / scale_r(ia);
              gr(ia, 0) = f * grho(0, ia);
              gr(ia, 1) = f * grho(1, ia);
            }
            dftgrid::increment_gga<double>(Hm, gr, bf_m[im], dr_m[im], dth_m[im], bf_m[im]);
          }

          if (do_mgga_t) {
            helfem::Vector vtl = 0.5 * vtau.row(0).transpose();
            vtl.array() *= wtot.array();
            helfem::Vector wr = vtl.array() * inv_scale_r2.array();
            helfem::dftgrid_common::increment_lda<double>(Hm, wr, dr_m[im]);
            helfem::dftgrid_common::increment_lda<double>(Hm, wr, dth_m[im]);
            // Analytic phi term: bf_phi = i m bf, and (i m bf)(i m bf)^dagger
            // = m^2 bf bf^T, so this is an LDA-type increment on bf itself
            // with the weight vtl m^2 / h_phi^2.
            const double m2 = (double) (mlist[im] * mlist[im]);
            if (m2 != 0.0) {
              helfem::Vector wp = vtl.array() * inv_scale_phi2.array() * m2;
              helfem::dftgrid_common::increment_lda<double>(Hm, wp, bf_m[im]);
            }
          }

          for (Eigen::Index i = 0; i < nbf; i++)
            for (Eigen::Index j = 0; j < nbf; j++)
              H(idx[i], idx[j]) += Hm(i, j);
        }
      }

      void PureMDFTGridWorker::eval_Fxc(helfem::Matrix & Ha, helfem::Matrix & Hb, bool beta) const {
        if (!polarized)
          throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
        const Eigen::Index nang = cth.size();

        for (size_t im = 0; im < mlist.size(); im++) {
          const std::vector<Eigen::Index> & idx = bf_ind_m[im];
          if (idx.empty()) continue;
          const Eigen::Index nbf = (Eigen::Index) idx.size();
          const double m2 = (double) (mlist[im] * mlist[im]);

          helfem::Matrix Hma = helfem::Matrix::Zero(nbf, nbf);
          helfem::Matrix Hmb = beta ? helfem::Matrix::Zero(nbf, nbf) : helfem::Matrix();

          // LDA
          {
            helfem::Vector va = vxc.row(0).transpose();
            va.array() *= wtot.array();
            helfem::dftgrid_common::increment_lda<double>(Hma, va, bf_m[im]);
            if (beta) {
              helfem::Vector vb = vxc.row(1).transpose();
              vb.array() *= wtot.array();
              helfem::dftgrid_common::increment_lda<double>(Hmb, vb, bf_m[im]);
            }
          }

          if (do_gga) {
            const helfem::Vector vs_aa = vsigma.row(0).transpose();
            const helfem::Vector vs_ab = vsigma.row(1).transpose();
            helfem::Matrix gr_a = helfem::Matrix::Zero(nang, 3);
            for (Eigen::Index ia = 0; ia < nang; ia++) {
              const double f = wtot(ia) / scale_r(ia);
              gr_a(ia, 0) = f * (2.0 * vs_aa(ia) * grho(0, ia) + vs_ab(ia) * grho(2, ia));
              gr_a(ia, 1) = f * (2.0 * vs_aa(ia) * grho(1, ia) + vs_ab(ia) * grho(3, ia));
            }
            dftgrid::increment_gga<double>(Hma, gr_a, bf_m[im], dr_m[im], dth_m[im], bf_m[im]);

            if (beta) {
              const helfem::Vector vs_bb = vsigma.row(2).transpose();
              helfem::Matrix gr_b = helfem::Matrix::Zero(nang, 3);
              for (Eigen::Index ia = 0; ia < nang; ia++) {
                const double f = wtot(ia) / scale_r(ia);
                gr_b(ia, 0) = f * (2.0 * vs_bb(ia) * grho(2, ia) + vs_ab(ia) * grho(0, ia));
                gr_b(ia, 1) = f * (2.0 * vs_bb(ia) * grho(3, ia) + vs_ab(ia) * grho(1, ia));
              }
              dftgrid::increment_gga<double>(Hmb, gr_b, bf_m[im], dr_m[im], dth_m[im], bf_m[im]);
            }
          }

          if (do_mgga_t) {
            helfem::Vector vtl_a = 0.5 * vtau.row(0).transpose();
            vtl_a.array() *= wtot.array();
            helfem::Vector wra = vtl_a.array() * inv_scale_r2.array();
            helfem::dftgrid_common::increment_lda<double>(Hma, wra, dr_m[im]);
            helfem::dftgrid_common::increment_lda<double>(Hma, wra, dth_m[im]);
            if (m2 != 0.0) {
              helfem::Vector wpa = vtl_a.array() * inv_scale_phi2.array() * m2;
              helfem::dftgrid_common::increment_lda<double>(Hma, wpa, bf_m[im]);
            }
            if (beta) {
              helfem::Vector vtl_b = 0.5 * vtau.row(1).transpose();
              vtl_b.array() *= wtot.array();
              helfem::Vector wrb = vtl_b.array() * inv_scale_r2.array();
              helfem::dftgrid_common::increment_lda<double>(Hmb, wrb, dr_m[im]);
              helfem::dftgrid_common::increment_lda<double>(Hmb, wrb, dth_m[im]);
              if (m2 != 0.0) {
                helfem::Vector wpb = vtl_b.array() * inv_scale_phi2.array() * m2;
                helfem::dftgrid_common::increment_lda<double>(Hmb, wpb, bf_m[im]);
              }
            }
          }

          for (Eigen::Index i = 0; i < nbf; i++)
            for (Eigen::Index j = 0; j < nbf; j++) {
              Ha(idx[i], idx[j]) += Hma(i, j);
              if (beta) Hb(idx[i], idx[j]) += Hmb(i, j);
            }
        }
      }

      // ---------------------------------------------------------------------

      PureMDFTGrid::PureMDFTGrid() : basp(nullptr), lang(0) {
      }

      PureMDFTGrid::PureMDFTGrid(const helfem::diatomic::basis::TwoDBasis * basp_, int lang_) : basp(basp_), lang(lang_) {
        arma::vec cth, wang;
        chebyshev::chebyshev(lang, cth, wang);
        printf("Pure-m DFT grid: nu rule of order l=%i has %i points; the phi integral is analytic (2 pi).\n",
                lang, (int) wang.n_elem);
      }

      PureMDFTGrid::~PureMDFTGrid() {
      }

      void PureMDFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars,
                                   const helfem::Matrix & P, helfem::Matrix & H_e,
                                   double & Exc, double & Nel, double & Ekin, double thr) {
        helfem::Matrix H = helfem::Matrix::Zero(basp->Ndummy(), basp->Ndummy());
        double exc = 0.0, ekin = 0.0, nel = 0.0;
        {
          PureMDFTGridWorker grid(basp, lang);
          grid.check_grad_tau_lapl(x_func, c_func);

          for (size_t iel = 0; iel < basp->get_rad_Nel(); iel++) {
            for (size_t irad = 0; irad < basp->get_r(iel).n_elem; irad++) {
              grid.compute_bf(iel, irad);
              grid.update_density(P);
              nel  += grid.compute_Nel();
              ekin += grid.compute_Ekin();

              grid.init_xc();
              if (x_func > 0) grid.compute_xc(x_func, x_pars, thr);
              if (c_func > 0) grid.compute_xc(c_func, c_pars, thr);

              exc += grid.eval_Exc();
              grid.eval_Fxc(H);
            }
          }
        }
        Exc = exc; Ekin = ekin; Nel = nel;
        H_e = helfem::to_eigen(basp->remove_boundaries(helfem::to_arma(H)));
      }

      void PureMDFTGrid::eval_Fxc(int x_func, const helfem::Vector & x_pars, int c_func, const helfem::Vector & c_pars,
                                   const helfem::Matrix & Pa, const helfem::Matrix & Pb,
                                   helfem::Matrix & Ha_e, helfem::Matrix & Hb_e,
                                   double & Exc, double & Nel, double & Ekin, bool beta, double thr) {
        helfem::Matrix Ha = helfem::Matrix::Zero(basp->Ndummy(), basp->Ndummy());
        helfem::Matrix Hb = helfem::Matrix::Zero(basp->Ndummy(), basp->Ndummy());
        double exc = 0.0, ekin = 0.0, nel = 0.0;
        {
          PureMDFTGridWorker grid(basp, lang);
          grid.check_grad_tau_lapl(x_func, c_func);

          for (size_t iel = 0; iel < basp->get_rad_Nel(); iel++) {
            for (size_t irad = 0; irad < basp->get_r(iel).n_elem; irad++) {
              grid.compute_bf(iel, irad);
              grid.update_density(Pa, Pb);
              nel  += grid.compute_Nel();
              ekin += grid.compute_Ekin();

              grid.init_xc();
              if (x_func > 0) grid.compute_xc(x_func, x_pars, thr);
              if (c_func > 0) grid.compute_xc(c_func, c_pars, thr);

              exc += grid.eval_Exc();
              grid.eval_Fxc(Ha, Hb, beta);
            }
          }
        }
        Exc = exc; Ekin = ekin; Nel = nel;
        Ha_e = helfem::to_eigen(basp->remove_boundaries(helfem::to_arma(Ha)));
        if (beta)
          Hb_e = helfem::to_eigen(basp->remove_boundaries(helfem::to_arma(Hb)));
      }

    }
  }
}
