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
#ifndef HELFEM_DFTGRID_COMMON_H
#define HELFEM_DFTGRID_COMMON_H

// Shared DFT-grid worker plumbing: libxc dispatch, xc buffer
// allocation, gradient/tau/lapl need detection, energy accumulation.
// Extracted from three near-identical copies in src/atomic/dftgrid.cpp,
// src/sadatom/dftgrid.cpp, src/diatomic/dftgrid.cpp. The
// geometry-specific bits (compute_bf, update_density, eval_Fxc,
// compute_Nel, etc.) stay in the derived classes.

#include <Matrix.h>
#include <deque>
#include <memory>
#include <vector>
#include <sstream>
#include <stdexcept>

/// libxc's functional handle. Forward declared -- exactly as libxc's own
/// header does -- so that <xc.h> stays out of every consumer of this one.
struct xc_func_type;

namespace helfem {
  namespace dftgrid_common {
    /// Base class holding the shared XC state and libxc-facing
    /// plumbing for the three DFTGridWorker variants. Geometry-specific
    /// derived classes inherit and add their own basis-function buffers,
    /// density update, and Fxc reassembly.
    class DFTGridWorkerBase {
    protected:
      /// Total quadrature weights on the current element's grid (Npts)
      helfem::Vector wtot;

      /// Is gradient needed?
      bool do_grad;
      /// Is kinetic energy density needed?
      bool do_tau;
      /// Is laplacian needed?
      bool do_lapl;
      /// Spin-polarized calculation?
      bool polarized;

      /// GGA functional used? (Set in compute_xc, only affects eval_Fxc)
      bool do_gga;
      /// Meta-GGA tau used? (Set in compute_xc, only affects eval_Fxc)
      bool do_mgga_t;
      /// Meta-GGA lapl used? (Set in compute_xc, only affects eval_Fxc)
      bool do_mgga_l;

      // LDA
      /// Density, Nrho x Npts
      helfem::Matrix rho;
      /// Energy density, Npts
      helfem::Vector exc;
      /// Functional derivative wrt density
      helfem::Matrix vxc;

      // GGA
      /// Dot products of density gradient
      helfem::Matrix sigma;
      /// Functional derivative wrt density gradient
      helfem::Matrix vsigma;

      // Meta-GGA
      /// Laplacian of density
      helfem::Matrix lapl;
      /// Kinetic energy density
      helfem::Matrix tau;
      /// Functional derivative wrt laplacian
      helfem::Matrix vlapl;
      /// Functional derivative wrt kinetic energy density
      helfem::Matrix vtau;

      /// Coefficient of the basis-function gradient pair, i.e. the
      /// vector V of  K = sum_g w [ u chi chi + V . grad(chi chi) ].
      /// 3 x Npts unpolarized, 6 x Npts polarized (alpha rows first).
      /// The ground state fills it from vsigma via build_vgrad; a
      /// response fills it from the kernel chain rule, where V is a
      /// combination of grad(rho) AND grad(drho) and so cannot be
      /// written as a scalar times grad(rho).
      helfem::Matrix vgrad;

      // Response kernel (second derivatives), in libxc's own packed
      // layout: 1 / 3 rows unpolarized / polarized for the rho-rho type
      // blocks, 6 rows for the ones carrying a sigma index.
      /// d^2 e_xc / d rho^2
      helfem::Matrix v2rho2;
      /// GGA blocks
      helfem::Matrix v2rhosigma, v2sigma2;
      /// meta-GGA (kinetic energy density) blocks
      helfem::Matrix v2rhotau, v2sigmatau, v2tau2;

      /// One initialized libxc functional, together with the answers to
      /// the constant questions HelFEM asks about it.
      ///
      /// None of this changes over a run, but all of it used to be
      /// rebuilt constantly: compute_xc and compute_fxc did their own
      /// xc_func_init / xc_func_end on every element, is_gga_mgga did
      /// another pair on every call, and has_exc -- which does one too --
      /// sat inside the NaN guard's loop over QUADRATURE POINTS, so a
      /// libxc functional was constructed and destroyed once per grid
      /// point per element per functional to answer a question whose
      /// answer never changed.
      struct Functional {
        int id = 0;
        int nspin = 0;
        double thr = 0.0;
        helfem::Vector params;
        bool gga = false, mgga_t = false, mgga_l = false, have_exc = false;
        /// The deleter carries the xc_func_end call, so it is bound in
        /// the .cpp and <xc.h> is not needed here.
        std::shared_ptr<xc_func_type> func;
      };
      /// A deque rather than a vector: functional() hands out references
      /// into this, and a deque never invalidates them when it grows.
      std::deque<Functional> funcs;

      /// The initialized functional and its properties for this id, spin
      /// channel, density threshold and parameter set, built on first use
      /// and kept for the lifetime of the worker. Workers are per-thread,
      /// so this needs no locking.
      const Functional & functional(int func_id, const helfem::Vector & params,
                                    double thr);

    public:
      DFTGridWorkerBase();
      virtual ~DFTGridWorkerBase();

      /// Check necessity of computing gradient / tau / laplacian for the
      /// given exchange + correlation functional ids.
      void check_grad_tau_lapl(int x_func, int c_func);
      /// Explicit override of the do_grad / do_tau / do_lapl flags
      void set_grad_tau_lapl(bool grad, bool tau, bool lapl);

      /// Which orders of the basis functions the current functional
      /// actually needs. Lets a caller prime a basis-value cache for the
      /// right orders before entering the parallel grid loop, instead of
      /// guessing or computing derivatives an LDA will never look at.
      bool needs_grad() const { return do_grad; }
      bool needs_lapl() const { return do_lapl; }

      /// Initialise vxc / vsigma / vtau / vlapl buffers with the
      /// correct shape and zero exc.
      void init_xc();

      /// Evaluate int wtot * exc * rho_total
      double eval_Exc() const;

      /// Integrate the (total) electron density over the current
      /// element grid: int wtot * rho. Geometry-independent -- reads
      /// only the shared wtot / rho / polarized state.
      double compute_Nel() const;

      /// Compute libxc functional contribution and add to exc / vxc /
      /// vsigma / vtau / vlapl. pot=true also computes potentials.
      void compute_xc(int func_id, const helfem::Vector & params, double thr, bool pot = true);

      /// Initialise the kernel buffer. Call between update_density and
      /// compute_fxc, the same way init_xc precedes compute_xc.
      void init_fxc();

      /// Accumulate one functional's second derivatives at the density
      /// update_density last set: the density-density block for every
      /// rung, plus the sigma blocks for a GGA and the tau blocks for a
      /// meta-GGA. Works for LDA, GGA and meta-GGA functionals alike --
      /// libxc has the entry point in every case.
      void compute_fxc(int func_id, const helfem::Vector & params, double thr);

      /// Build the ground-state gradient coefficient V from vsigma and
      /// the density gradient, so that every assembly site contracts a
      /// general vector field rather than reconstructing it from vsigma.
      void build_vgrad(const helfem::Matrix & grho);

      /// Overwrite the potential buffers with the response potential
      /// f_xc . drho, so that the derived worker's eval_Fxc assembles the
      /// linear response of the XC matrix instead of the XC matrix
      /// itself -- the assembly is the same integral either way, and
      /// this is what lets the response reuse it verbatim.
      ///
      /// drho, dgrad_rho and dtau carry the perturbation's density,
      /// density gradient and kinetic energy density in the same layouts
      /// as rho, grho and tau; grho is the REFERENCE gradient, which the
      /// gradient channel needs alongside the perturbed one. Pass empty
      /// matrices for the channels a given functional does not use.
      ///
      /// The chain rule assembled here is exact for every rung: the
      /// gradient channel is
      ///   V = 2 vsigma grad(drho)
      ///       + [4 v2sigma2 (grad rho . grad drho) + 2 v2rhosigma drho
      ///          + 2 v2sigmatau dtau] grad(rho),
      /// which is why V must be stored as a vector (vgrad) rather than
      /// as a scalar multiplying the reference gradient.
      void set_response_potential(const helfem::Matrix & drho,
                                  const helfem::Matrix & grho,
                                  const helfem::Matrix & dgrad_rho,
                                  const helfem::Matrix & dtau);
    };

    /// Same as increment_lda below, but for a basis whose real and
    /// imaginary parts have already been split.
    ///
    /// H is real, so for complex f the imaginary half of f diag(v) f^H is
    /// built and thrown away. With f = a + i b,
    ///     Re( f diag(v) f^H ) = a diag(v) a^T + b diag(v) b^T,
    /// two real products instead of one complex one -- half the multiplies,
    /// on the better-optimised real kernels.
    ///
    /// Taking a and b as arguments rather than splitting f here is the
    /// point: extracting .real()/.imag() per call costs about what the
    /// halved flops save, so the split has to be hoisted to where the
    /// basis is built.
    inline void increment_lda_split(helfem::Matrix & H, const helfem::Vector & vxc,
                                    const helfem::Matrix & a, const helfem::Matrix & b) {
      if(a.cols() != vxc.size() || b.cols() != vxc.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << a.cols() << " and potential values " << vxc.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      helfem::Matrix av(a), bv(b);
      for(Eigen::Index j=0;j<av.cols();j++) {
        av.col(j) *= vxc(j);
        bv.col(j) *= vxc(j);
      }
      H.noalias() += av * a.transpose();
      H.noalias() += bv * b.transpose();
    }

    /// BLAS routine for LDA-type quadrature: accumulate
    /// H += Re( (f .* vxc) * f^T ), i.e. the weighted outer product of
    /// the basis-function values f (Nbf x Npts) against themselves with
    /// per-point potential weights vxc (1 x Npts). Geometry-independent
    /// -- shared verbatim by all three DFTGridWorker variants (real f
    /// for sadatom, complex f for atomic/diatomic).
    template<typename T> void increment_lda(helfem::Matrix & H, const helfem::Vector & vxc,
                                             const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f) {
      if(f.cols() != vxc.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << f.cols() << " and potential values " << vxc.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      if(H.rows() != f.rows() || H.cols() != f.rows()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Fock matrix (" << H.rows() << "," << H.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }

      // Weighted helper: fhlp(:,j) = f(:,j) * vxc(j). Then
      // H += Re( fhlp * f^H ). The conjugate transpose of a complex
      // matrix is .adjoint() (== .transpose() for the real, sadatom,
      // instantiation).
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fhlp = f;
      for(Eigen::Index j=0;j<fhlp.cols();j++)
        fhlp.col(j) *= vxc(j);
      H += (fhlp * f.adjoint()).real();
    }

    /// GGA accumulation, shared by all three geometries.
    ///
    /// gn is (npts x ncomp): one column of gradient weights per spatial
    /// component -- one for the spherically averaged atom (radial only),
    /// three for the atomic and diatomic grids. ga/gb hold the matching
    /// component derivatives of the basis, split into real and imaginary
    /// parts; pass gb empty for a real basis, where the imaginary work
    /// drops out entirely.
    ///
    /// With f = a + i b and the gradient-weighted helper gamma likewise
    /// split, Re( gamma f^H + f gamma^H ) = X + X^T with
    ///     X = Re(gamma) a^T + Im(gamma) b^T,
    /// so the two complex products of the old formulation collapse to two
    /// real ones plus a symmetrisation -- the symmetry was being computed
    /// and half-discarded along with the imaginary part.
    inline void increment_gga_split(helfem::Matrix & H, const helfem::Matrix & gn,
                                    const helfem::Matrix & a, const helfem::Matrix & b,
                                    const std::vector<const helfem::Matrix *> & ga,
                                    const std::vector<const helfem::Matrix *> & gb) {
      const bool complex_basis = !gb.empty();
      if(gn.cols() != (Eigen::Index) ga.size()) {
        std::ostringstream oss;
        oss << "Grad rho has " << gn.cols() << " columns but " << ga.size()
            << " gradient components were given!\n";
        throw std::runtime_error(oss.str());
      }
      if(complex_basis && gb.size() != ga.size())
        throw std::runtime_error("Real and imaginary gradient component counts differ!\n");
      if(H.rows() != a.rows() || H.cols() != a.rows())
        throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");

      helfem::Matrix gre(helfem::Matrix::Zero(a.rows(), a.cols()));
      helfem::Matrix gim;
      if(complex_basis) gim = helfem::Matrix::Zero(a.rows(), a.cols());
      for(size_t c=0;c<ga.size();c++)
        for(Eigen::Index j=0;j<gre.cols();j++) {
          gre.col(j) += gn(j,(Eigen::Index) c) * ga[c]->col(j);
          if(complex_basis) gim.col(j) += gn(j,(Eigen::Index) c) * gb[c]->col(j);
        }

      helfem::Matrix X(gre * a.transpose());
      if(complex_basis) X.noalias() += gim * b.transpose();
      H += X;
      H += X.transpose();
    }

    /// Laplacian meta-GGA accumulation for an already-split basis.
    ///
    /// H is real, so with f = a + i b and l = c + i d,
    ///     Re( f diag(v) l^H + l diag(v) f^H ) = X + X^T,
    ///     X = a diag(v) c^T + b diag(v) d^T,
    /// i.e. two real products and a symmetrisation, against two complex
    /// products (eight real ones) before.
    inline void increment_mgga_lapl_split(helfem::Matrix & H, const helfem::Vector & vlapl,
                                          const helfem::Matrix & a, const helfem::Matrix & b,
                                          const helfem::Matrix & c, const helfem::Matrix & d) {
      if(a.cols() != vlapl.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << a.cols() << " and potential values " << vlapl.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      helfem::Matrix av(a), bv(b);
      for(Eigen::Index j=0;j<av.cols();j++) {
        av.col(j) *= vlapl(j);
        bv.col(j) *= vlapl(j);
      }
      helfem::Matrix X(av * c.transpose());
      X.noalias() += bv * d.transpose();
      H += X;
      H += X.transpose();
    }

    /// BLAS routine for the Laplacian part of meta-GGA quadrature:
    ///   H += f diag(w vlapl) l^dagger + l diag(w vlapl) f^dagger
    template<typename T> void increment_mgga_lapl(helfem::Matrix & H, const helfem::Vector & vlapl, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & f, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & l) {
      if(f.cols() != vlapl.size()) {
        std::ostringstream oss;
        oss << "Number of functions " << f.cols() << " and potential values " << vlapl.size() << " do not match!\n";
        throw std::runtime_error(oss.str());
      }
      if(H.rows() != f.rows() || H.cols() != f.rows()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Fock matrix (" << H.rows() << "," << H.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }
      if(l.rows() != f.rows() || l.cols() != f.cols()) {
        std::ostringstream oss;
        oss << "Size of basis function (" << f.rows() << "," << f.cols() << ") and Laplacian matrix (" << l.rows() << "," << l.cols() << ") doesn't match!\n";
        throw std::runtime_error(oss.str());
      }

      // Form helper matrix
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fhlp = f;
      for(Eigen::Index j=0;j<fhlp.cols();j++)
        fhlp.col(j) *= vlapl(j);
      H += (fhlp*l.adjoint() + l*fhlp.adjoint()).real();
    }
  }
}

#endif
