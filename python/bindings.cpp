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
 *
 *
 * Python bindings for HelFEM (step b of the PySCF interop arc).
 *
 * Exposes the minimum API a PySCF-style driver needs:
 *
 *   AtomicBasis(Z, lmax, mmax, ...) -> opaque basis object
 *     .Nbf()           -> number of basis functions
 *     .overlap()       -> S matrix (Nbf x Nbf)
 *     .kinetic()       -> T matrix (kinetic + centrifugal already in)
 *     .nuclear()       -> V matrix (Z-scaled, sign-included)
 *     .hcore()         -> T + V (convenience)
 *     .get_jk(P)       -> (J, K) tuple from a density matrix
 *     .coulomb(P)      -> J
 *     .exchange(P)     -> -K  (HelFEM stores K with HF minus baked in;
 *                              this matches what PySCF's get_jk returns)
 *
 * The intent is to let a Python driver build the AtomicBasis once,
 * then patch a PySCF SCF object's get_hcore / get_ovlp / get_jk hooks
 * to call these. CASSCF / CC / MP2 / etc. then flow from PySCF on top.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "../src/atomic/basis.h"
#include "../src/atomic/TwoDBasis.h"
#include "../libhelfem/include/Matrix.h"
#include "../libhelfem/include/PolynomialBasis.h"
#include "../libhelfem/include/ModelPotential.h"

namespace py = pybind11;

namespace helfem_py {

  // Copy helfem::Matrix -> C-order numpy. Eigen is column-major by
  // default, so the loop transposes implicitly and the resulting numpy
  // array is C-contiguous (rows-fast).
  static py::array_t<double> eigen_to_numpy(const helfem::Matrix & M) {
    std::vector<py::ssize_t> shape{(py::ssize_t) M.rows(), (py::ssize_t) M.cols()};
    py::array_t<double> out(shape);
    double * dst = out.mutable_data();
    for (Eigen::Index i = 0; i < M.rows(); ++i)
      for (Eigen::Index j = 0; j < M.cols(); ++j)
        dst[i * M.cols() + j] = M(i, j);
    return out;
  }

  // C-contiguous numpy -> helfem::Matrix (forces double conversion).
  static helfem::Matrix numpy_to_eigen(
      py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() != 2)
      throw std::runtime_error("numpy_to_eigen: expected 2-D ndarray.");
    const py::ssize_t nr = a.shape(0);
    const py::ssize_t nc = a.shape(1);
    helfem::Matrix M((Eigen::Index) nr, (Eigen::Index) nc);
    const double * src = a.data();
    for (py::ssize_t i = 0; i < nr; ++i)
      for (py::ssize_t j = 0; j < nc; ++j)
        M((Eigen::Index) i, (Eigen::Index) j) = src[i * nc + j];
    return M;
  }

  /// PySCF-facing wrapper around atomic::basis::TwoDBasis. Holds the basis
  /// + caches the two-electron integral tensors after first use.
  class AtomicBasis {
   public:
    AtomicBasis(int Z, int lmax, int mmax,
                int primbas, int nnodes,
                int nelem, double Rmax,
                int igrid, double zexp,
                int finitenuc, double Rrms)
        : Z_(Z), lmax_(lmax), mmax_(mmax), tei_done_(false) {
      using namespace helfem;
      // Polynomial primitive (LIP=4, HIP=5, HIP2=8, HIP3=9, Legendre=3).
      poly_ = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
          polynomial_basis::make_basis(primbas, nnodes));
      // Angular (l, m) list for shells up to (lmax, mmax).
      Eigen::VectorXi lval, mval;
      atomic::basis::angular_basis(lmax, mmax, lval, mval);
      // Radial element boundary grid.
      const helfem::Vector bval = atomic::basis::form_grid(
          (modelpotential::nuclear_model_t) finitenuc,
          Rrms, nelem, Rmax, igrid, zexp,
          /*Nelem0=*/0, /*igrid0=*/igrid, /*zexp0=*/zexp,
          Z, /*Zl=*/0, /*Zr=*/0, /*Rhalf=*/0.0);
      // Quadrature density (matches atomic/main.cpp).
      const int Nquad = 5 * poly_->nbf();
      basis_ = atomic::basis::TwoDBasis(
          Z, (modelpotential::nuclear_model_t) finitenuc, Rrms,
          poly_, /*zeroder=*/false,
          Nquad, bval, lval, mval, /*Zl=*/0, /*Zr=*/0, /*Rhalf=*/0.0);
    }

    int Z() const { return Z_; }
    size_t Nbf()  const { return basis_.Nbf(); }
    size_t Nrad() const { return basis_.Nrad(); }
    size_t Nang() const { return basis_.Nang(); }
    size_t Nel()  const { return basis_.rad_Nel(); }
    /// Return (ifirst, ilast) inclusive radial-function index range
    /// for element iel. Adjacent elements OVERLAP by `noverlap`
    /// (= 1 for LIP, 2 for HIP) at boundary indices.
    std::pair<size_t, size_t> radial_element_range(size_t iel) const {
      return basis_.radial_element_range(iel);
    }
    py::list lvals() const {
      py::list out;
      const Eigen::VectorXi lv = basis_.lval();
      for (Eigen::Index i = 0; i < lv.size(); ++i) out.append((int) lv(i));
      return out;
    }
    py::list mvals() const {
      py::list out;
      const Eigen::VectorXi mv = basis_.mval();
      for (Eigen::Index i = 0; i < mv.size(); ++i) out.append((int) mv(i));
      return out;
    }

    // basis_'s SCF surface is natively helfem::Matrix (Eigen), so these
    // go straight to numpy -- the Armadillo hop the migration left behind
    // here is gone.
    py::array_t<double> overlap() const { return eigen_to_numpy(basis_.overlap()); }
    py::array_t<double> kinetic() const { return eigen_to_numpy(basis_.kinetic()); }
    py::array_t<double> nuclear() const { return eigen_to_numpy(basis_.nuclear()); }
    py::array_t<double> hcore()   const {
      return eigen_to_numpy(helfem::Matrix(basis_.kinetic() + basis_.nuclear()));
    }

    /// Build (J, K) from a density matrix. K returned with HF MINUS sign
    /// already applied (so it can be added to F directly: F = h + J + K).
    /// PySCF's get_jk convention returns (J, K) both POSITIVE; the driver
    /// builds F = h + J - K. We match that: return (J, -K_helfem) i.e.
    /// flip the sign on K so PySCF can subtract it.
    std::tuple<py::array_t<double>, py::array_t<double>>
    get_jk(py::array_t<double, py::array::c_style | py::array::forcecast> P_in) {
      ensure_tei_();
      const helfem::Matrix P_E = numpy_to_eigen(P_in);
      const helfem::Matrix J = basis_.coulomb(P_E);
      // flip sign for PySCF positivity
      const helfem::Matrix K = -basis_.exchange(P_E);
      return std::make_tuple(eigen_to_numpy(J), eigen_to_numpy(K));
    }

    py::array_t<double>
    coulomb(py::array_t<double, py::array::c_style | py::array::forcecast> P_in) {
      ensure_tei_();
      return eigen_to_numpy(basis_.coulomb(numpy_to_eigen(P_in)));
    }

    py::array_t<double>
    exchange(py::array_t<double, py::array::c_style | py::array::forcecast> P_in) {
      ensure_tei_();
      // Match PySCF positive-K convention.
      return eigen_to_numpy(helfem::Matrix(-basis_.exchange(numpy_to_eigen(P_in))));
    }

    /// Density-fitted (Cholesky-factored) BARE radial Slater integrals.
    /// Returns a Python list over multipoles k = 0..2*lmax, each entry
    /// being a numpy array of shape (naux_k, Nrad, Nrad). Convention:
    ///     R^k(i, j, m, n) = sum_Q B[k][Q, i, j] * B[k][Q, m, n]
    /// where R^k is the BARE radial Slater integral (no 4*pi/(2k+1), no
    /// Gaunt -- libatomscf applies those at angular assembly).
    py::list radial_df_factors(double tol = 1e-10) {
      ensure_tei_();
      // Public API now returns std::vector<std::vector<helfem::Matrix>>
      // (outer = multipole k, inner = Cholesky factor Q, each Nrad x Nrad
      // Eigen matrix). Repack to numpy shape (naux_k, Nrad, Nrad) per k.
      auto factors = basis_.radial_df_factors(tol);
      const size_t Nrad = basis_.Nrad();
      py::list out;
      for (auto & per_L : factors) {
        const size_t naux = per_L.size();
        py::array_t<double> arr({naux, Nrad, Nrad});
        if (naux > 0) {
          auto buf = arr.mutable_unchecked<3>();
          for (size_t q = 0; q < naux; ++q) {
            const helfem::Matrix & M = per_L[q];
            for (size_t i = 0; i < Nrad; ++i) {
              for (size_t j = 0; j < Nrad; ++j) {
                buf(q, i, j) = M(i, j);
              }
            }
          }
        }
        out.append(std::move(arr));
      }
      return out;
    }

   private:
    void ensure_tei_() {
      if (!tei_done_) {
        basis_.compute_tei(/*exchange=*/true);
        tei_done_ = true;
      }
    }

    int Z_;
    int lmax_;
    int mmax_;
    bool tei_done_;
    std::shared_ptr<const helfem::polynomial_basis::PolynomialBasis> poly_;
    helfem::atomic::basis::TwoDBasis basis_;
  };

} // namespace helfem_py

PYBIND11_MODULE(_helfem, m) {
  m.doc() =
      "HelFEM Python bindings (step b of the PySCF interop arc).\n"
      "\n"
      "Provides an AtomicBasis class with the matrix builders a PySCF\n"
      "SCF driver needs: overlap, kinetic, nuclear, hcore, get_jk(P)\n"
      "(and individual coulomb/exchange). Plug into pyscf.scf by\n"
      "overriding mf.get_hcore, mf.get_ovlp, mf.get_jk on a fake-mol.";

  py::class_<helfem_py::AtomicBasis>(m, "AtomicBasis")
      .def(py::init<int, int, int, int, int, int, double, int, double, int, double>(),
           py::arg("Z"),
           py::arg("lmax"),
           py::arg("mmax"),
           py::arg("primbas") = 4,     // LIP
           py::arg("nnodes")  = 15,
           py::arg("nelem")   = 20,
           py::arg("Rmax")    = 40.0,
           py::arg("igrid")   = 4,     // exponential
           py::arg("zexp")    = 2.0,
           py::arg("finitenuc") = 0,   // POINT_NUCLEUS
           py::arg("Rrms")    = 0.0,
           "Construct an atomic FE basis for nucleus of charge Z with\n"
           "angular momenta up to (lmax, mmax).")
      .def("Z",       &helfem_py::AtomicBasis::Z)
      .def("Nbf",     &helfem_py::AtomicBasis::Nbf,
           "Total number of basis functions = Nrad * Nang.")
      .def("Nrad",    &helfem_py::AtomicBasis::Nrad)
      .def("Nang",    &helfem_py::AtomicBasis::Nang)
      .def("Nel",     &helfem_py::AtomicBasis::Nel,
           "Number of radial FE elements.")
      .def("radial_element_range",
           &helfem_py::AtomicBasis::radial_element_range,
           py::arg("iel"),
           "Inclusive (ifirst, ilast) radial-function index range for\n"
           "element iel. Adjacent elements OVERLAP at boundary indices\n"
           "(noverlap = 1 for LIP, 2 for HIP) -- callers partitioning\n"
           "per-element quantities must account for the shared boundary.")
      .def("lvals",   &helfem_py::AtomicBasis::lvals)
      .def("mvals",   &helfem_py::AtomicBasis::mvals)
      .def("overlap", &helfem_py::AtomicBasis::overlap, "Overlap matrix S.")
      .def("kinetic", &helfem_py::AtomicBasis::kinetic,
           "Kinetic energy matrix T (centrifugal l(l+1)/2 term included).")
      .def("nuclear", &helfem_py::AtomicBasis::nuclear,
           "Nuclear attraction matrix V = -Z * <i|1/r|j>.")
      .def("hcore",   &helfem_py::AtomicBasis::hcore,
           "Core Hamiltonian = T + V.")
      .def("get_jk",  &helfem_py::AtomicBasis::get_jk,
           py::arg("dm"),
           "Build (J, K) from a density matrix dm. Both returned with\n"
           "POSITIVE sign convention (PySCF style: F = h + J - K).")
      .def("coulomb", &helfem_py::AtomicBasis::coulomb, py::arg("dm"))
      .def("exchange", &helfem_py::AtomicBasis::exchange, py::arg("dm"),
           "Returns POSITIVE K (sign flipped vs HelFEM's internal -K).")
      .def("radial_df_factors", &helfem_py::AtomicBasis::radial_df_factors,
           py::arg("tol") = 1e-10,
           "Density-fitted (Cholesky) BARE radial Slater integrals.\n"
           "Returns list[k] of numpy arrays of shape (naux_k, Nrad, Nrad)\n"
           "such that R^k(i, j, m, n) = sum_Q B[k][Q,i,j] * B[k][Q,m,n].\n"
           "k runs 0..2*lmax. R^k is the BARE radial Slater integral; the\n"
           "caller (libatomscf) applies 4*pi/(2k+1) and Gaunt at angular\n"
           "assembly. tol is the residual diagonal threshold for the\n"
           "pivoted Cholesky iteration.");
}
