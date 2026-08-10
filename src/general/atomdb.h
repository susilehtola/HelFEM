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
#ifndef ATOMDB_H
#define ATOMDB_H

// Only libhelfem headers: model_potential.h includes this one, and it is
// itself pulled in early by src/atomic/basis.h.
#include <RadialBasis.h>
#include <PolynomialBasis.h>
#include <memory>
#include <vector>

namespace helfem {
  /// Tabulated spherically averaged atomic wave functions for H..Og.
  ///
  /// The database ships the *orbitals*, not the potential they generate.
  /// A finite-element representation of the potential would need a finer
  /// grid than the one the orbitals were solved on -- the density is
  /// twice the polynomial degree of the orbitals and the potential higher
  /// still -- so everything derived (density, Hartree potential, SAP
  /// effective charge) is instead evaluated on the fly from the stored
  /// expansion. That is exact at every r, with no tabulation knots and no
  /// interpolation, which is what the tabulated-Zeff SAP guess in sap.cpp
  /// could not offer.
  namespace atomdb {
    /// The raw table, emitted by tools/gen_atomdb.py.
    namespace data {
      extern const int max_Z, lmax, Nbf, nelem, nnodes, primbas, nquad, norbital;
      extern const double bval[];
      extern const int norb[][4];
      extern const int offset[][4];
      extern const double occupations[];
      extern const double coefficients[];
    }

    /// Highest nuclear charge in the database.
    int max_Z();
    /// Highest angular momentum stored.
    int lmax();
    /// Number of radial basis functions the coefficients refer to.
    int Nbf();
    /// Element boundaries of the shared radial grid (nelem+1 entries).
    helfem::Vector element_boundaries();

    /// Number of stored orbitals for (Z, l). Orbitals whose occupation
    /// fell below the storage threshold are absent, so this is smaller
    /// than the number of orbitals the calculation produced.
    int norb(int Z, int l);
    /// Occupations of the stored (Z, l) orbitals. Generally fractional:
    /// the database was generated without pinning occupations.
    helfem::Vector occupations(int Z, int l);
    /// Expansion coefficients of the stored (Z, l) orbitals, Nbf x norb.
    helfem::Matrix coefficients(int Z, int l);

    /// Initialized libxc handles, cached across evaluations. Opaque here
    /// so that atomdb.h -- and through it model_potential.h, an installed
    /// public header -- does not drag libxc into every consumer.
    struct XCFunctionals;

    /// A tabulated atom, ready to be evaluated at arbitrary r.
    ///
    /// Construction rebuilds the shared finite-element basis and
    /// contracts the orbitals into per-l density matrices; the per-element
    /// charge moments are accumulated once, so a subsequent evaluation
    /// costs one partial-element quadrature rather than a sweep over the
    /// whole grid.
    class Atom {
      /// Nuclear charge
      int Z_;
      /// The radial basis, shared by every record in the database and
      /// therefore built once rather than per atom.
      const atomic::basis::FEMRadialBasis & radial_;
      /// Total density matrix, restricted to each element's own functions
      std::vector<helfem::Matrix> Psub_;
      /// Charge contained in the elements strictly below iel
      helfem::Vector Qbelow_;
      /// Outer moment integral(rho/r) of the elements strictly above iel
      helfem::Vector Mabove_;
      /// Total charge carried by the stored orbitals
      double Ntot_;
      /// Practical infinity, i.e. the end of the last element
      double Rmax_;
      /// libxc handles, initialized on first use for whichever
      /// functionals are asked for and reused afterwards.
      mutable std::shared_ptr<XCFunctionals> xc_;

    public:
      /// Integral over the part of element iel between the reference
      /// coordinates xa and xb, of the radial charge distribution
      /// 4 pi r^2 rho(r) (over_r = false) or of it divided by r
      /// (over_r = true). These are the two halves of the multipole
      /// expansion of 1/r_>, restricted to the element the evaluation
      /// point falls in; everything outside is already summed into
      /// Qbelow_ and Mabove_.
      double partial_integral(size_t iel, double xa, double xb, bool over_r) const;

    public:
      /// Construct the record for nuclear charge Z (1 <= Z <= max_Z()).
      explicit Atom(int Z);

      /// Nuclear charge
      int charge() const;
      /// Number of electrons carried by the stored orbitals. Slightly
      /// below Z, by the occupation left in the discarded orbitals.
      double nelectrons() const;
      /// The underlying radial basis, for callers that want to project.
      const atomic::basis::FEMRadialBasis & basis() const;
      /// Element boundaries: the radii at which the derived quantities
      /// are only piecewise smooth. Feed these to the quadrature.
      helfem::Vector element_boundaries() const;

      /// Radial functions R_nl(r) of the stored orbitals of this l, one
      /// entry per stored orbital, evaluated at radius r.
      ///
      /// The database stores the ORBITALS rather than the density (see
      /// [[sap-ship-wave-functions-not-potential]]: the density is twice
      /// their polynomial degree and the potential worse still), so this
      /// is the primitive a projected guess needs -- everything else here
      /// is derived from it. The identity that fixes the convention is
      ///
      ///     rho(r) = sum_l sum_n occupations(Z,l)(n) * R_nl(r)^2 / (4 pi)
      ///
      /// which atomdb_test checks against density() rather than leaving
      /// the normalization to be rediscovered by each caller.
      helfem::Vector orbitals(int l, double r) const;

      /// Electron density rho(r).
      double density(double r) const;
      /// Radial derivative of the density, d rho / dr. The density is
      /// spherically symmetric, so this is also the magnitude of its
      /// gradient.
      double density_gradient(double r) const;
      /// Laplacian of the density, rho'' + 2 rho' / r.
      double density_laplacian(double r) const;
      /// Radial charge distribution 4 pi r^2 rho(r).
      double radial_density(double r) const;
      /// Charge enclosed by the sphere of radius r.
      double enclosed_charge(double r) const;
      /// Hartree screening r * V_Hartree(r), i.e. the amount of nuclear
      /// charge the electrons screen at r. Tends to nelectrons() as
      /// r -> infinity and to zero at the nucleus.
      double hartree_screening(double r) const;
      /// Exchange-correlation screening r * v_xc(r) for the given libxc
      /// functional ids (0 = none). LDAs and GGAs; for a GGA the screening
      /// is v_rho - div(2 v_sigma grad rho), which the spherical symmetry
      /// reduces to a radial derivative plus a 2/r term. Meta-GGAs are
      /// rejected, as in the grid implementation.
      double xc_screening(double r, int x_func, int c_func) const;
      /// SAP effective charge Z - r * (V_Hartree + v_xc), i.e. the
      /// nuclear charge an electron at r actually sees.
      double effective_charge(double r, int x_func, int c_func) const;
    };
  }
}

#endif
