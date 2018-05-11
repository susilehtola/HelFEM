#ifndef BASIS_H
#define BASIS_H

#include <armadillo>

namespace helfem {
  namespace basis {
    /// Radial basis set
    class RadialBasis {
      /// Quadrature points
      arma::vec xq;
      /// Quadrature weights
      arma::vec wq;

      /// Primitive polynomial basis
      arma::mat bf;
      /// .. and its derivatives
      arma::mat df;

      /// Number of overlapping functions (= der_order+1)
      int noverlap;
      /// Element boundary values
      arma::vec bval;

    public:
      /// Dummy constructor
      RadialBasis();
      /// Construct radial basis
      RadialBasis(int n_nodes, int der_order, int n_quad, int num_el, double rmax);
      /// Destructor
      ~RadialBasis();

      /// Number of basis functions
      size_t Nbf() const;
      /// Number of primitive functions in element
      size_t Nprim(size_t iel) const;

      /// Number of elements
      size_t Nel() const;
      /// Get function indices
      void get_idx(size_t iel, size_t & ifirst, size_t & ilast) const;

      /// Compute radial matrix elements <r^n> in element (overlap is n=2, nuclear is n=1)
      arma::mat radial_integral(int n, size_t iel) const;
      /// Compute overlap matrix in element
      arma::mat overlap(size_t iel) const;
      /// Compute primitive kinetic energy matrix in element (excluding l part)
      arma::mat kinetic(size_t iel) const;
      /// Compute l part of kinetic energy matrix
      arma::mat kinetic_l(size_t iel) const;
      /// Compute nuclear attraction matrix in element
      arma::mat nuclear(size_t iel) const;
      /// Compute primitive two-electron integral
      arma::mat twoe_integral(int L, size_t iel) const;
    };

    /// Two-dimensional basis set
    class TwoDBasis {
      /// Nuclear charge
      int Z;

      /// Radial basis set
      RadialBasis radial;
      /// Angular basis set: function l values
      arma::ivec lval;
      /// Angular basis set: function m values
      arma::ivec mval;

      /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
      std::vector<arma::mat> prim_tei;

    public:
      /// Constructor
      TwoDBasis(int Z, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax);
      /// Destructor
      ~TwoDBasis();

      /// Compute two-electron integrals
      void compute_tei();

      /// Number of basis functions
      size_t Nbf() const;

      /// Form half-inverse overlap matrix
      arma::mat Sinvh() const;
      /// Form overlap matrix
      arma::mat overlap() const;
      /// Form kinetic energy matrix
      arma::mat kinetic() const;
      /// Form nuclear attraction matrix
      arma::mat nuclear() const;

      /// Form Coulomb matrix
      arma::mat coulomb(const arma::mat & P) const;
      /// Form exchange matrix
      arma::mat exchange(const arma::mat & P) const;
    };
  }
}


#endif
