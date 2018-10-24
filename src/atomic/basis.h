#ifndef BASIS_H
#define BASIS_H

#include <armadillo>
#include "../general/polynomial_basis.h"

namespace helfem {
  namespace atomic {
    namespace basis {
      /// Radial basis set
      class RadialBasis {
        /// Quadrature points
        arma::vec xq;
        /// Quadrature weights
        arma::vec wq;

        /// Polynomial basis
        const polynomial_basis::PolynomialBasis * poly;
        /// Primitive polynomial basis functions evaluated on the quadrature grid
        arma::mat bf;
        /// Primitive polynomial basis function derivatives evaluated on the quadrature grid
        arma::mat df;

        /// Element boundary values
        arma::vec bval;

	/// Used basis function indices in element
	arma::uvec basis_indices(size_t iel) const;
        /// Get basis functions in element
        arma::mat get_basis(const arma::mat & b, size_t iel) const;
        /// Get basis functions in element
        polynomial_basis::PolynomialBasis * get_basis(const polynomial_basis::PolynomialBasis * poly, size_t iel) const;

      public:
        /// Dummy constructor
        RadialBasis();
        /// Construct radial basis
        RadialBasis(const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el, double rmax, int igrid, double zexp);
        /// Construct radial basis
        RadialBasis(const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el0, int Zm, int Zlr, double Rhalf, int num_el, double rmax, int igrid, double zexp);
        /// Destructor
        ~RadialBasis();

        /// Get number of overlapping functions
        size_t get_noverlap() const;
        /// Number of basis functions
        size_t Nbf() const;
        /// Number of primitive functions in element
        size_t Nprim(size_t iel) const;
        /// Number of primitive functions in element
        size_t max_Nprim() const;

        /// Number of elements
        size_t Nel() const;
        /// Get function indices
        void get_idx(size_t iel, size_t & ifirst, size_t & ilast) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & Cl, const arma::mat & Cr, size_t nocc) const;

        /// Compute radial matrix elements <r^n> in element (overlap is n=0, nuclear is n=-1)
        arma::mat radial_integral(const arma::mat & bf_c, int n, size_t iel) const;
        /// Compute radial matrix elements <r^n> in element (overlap is n=0, nuclear is n=-1)
        arma::mat radial_integral(int n, size_t iel) const;
        /// Compute overlap matrix in element
        arma::mat overlap(size_t iel) const;
        /// Compute primitive kinetic energy matrix in element (excluding l part)
        arma::mat kinetic(size_t iel) const;
        /// Compute l part of kinetic energy matrix
        arma::mat kinetic_l(size_t iel) const;
        /// Compute nuclear attraction matrix in element
        arma::mat nuclear(size_t iel) const;
        /// Compute off-center nuclear attraction matrix in element
        arma::mat nuclear_offcenter(size_t iel, double Rhalf, int L) const;

        /// Compute primitive two-electron integral
        arma::mat twoe_integral(int L, size_t iel) const;

        /// Evaluate basis functions at quadrature points
        arma::mat get_bf(size_t iel) const;
        /// Evaluate derivatives of basis functions at quadrature points
        arma::mat get_df(size_t iel) const;
        /// Get quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;

        /// Evaluate nuclear density
        double nuclear_density(const arma::mat & P) const;
      };

      /// Two-dimensional basis set
      class TwoDBasis {
        /// Nuclear charge
        int Z;

        /// Left-hand nuclear charge
        int Zl;
        /// Right-hand nuclear charge
        int Zr;
        /// Bond length
        double Rhalf;

        /// Radial basis set
        RadialBasis radial;
        /// Angular basis set: function l values
        arma::ivec lval;
        /// Angular basis set: function m values
        arma::ivec mval;

        /// Auxiliary integrals
        std::vector<arma::mat> disjoint_L, disjoint_m1L;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)>
        std::vector<arma::mat> prim_tei;
        /// Primitive two-electron integrals: <Nel^2 * (2L+1)> sorted for exchange
        std::vector<arma::mat> prim_ktei;

        /// Get indices of real basis functions
        arma::uvec pure_indices() const;

        /// Add to radial submatrix
        void add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Set radial submatrix
        void set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Get radial submatrix
        arma::mat get_sub(const arma::mat & M, size_t iang, size_t jang) const;

      public:
        TwoDBasis();
        /// Constructor
        TwoDBasis(int Z, const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp);
        TwoDBasis(int Z, const polynomial_basis::PolynomialBasis * poly, int n_quad, int num_el0, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp, int Zl, int Zr, double Rhalf);
        /// Destructor
        ~TwoDBasis();

        /// Expand boundary conditions
        arma::mat expand_boundaries(const arma::mat & H) const;
        /// Remove boundary conditions
        arma::mat remove_boundaries(const arma::mat & H) const;

        /// Memory for one-electron integral matrix
        size_t mem_1el() const;
        /// Memory for auxiliary one-electron integrals (off-center nuclei)
        size_t mem_1el_aux() const;
        /// Memory for auxiliary two-electron integrals
        size_t mem_2el_aux() const;

        /// Compute two-electron integrals
        void compute_tei(bool exchange);

        /// Number of basis functions
        size_t Nbf() const;
        /// Number of dummy basis functions
        size_t Ndummy() const;

        /// Number of radial functions
        size_t Nrad() const;
        /// Number of angular shells
        size_t Nang() const;

        /// Form half-overlap matrix
        arma::mat Shalf(bool chol, int sym) const;
        /// Form half-inverse overlap matrix
        arma::mat Sinvh(bool chol, int sym) const;
        /// Form radial integral
        arma::mat radial_integral(int n) const;
        /// Form overlap matrix
        arma::mat overlap() const;
        /// Form kinetic energy matrix
        arma::mat kinetic() const;
        /// Form nuclear attraction matrix
        arma::mat nuclear() const;
        /// Form dipole coupling matrix
        arma::mat dipole_z() const;
        /// Form quadrupole coupling matrix
        arma::mat quadrupole_zz() const;

        /// Coupling to magnetic field in z direction
        arma::mat Bz_field(double B) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & C, size_t nocc) const;

        /// Form Coulomb matrix
        arma::mat coulomb(const arma::mat & P) const;
        /// Form exchange matrix
        arma::mat exchange(const arma::mat & P) const;

        /// Get primitive integrals
        std::vector<arma::mat> get_prim_tei() const;

        /// Get l values
        arma::ivec get_l() const;
        /// Get m values
        arma::ivec get_m() const;
        /// Get indices of basis functions with wanted m quantum number
        arma::uvec m_indices(int m) const;
        /// Get indices of basis functions with wanted l and m quantum numbers
        arma::uvec lm_indices(int l, int m) const;
        /// Get indices for wanted symmetry
        std::vector<arma::uvec> get_sym_idx(int isym) const;

        /// Evaluate basis functions
        arma::cx_mat eval_bf(size_t iel, double cth, double phi) const;
        /// Evaluate basis functions derivatives
        void eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const;
        /// Get list of basis function indices in element
        arma::uvec bf_list(size_t iel) const;

        /// Get number of radial elements
        size_t get_rad_Nel() const;
        /// Get radial quadrature weights
        arma::vec get_wrad(size_t iel) const;
        /// Get r values
        arma::vec get_r(size_t iel) const;

        /// Electron density at nuclei
        arma::vec nuclear_density(const arma::mat & P) const;
      };
    }
  }
}


#endif
