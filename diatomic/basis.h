#ifndef BASIS_H
#define BASIS_H

#include <armadillo>

namespace helfem {
  namespace diatomic {
    namespace basis {
      /// Radial basis set
      class RadialBasis {
        /// Quadrature points
        arma::vec xq;
        /// Quadrature weights
        arma::vec wq;

        /// Primitive polynomial basis, polynomial expansion
        arma::mat bf_C;
        /// Primitive polynomial basis
        arma::mat bf;
        /// .. and its derivatives, polynomial expansion
        arma::mat df_C;
        /// evaluated on the quadrature grid
        arma::mat df;

        /// Number of overlapping functions (= der_order+1)
        size_t noverlap;
        /// Element boundary values
        arma::vec bval;

        /// Get basis functions in element
        arma::mat get_basis(const arma::mat & b, size_t iel) const;

      public:
        /// Dummy constructor
        RadialBasis();
        /// Construct radial basis
        RadialBasis(int n_nodes, int der_order, int n_quad, int num_el, double rmax, int igrid, double zexp);
        /// Destructor
        ~RadialBasis();

        /// Get number of overlapping functions
        size_t get_noverlap() const;
        /// Number of basis functions
        size_t Nbf() const;
        /// Number of primitive functions in element
        size_t Nprim(size_t iel) const;

        /// Number of elements
        size_t Nel() const;
        /// Get function indices
        void get_idx(size_t iel, size_t & ifirst, size_t & ilast) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & Cl, const arma::mat & Cr, size_t nocc) const;

        /// Compute radial matrix elements \f$ B_1(\mu) B_2(\mu) \sinh^m (\mu) \cosh^m (\mu) d\mu \f$
        arma::mat radial_integral(const arma::mat & bf, int m, int n, size_t iel) const;
        /// Compute radial matrix elements in element
        arma::mat radial_integral(int m, int n, size_t iel) const;
        /// Compute radial matrix elements
        arma::mat radial_integral(int m, int n) const;
        /// Compute primitive kinetic energy matrix in element (excluding l and m parts)
        arma::mat kinetic(size_t iel) const;
        /// Compute primitive kinetic energy matrix
        arma::mat kinetic() const;

        /// Compute Plm integral
        arma::mat Plm_integral(int beta, size_t iel, int L, int M) const;
        /// Compute Qlm integral
        arma::mat Qlm_integral(int alpha, size_t iel, int L, int M) const;
        /// Compute primitive two-electron integral
        arma::mat twoe_integral(int alpha, int beta, size_t iel, int L, int M) const;
      };

      /// L, |M| index type
      typedef std::pair<int, int> lmidx_t;
      /// Sort operator
      bool operator<(const lmidx_t & lh, const lmidx_t & rh);
      /// Equivalence operator
      bool operator==(const lmidx_t & lh, const lmidx_t & rh);

      /// Two-dimensional basis set
      class TwoDBasis {
        /// Nuclear charges
        int Z1, Z2;
        /// Half-bond distance
        double Rhalf;

        /// Radial basis set
        RadialBasis radial;
        /// Angular basis set: function l values
        arma::ivec lval;
        /// Angular basis set: function m values
        arma::ivec mval;

        /// L, |M| map
        std::vector<lmidx_t> lm_map;
        /// Primitive two-electron integrals: <Nel^2 * N_L>
        std::vector<arma::mat> prim_tei00, prim_tei02, prim_tei20, prim_tei22;
        /// Primitive two-electron integrals: <Nel^2 * N_L> sorted for exchange
        std::vector<arma::mat> prim_ktei00, prim_ktei02, prim_ktei20, prim_ktei22;

        /// Number of basis functions in angular block
        size_t angular_nbf(size_t amind) const;
        /// Offset for angular block
        size_t angular_offset(size_t amind) const;

        /// Add to radial submatrix
        void add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Set radial submatrix
        void set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Msub) const;
        /// Get radial submatrix
        arma::mat get_sub(const arma::mat & M, size_t iang, size_t jang) const;

        /// Expand boundary conditions
        arma::mat expand_boundaries(const arma::mat & H) const;
        /// Remove boundary conditions
        arma::mat remove_boundaries(const arma::mat & C) const;

        /// Find index in (L,|M|) table
        size_t lmind(int L, int M, bool check=true) const;
        /// Get L_max
        int L_max() const;

        /// Number of dummy basis functions
        size_t Ndummy() const;

      public:
        /// Constructor
        TwoDBasis(int Z1, int Z2, double Rbond, int n_nodes, int der_order, int n_quad, int num_el, double rmax, int lmax, int mmax, int igrid, double zexp);
        /// Destructor
        ~TwoDBasis();

        /// Compute two-electron integrals
        void compute_tei();

        /// Number of basis functions
        size_t Nbf() const;

        /// Form half-inverse overlap matrix
        arma::mat Sinvh() const;
        /// Form radial integral
        arma::mat radial_integral(int n) const;
        /// Form overlap matrix
        arma::mat overlap() const;
        /// Form kinetic energy matrix
        arma::mat kinetic() const;
        /// Form nuclear attraction matrix
        arma::mat nuclear() const;
        /// Form electric field coupling matrix
        arma::mat electric(double Ez) const;

        /// Form density matrix
        arma::mat form_density(const arma::mat & C, size_t nocc) const;

        /// Form Coulomb matrix
        arma::mat coulomb(const arma::mat & P) const;
        /// Form exchange matrix
        arma::mat exchange(const arma::mat & P) const;

        /// Get primitive integrals
        std::vector<arma::mat> get_prim_tei() const;

        /// Set elements to zero
        void set_zero(int lmax, arma::mat & M) const;
      };
    }
  }
}

#endif
