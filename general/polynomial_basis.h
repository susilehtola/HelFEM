#ifndef POLYNOMIAL_BASIS_H
#define POLYNOMIAL_BASIS_H

#include <armadillo>

namespace helfem {
  namespace polynomial_basis {
    /// Template for a primitive basis
    class PolynomialBasis {
    protected:
      /// Number of basis functions
      int nbf;
      /// Number of overlapping functions
      int noverlap;
    public:
      /// Constructor
      PolynomialBasis();
      /// Destructor
      virtual ~PolynomialBasis();
      /// Get a copy
      virtual PolynomialBasis * copy() const=0;

      /// Get number of basis functions
      int get_nbf() const;
      /// Get number of overlapping functions
      int get_noverlap() const;

      /// Drop first function
      virtual void drop_first()=0;
      /// Drop last function
      virtual void drop_last()=0;

      /// Evaluate polynomials at given points
      virtual arma::mat eval(const arma::vec & x) const=0;
      /// Evaluate polynomials and derivatives at given points
      virtual void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const=0;
    };

    /// Get the wanted basis
    PolynomialBasis * get_basis(int primbas, int Nnodes);

    /// Primitive polynomials with Hermite
    class HermiteBasis: public PolynomialBasis {
      /// Primitive polynomial basis expansion
      arma::mat bf_C;
      /// Primitive polynomial basis expansion, derivative
      arma::mat df_C;
    public:
      /// Constructor
      HermiteBasis(int n_nodes, int der_order);
      /// Destructor
      ~HermiteBasis();
      /// Get a copy
      HermiteBasis * copy() const;

      /// Drop first function
      void drop_first();
      /// Drop last function
      void drop_last();

      /// Evaluate polynomials at given points
      arma::mat eval(const arma::vec & x) const;
      /// Evaluate polynomials and derivatives at given points
      void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const;
    };

    /// Legendre functions
    class LegendreBasis: public PolynomialBasis {
      /// Maximum order
      int lmax;
      /// Transformation matrix
      arma::mat T;
    public:
      /// Constructor
      LegendreBasis(int lmax);
      /// Destructor
      ~LegendreBasis();
      /// Get a copy
      LegendreBasis * copy() const;

      /// Drop first function
      void drop_first();
      /// Drop last function
      void drop_last();

      /// Evaluate polynomials at given points
      arma::mat eval(const arma::vec & x) const;
      /// Evaluate polynomials and derivatives at given points
      void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const;
    };

    /// Lagrange interpolating polynomials
    class LIPBasis: public PolynomialBasis {
      /// Control nodes
      arma::vec x0;
      /// Indices of enabled functions
      arma::uvec enabled;
    public:
      /// Constructor
      LIPBasis(const arma::vec & x0);
      /// Destructor
      ~LIPBasis();
      /// Get a copy
      LIPBasis * copy() const;

      /// Drop first function
      void drop_first();
      /// Drop last function
      void drop_last();

      /// Evaluate polynomials at given points
      arma::mat eval(const arma::vec & x) const;
      /// Evaluate polynomials and derivatives at given points
      void eval(const arma::vec & x, arma::mat & f, arma::mat & df) const;
    };
  }
}
#endif
