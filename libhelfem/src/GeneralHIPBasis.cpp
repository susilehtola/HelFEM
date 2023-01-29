#include "GeneralHIPBasis.h"
#include "lobatto.h"

namespace helfem {
  namespace polynomial_basis {

    static void print_test(arma::mat f, const std::string & msg, double thr=1e-9) {
      // Set small elements to zero
      f.elem(arma::find(arma::abs(f)<thr)).zeros();
      // Print out with message
      f.print(msg);
    }

    // Dummy length
    static const double dummy_length = 1.0;

    GeneralHIPBasis::GeneralHIPBasis(const arma::vec & x, int id_, int nder_) {
      id=id_;
      nder=nder_;

      // Number of overlapping functions
      noverlap=nder+1;
      // The number of functions we need is
      int nfuncs=(nder+1)*x.n_elem;
      nprim=nfuncs;
      // All functions are enabled
      enabled=arma::linspace<arma::uvec>(0,nfuncs-1,nfuncs);
      // Number of nodes is
      nnodes=x.n_elem;

      // Construct the necessary LIP basis
      arma::vec xlip, wlip;
      ::lobatto_compute(nfuncs,xlip,wlip);
      lip = polynomial_basis::LIPBasis(xlip);

      printf("Setting up %i-node %i:th order HIPs from a %i-node LIP basis.\n", nnodes, nder, nfuncs);

      // Evaluate the values of the LIPs and their derivatives
      std::vector<arma::mat> dfval(nder+1);
      for(int ider=0;ider<=nder;ider++)
        dfval[ider] = lip.eval_dnf(x, ider, dummy_length).t();

      /*
       * Construct the equation for the transformation matrix.
       *
       *           T X = 1
       *
       * where T is the transformation matrix and X are the values of
       * LIPs and their derivatives at the nodes, so T = X^-1.
       *
       * We can easily build the matrix in our target basis by looping
       * over the nodes. The first nder+1 functions are the value of
       * the LIP and its nder derivaties at the first node. The next
       * nder+1 functions are the values at the second node. Etc.
       */
      arma::mat X(nfuncs, nfuncs, arma::fill::zeros);
      for(int inode=0;inode < nnodes;inode++) {
        for(int ider=0;ider<=nder;ider++)
          X.col((nder+1)*inode+ider) = dfval[ider].col(inode);
      }

      // X has our target functions in its columns so X^-1 has the
      // target in its rows; if we take the transpose then we get the
      // target functions in columns in T
      printf("Transformation matrix reciprocal condition number %e\n",arma::rcond(X));
      T = arma::inv(X.t());

      // Test the conditions
      for(int ider=0;ider<=nder;ider++) {
        std::ostringstream oss;
        oss << ider << "th derivative value at nodes";
        print_test(eval_dnf(x, ider, dummy_length),  oss.str());
      }
    }

    GeneralHIPBasis::~GeneralHIPBasis() {
    }

    GeneralHIPBasis * GeneralHIPBasis::copy() const {
      return new GeneralHIPBasis(*this);
    }

    void GeneralHIPBasis::scale_derivatives(arma::mat & f, double element_length) const {
      for(int inode=0; inode<nnodes; inode++) {
        for(int ider=1;ider<=nder;ider++)
          f.col((nder+1)*inode+ider) *= std::pow(element_length,ider);
      }
    }

    void GeneralHIPBasis::eval_prim_dnf(const arma::vec & x, arma::mat & dnf, int n, double element_length) const {
      // Evaluate the primitive LIP polynomials
      lip.eval_prim_dnf(x, dnf, n, dummy_length);
      dnf=dnf*T;
      // and scale the derivative functions
      scale_derivatives(dnf, element_length);
    }

    void GeneralHIPBasis::drop_first(bool func, bool deriv) {
      // Subset of functions in the first
      int nfuncs=nder+1;
      arma::uvec same_funcs(enabled.subvec(nfuncs,enabled.n_elem-1));
      arma::uvec first_funcs(enabled.subvec(0, nfuncs-1));

      // Number of new functions
      size_t nnew = same_funcs.n_elem;
      if(!func) nnew++;
      if(!deriv) nnew+=nder;
      arma::uvec new_funcs(nnew);

      int idx = 0;
      if(!func)
        new_funcs(idx++) = first_funcs(0);
      if(!deriv)
        for(int ider=0;ider<nder;ider++)
          new_funcs(idx++) = first_funcs(1+ider);
      new_funcs.subvec(idx,new_funcs.n_elem-1) = same_funcs;

      enabled = new_funcs;
    }

    void GeneralHIPBasis::drop_last(bool func, bool deriv) {
      // Subset of functions in the last element
      int nfuncs=nder+1;
      arma::uvec same_funcs(enabled.subvec(0,enabled.n_elem-nfuncs-1));
      arma::uvec last_funcs(enabled.subvec(enabled.n_elem-nfuncs, enabled.n_elem-1));

      // Number of new functions
      size_t nnew = same_funcs.n_elem;
      if(!func) nnew++;
      if(!deriv) nnew+=nder;
      arma::uvec new_funcs(nnew);
      new_funcs.subvec(0,same_funcs.n_elem-1) = same_funcs;

      int idx = same_funcs.n_elem;
      if(!func)
        new_funcs(idx++) = last_funcs(0);
      if(!deriv)
        for(int ider=0;ider<nder;ider++)
          new_funcs(idx++) = last_funcs(1+ider);

      enabled = new_funcs;
    }
  }
}
