#ifndef UTILS_H
#define UTILS_H

#include <armadillo>

namespace helfem {
  namespace utils {
    /// inverse cosh
    double arcosh(double x);
    /// inverse sinh
    double arsinh(double x);

    /// Form two-electron integrals from product of large-r and small-r radial moment matrices
    arma::mat product_tei(const arma::mat & big, const arma::mat & small);

    /// Check that the two-electron integral has proper symmetry i<->j and k<->l
    void check_tei_symmetry(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);

    /// Permute indices (ij|kl) -> (jk|il)
    arma::mat exchange_tei(const arma::mat & tei, size_t Ni, size_t Nj, size_t Nk, size_t Nl);
  }
}

#endif
