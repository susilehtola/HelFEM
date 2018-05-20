#include "legendre.h"
#include "legendre_pq.h"

namespace helfem {
  namespace legendre {
    arma::vec legendreP(int l, int m, const arma::vec & x) {
      if(l>::legendre::legendrePQ_max_l())
        throw std::logic_error("l value outside range!\n");
      if(m<0)
        throw std::logic_error("m value outside range!\n");
      if(m>::legendre::legendrePQ_max_m())
        throw std::logic_error("m value outside range!\n");
  
      arma::vec r(x.n_elem);
      for(size_t i=0;i<x.n_elem;i++)
        r(i)=::legendre::legendreP(l,m,x(i));

      return r;
    }

    arma::vec legendreQ(int l, int m, const arma::vec & x) {
      if(l>::legendre::legendrePQ_max_l())
        throw std::logic_error("l value outside range!\n");
      if(m<0)
        throw std::logic_error("m value outside range!\n");
      if(m>::legendre::legendrePQ_max_m())
        throw std::logic_error("m value outside range!\n");
  
      arma::vec r(x.n_elem);
      for(size_t i=0;i<x.n_elem;i++)
        r(i)=::legendre::legendreQ(l,m,x(i));

      return r;
    }

    arma::vec legendreP_prolate(int l, int m, const arma::vec & x) {
      if(l>::legendre::legendrePQ_max_l())
        throw std::logic_error("l value outside range!\n");
      if(m<0)
        throw std::logic_error("m value outside range!\n");
      if(m>::legendre::legendrePQ_max_m())
        throw std::logic_error("m value outside range!\n");
  
      arma::vec r(x.n_elem);
      for(size_t i=0;i<x.n_elem;i++)
        r(i)=::legendre::legendreP_prolate(l,m,x(i));

      return r;
    }

    arma::vec legendreQ_prolate(int l, int m, const arma::vec & x) {
      if(l>::legendre::legendrePQ_max_l())
        throw std::logic_error("l value outside range!\n");
      if(m<0)
        throw std::logic_error("m value outside range!\n");
      if(m>::legendre::legendrePQ_max_m())
        throw std::logic_error("m value outside range!\n");
  
      arma::vec r(x.n_elem);
      for(size_t i=0;i<x.n_elem;i++)
        r(i)=::legendre::legendreQ_prolate(l,m,x(i));

      return r;
    }
  }
}


