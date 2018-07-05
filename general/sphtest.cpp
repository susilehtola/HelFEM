#include "chebyshev.h"
#include "spherical_harmonics.h"

int main(void) {
  int lsph=5;
  for(int lang=2*lsph;lang<=15*lsph;lang+=lsph) {

    // Get quadrature rule
    arma::vec cth, phi, w;
    helfem::chebyshev::angular_chebyshev(lang,cth,phi,w);

    // Evaluate spherical harmonics
    int nsph=(lsph+1)*(lsph+1);
    arma::cx_mat sph(nsph,cth.n_elem);
    for(size_t iang=0;iang<cth.n_elem;iang++) {
      size_t ioff=0;
      for(int l=0;l<=lsph;l++)
        for(int m=-l;m<=l;m++)
          sph(ioff++,iang)=spherical_harmonics(l,m,cth(iang),phi(iang));
      if(ioff!=nsph)
        throw std::logic_error("Count wrong!\n");
    }

    // Calculate quadrature
    arma::mat ovl(arma::abs(sph*arma::diagmat(w)*arma::trans(sph)));
    //ovl.print("Overlap");
    ovl-=arma::eye<arma::mat>(ovl.n_rows,ovl.n_cols);
    printf("Difference from orthonormality with lang=%i is %e max %e\n",lang,arma::norm(ovl,"fro"),arma::max(ovl));
  }
  
  return 0;
}

  
