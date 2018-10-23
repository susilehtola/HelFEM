#ifndef LCAO_H
#define LCAO_H

namespace helfem {
  namespace lcao {
    /// Evaluate radial GTO
    double radial_GTO(double r, int l, double alpha);
    /// Evaluate radial STO
    double radial_STO(double r, int l, double zeta);
  }
}

#endif
