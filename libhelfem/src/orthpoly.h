// LIC// ====================================================================
// LIC// This file forms part of oomph-lib, the object-oriented,
// LIC// multi-physics finite-element library, available
// LIC// at http://www.oomph-lib.org.
// LIC//
// LIC//    Version 1.0; svn revision $LastChangedRevision: 1097 $
// LIC//
// LIC// $LastChangedDate: 2015-12-17 13:53:17 +0200 (Thu, 17 Dec 2015) $
// LIC//
// LIC// Copyright (C) 2006-2016 Matthias Heil and Andrew Hazel
// LIC//
// LIC// This library is free software; you can redistribute it and/or
// LIC// modify it under the terms of the GNU Lesser General Public
// LIC// License as published by the Free Software Foundation; either
// LIC// version 2.1 of the License, or (at your option) any later version.
// LIC//
// LIC// This library is distributed in the hope that it will be useful,
// LIC// but WITHOUT ANY WARRANTY; without even the implied warranty of
// LIC// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// LIC// Lesser General Public License for more details.
// LIC//
// LIC// You should have received a copy of the GNU Lesser General Public
// LIC// License along with this library; if not, write to the Free Software
// LIC// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// LIC// 02110-1301  USA.
// LIC//
// LIC// The authors may be contacted at oomph-lib@maths.man.ac.uk.
// LIC//
// LIC//====================================================================
// Header functions for functions used to generate orthogonal polynomials
// Include guards to prevent multiple inclusion of the header

#ifndef OOMPH_ORTHPOLY_HEADER
#define OOMPH_ORTHPOLY_HEADER

#include <cmath>

namespace oomph {
  // Let's put these things in a namespace
  namespace Orthpoly {
    ///\short Calculates Legendre polynomial of degree p at x
    /// using the three term recurrence relation
    /// \f$ (n+1) P_{n+1} = (2n+1)xP_{n} - nP_{n-1} \f$
    inline double legendre(const unsigned &p, const double &x) {
      // Return the constant value
      if (p == 0)
        return 1.0;
      // Return the linear polynomial
      else if (p == 1)
        return x;
      // Otherwise use the recurrence relation
      else {
        // Initialise the terms in the recurrence relation, we're going
        // to shift down before using the relation.
        double L0 = 1.0, L1 = 1.0, L2 = x;
        // Loop over the remaining polynomials
        for (unsigned n = 1; n < p; n++) {
          // Shift down the values
          L0 = L1;
          L1 = L2;
          // Use the three term recurrence
          L2 = ((2 * n + 1) * x * L1 - n * L0) / (n + 1);
        }
        // Once we've finished return the final value
        return L2;
      }
    }

    /// \short  Calculates first derivative of Legendre
    /// polynomial of degree p at x
    /// using three term recursive formula.
    /// \f$ nP_{n+1}^{'} = (2n+1)xP_{n}^{'} - (n+1)P_{n-1}^{'} \f$
    inline double dlegendre(const unsigned &p, const double &x) {
      double dL1 = 1.0, dL2 = 3 * x, dL3 = 0.0;
      if (p == 0)
        return 0.0;
      else if (p == 1)
        return dL1;
      else if (p == 2)
        return dL2;
      else {
        for (unsigned n = 2; n < p; n++) {
          dL3 = 1.0 / n * ((2.0 * n + 1.0) * x * dL2 - (n + 1.0) * dL1);
          dL1 = dL2;
          dL2 = dL3;
        }
        return dL3;
      }
    }

    /// \short Calculates second derivative of Legendre
    /// polynomial of degree p at x
    /// using three term recursive formula.
    inline double ddlegendre(const unsigned &p, const double &x) {
      double ddL2 = 3.0, ddL3 = 15 * x, ddL4 = 0.0;
      if (p == 0)
        return 0.0;
      else if (p == 1)
        return 0.0;
      else if (p == 2)
        return ddL2;
      else if (p == 3)
        return ddL3;
      else {
        for (unsigned i = 3; i < p; i++) {
          ddL4 =
              1.0 / (i - 1.0) * ((2.0 * i + 1.0) * x * ddL3 - (i + 2.0) * ddL2);
          ddL2 = ddL3;
          ddL3 = ddL4;
        }
        return ddL4;
      }
    }
  } // namespace Orthpoly

} // namespace oomph

#endif
