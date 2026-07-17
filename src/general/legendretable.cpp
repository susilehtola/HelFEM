/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#include "legendretable.h"
#include <algorithm>
#include "../legendre/Legendre.h"

namespace helfem {
  namespace legendretable {
    bool operator<(const legendre_table_t & lh, const legendre_table_t & rh) {
      return lh.xi < rh.xi;
    }

    // Compute the P_{l,m} / Q_{l,m} block at a single argument xi. This is the
    // shared kernel behind both compute() (which caches the result) and the
    // on-miss path of the getters (which computes it directly for arguments
    // that were never tabulated -- e.g. the auto-converging two-electron
    // quadrature evaluates the Legendre functions at points chosen at a
    // quadrature order OTHER than the one the table was built for). Keeping it
    // in one place guarantees the direct and cached values are bit-identical.
    static legendre_table_t compute_entry(int Lmax, int Mmax, double xi) {
      legendre_table_t entry;
      entry.xi=xi;
      entry.Plm=helfem::Matrix::Zero(Lmax+1,Mmax+1);
      entry.Qlm=helfem::Matrix::Zero(Lmax+1,Mmax+1);

      // Q is singular at xi == 1; the table treats that case as zero
      // everywhere via the ::Zero above.
      if(xi!=1.0) {
        ::helfem::legendre::plm(entry.Plm.data(),Lmax,Mmax,xi);
        ::helfem::legendre::qlm(entry.Qlm.data(),Lmax,Mmax,xi);
      }

      // Drop any non-normal entries (zero is fine, denormals/NaN/Inf get
      // forced to zero so that downstream code doesn't propagate them).
      for(int L=0;L<=Lmax;L++)
        for(int M=0;M<=Mmax;M++) {
          if(entry.Plm(L,M) != 0.0 && !std::isnormal(entry.Plm(L,M)))
            entry.Plm(L,M)=0.0;
          if(entry.Qlm(L,M) != 0.0 && !std::isnormal(entry.Qlm(L,M)))
            entry.Qlm(L,M)=0.0;
        }

      return entry;
    }

    // Single-value version of the above, for the getters' on-miss path: compute
    // just the requested function (P if wantP, else Q) at xi, bit-identically
    // to what compute_entry / compute() would cache -- same plm/qlm arguments,
    // same (Lmax+1)-row column-major layout, same xi==1 and isnormal handling.
    // Avoids allocating the two Eigen blocks and computing the unused sibling
    // function on every miss.
    static double compute_one(bool wantP, int l, int m, int Lmax, int Mmax, double xi) {
      // Q is singular at xi == 1; the table treats that case as zero.
      if(xi==1.0)
        return 0.0;
      std::vector<double> buf((size_t) (Lmax+1)*(Mmax+1), 0.0);
      if(wantP)
        ::helfem::legendre::plm(buf.data(),Lmax,Mmax,xi);
      else
        ::helfem::legendre::qlm(buf.data(),Lmax,Mmax,xi);
      // Column-major (Lmax+1) x (Mmax+1): element (l,m) is at l + m*(Lmax+1).
      double v = buf[(size_t) l + (size_t) m*(Lmax+1)];
      if(v!=0.0 && !std::isnormal(v))
        v=0.0;
      return v;
    }

    LegendreTable::LegendreTable() {
      Lmax=-1;
      Mmax=-1;
    }

    LegendreTable::LegendreTable(int Lmax_, int Mmax_) : Lmax(Lmax_), Mmax(Mmax_) {
    }

    LegendreTable::~LegendreTable() {
    }

    size_t LegendreTable::get_index(double xi, bool check) const {
      // Search on xi alone. Constructing a legendre_table_t just to carry the
      // search key would construct -- and immediately destroy -- its two
      // helfem::Matrix members on EVERY lookup, and get_index is called once per
      // quadrature point, inside the (L, M, subinterval) loops of the
      // two-electron integral build. That churn showed up as several percent
      // of the run time in malloc/memmove.
      std::vector<legendre_table_t>::const_iterator low(
          std::lower_bound(stor.begin(), stor.end(), xi,
                           [](const legendre_table_t & e, double v) {
                             return e.xi < v;
                           }));
      if(check && low == stor.end()) {
        std::ostringstream oss;
        oss << "Could not find xi=" << xi << " on the list!\n";
        throw std::logic_error(oss.str());
      }

      // Index is
      size_t idx(low-stor.begin());
      if(check && (stor[idx].xi != xi)) {
        std::ostringstream oss;
        oss << "Map error: tried to get xi = " << xi << " but got xi = " << stor[idx].xi << "!\n";
        throw std::logic_error(oss.str());
      }

      return idx;
    }

    void LegendreTable::compute(double xi) {
#ifdef _OPENMP
#pragma omp critical
#endif
      {
        legendre_table_t entry(compute_entry(Lmax, Mmax, xi));

        if(!stor.size())
          stor.push_back(entry);
        else
          // Insert at lower bound
          stor.insert(stor.begin()+get_index(xi,false),entry);
      }
    }

    double LegendreTable::get_Plm(int l, int m, double xi) const {
      // Fast path: the argument was tabulated (the stored-order quadrature
      // points the table was built for). get_index(check=false) returns the
      // lower-bound slot without throwing; a hit is an exact xi match.
      const size_t idx(get_index(xi, false));
      if(idx < stor.size() && stor[idx].xi == xi)
        return stor[idx].Plm(l,m);
      // Miss: an argument at some other quadrature order (the auto-converging
      // two-electron refinement). Compute it directly -- bit-identical to what
      // the table would hold -- instead of forcing the caller onto a fixed
      // --nquad grid.
      return compute_one(true, l, m, Lmax, Mmax, xi);
    }

    double LegendreTable::get_Qlm(int l, int m, double xi) const {
      const size_t idx(get_index(xi, false));
      if(idx < stor.size() && stor[idx].xi == xi)
        return stor[idx].Qlm(l,m);
      return compute_one(false, l, m, Lmax, Mmax, xi);
    }

    helfem::Vector LegendreTable::get_Plm(int l, int m, const helfem::Vector & xi) const {
      helfem::Vector plm(xi.size());
      for(size_t i=0;i<(size_t) xi.size();i++)
        plm(i)=get_Plm(l,m,xi(i));
      return plm;
    }

    helfem::Vector LegendreTable::get_Qlm(int l, int m, const helfem::Vector & xi) const {
      helfem::Vector qlm(xi.size());
      for(size_t i=0;i<(size_t) xi.size();i++)
        qlm(i)=get_Qlm(l,m,xi(i));
      return qlm;
    }
  }
}
