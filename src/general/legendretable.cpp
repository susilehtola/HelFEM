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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include "legendretable.h"
#include <algorithm>
#include "../legendre/Legendre_Wrapper.h"

namespace helfem {
  namespace legendretable {
    bool operator<(const legendre_table_t & lh, const legendre_table_t & rh) {
      return lh.xi < rh.xi;
    }

    LegendreTable::LegendreTable() {
      Lpad=-1;
      Lmax=-1;
      Mmax=-1;
    }

    LegendreTable::LegendreTable(int Lpad_, int Lmax_, int Mmax_) : Lpad(Lpad_), Lmax(Lmax_), Mmax(Mmax_) {
    }

    LegendreTable::~LegendreTable() {
    }

    size_t LegendreTable::get_index(double xi, bool check) const {
      legendre_table_t p;
      p.xi=xi;

      std::vector<legendre_table_t>::const_iterator low(std::lower_bound(stor.begin(),stor.end(),p));
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
        legendre_table_t entry;

        // Allocate memory
        entry.xi=xi;
        entry.Plm.zeros(Lpad+1,Lpad+1);
        entry.Qlm.zeros(Lpad+1,Lpad+1);

        // Compute
        if(xi!=1.0) {
          ::calc_Plm_arr(entry.Plm.memptr(),Lpad,Lpad,xi);
          ::calc_Qlm_arr(entry.Qlm.memptr(),Lpad,Lpad,xi);
        }

        // Store only 0 to lmax
        entry.Plm=entry.Plm.submat(0,0,Lmax,Mmax);
        entry.Qlm=entry.Qlm.submat(0,0,Lmax,Mmax);

        // Get rid of any non-normal entries
        for(int L=0;L<=Lmax;L++)
          for(int M=0;M<=Mmax;M++) {
            if(!std::isnormal(entry.Plm(L,M)))
              entry.Plm(L,M)=0.0;
            if(!std::isnormal(entry.Qlm(L,M)))
              entry.Qlm(L,M)=0.0;
          }

        if(!stor.size())
          stor.push_back(entry);
        else
          // Insert at lower bound
          stor.insert(stor.begin()+get_index(xi,false),entry);
      }
    }

    double LegendreTable::get_Plm(int l, int m, double xi) const {
      return stor[get_index(xi)].Plm(l,m);
    }

    double LegendreTable::get_Qlm(int l, int m, double xi) const {
      return stor[get_index(xi)].Qlm(l,m);
    }

    arma::vec LegendreTable::get_Plm(int l, int m, const arma::vec & xi) const {
      arma::vec plm(xi.n_elem);
      for(size_t i=0;i<xi.n_elem;i++)
        plm(i)=get_Plm(l,m,xi(i));
      return plm;
    }

    arma::vec LegendreTable::get_Qlm(int l, int m, const arma::vec & xi) const {
      arma::vec qlm(xi.n_elem);
      for(size_t i=0;i<xi.n_elem;i++)
        qlm(i)=get_Qlm(l,m,xi(i));
      return qlm;
    }
  }
}
