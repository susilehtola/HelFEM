/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include "elements.h"
#include <cstdio>
#include <cstdlib>
#include <strings.h>
#include <sstream>
#include <stdexcept>

static int stricmp(const std::string & str1, const std::string & str2) {
  return strcasecmp(str1.c_str(),str2.c_str());
}

int get_Z(std::string el) {
  if(!el.size())
    // Assume no nucleus
    return 0;
  
  // Is input an integer?
  if(!isalpha(el[0]))
    return atoi(el.c_str());

  // Parse for input
  for(int Z=1;Z<(int) (sizeof(element_symbols)/sizeof(element_symbols[0]));Z++)
    if(stricmp(el,element_symbols[Z])==0)
      return Z;

  std::ostringstream oss;
  oss << "Element \"" << el << "\" not found in table of elements!\n";
  throw std::runtime_error(oss.str());

  // Not found, return dummy charge.
  return 0;
}

static void num_orbs(int Z, int & sigma, int & pi, int & delta, int & phi) {
  const int shellZ[]={1, 3, 5, 11, 13, 19, 21, 31, 37, 39, 49, 55, 57, 71, 81, 87, 89, 103, 113};
  const int shellL[]={0, 0, 1,  0,  1,  0,  2,  1,  0,  2,  1,  0,  3,  2,  1,  0,  3,   2,   1};

  for(size_t ii=0;ii<sizeof(shellZ)/sizeof(shellZ[0]);ii++) {
    if(Z>=shellZ[ii]) {
      switch(shellL[ii]) {
      case(3):
        phi++;
      case(2):
        delta++;
      case(1):
        pi++;
      case(0):
        sigma++;
        break;

      default:
        throw std::logic_error("Unknown shell type.\n");
      }
    }
  }
}

void num_orbs(int Z1, int Z2, int & sigma, int & pi, int & delta, int & phi) {
  sigma=0;
  pi=0;
  delta=0;
  phi=0;
  num_orbs(Z1,sigma,pi,delta,phi);
  num_orbs(Z2,sigma,pi,delta,phi);
}
