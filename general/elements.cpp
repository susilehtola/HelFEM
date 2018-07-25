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
