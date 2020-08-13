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

#include "configurations.h"
#include <vector>
#include <string>

namespace helfem {
  namespace sadatom {

    arma::ivec get_configuration(int Z) {
      std::vector<std::string> confs(119);
      confs[1]="1s";
      confs[2]="2s"; // He

      confs[3]="2+1s";
      confs[4]="2+2s";
      confs[5]="2+2s1p";
      confs[6]="2+2s2p";
      confs[7]="2+2s3p";
      confs[8]="2+2s4p";
      confs[9]="2+2s5p";
      confs[10]="2+2s6p"; // Ne

      confs[11]="10+1s";
      confs[12]="10+2s";
      confs[13]="10+2s1p";
      confs[14]="10+2s2p";
      confs[15]="10+2s3p";
      confs[16]="10+2s4p";
      confs[17]="10+2s5p";
      confs[18]="10+2s6p"; // Ar

      confs[19]="18+1s";
      confs[20]="18+2s";
      confs[21]="18+2s1d";
      confs[22]="18+2s2d";
      confs[23]="18+2s3d";
      confs[24]="18+1s5d";
      confs[25]="18+2s5d";
      confs[26]="18+2s6d";
      confs[27]="18+2s7d";
      confs[28]="18+2s8d";
      confs[29]="18+1s10d";
      confs[30]="18+2s10d";
      confs[31]="18+2s10d1p";
      confs[32]="18+2s10d2p";
      confs[33]="18+2s10d3p";
      confs[34]="18+2s10d4p";
      confs[35]="18+2s10d5p";
      confs[36]="18+2s10d6p"; // Kr

      confs[37]="36+1s";
      confs[38]="36+2s";
      confs[39]="36+2s1d";
      confs[40]="36+2s2d";
      confs[41]="36+1s4d";
      confs[42]="36+1s5d";
      confs[43]="36+2s5d";
      confs[44]="36+1s7d";
      confs[45]="36+1s8d";
      confs[46]="36+10d";
      confs[47]="36+1s10d";
      confs[48]="36+2s10d";
      confs[49]="36+2s10d1p";
      confs[50]="36+2s10d2p";
      confs[51]="36+2s10d3p";
      confs[52]="36+2s10d4p";
      confs[53]="36+2s10d5p";
      confs[54]="36+2s10d6p"; // Xe

      confs[55]="54+1s";
      confs[56]="54+2s";
      confs[57]="54+2s1d";
      confs[58]="54+2s1f1d";
      confs[59]="54+2s3f";
      confs[60]="54+2s4f";
      confs[61]="54+2s5f";
      confs[62]="54+2s6f";
      confs[63]="54+2s7f";
      confs[64]="54+2s7f1d";
      confs[65]="54+2s9f";
      confs[66]="54+2s10f";
      confs[67]="54+2s11f";
      confs[68]="54+2s12f";
      confs[69]="54+2s13f";
      confs[70]="54+2s14f";
      confs[71]="54+2s14f1d";
      confs[72]="54+2s14f2d";
      confs[73]="54+2s14f3d";
      confs[74]="54+2s14f4d";
      confs[75]="54+2s14f5d";
      confs[76]="54+2s14f6d";
      confs[77]="54+2s14f7d";
      confs[78]="54+1s14f9d";
      confs[79]="54+1s14f10d";
      confs[80]="54+2s14f10d";
      confs[81]="54+2s14f10d1p";
      confs[82]="54+2s14f10d2p";
      confs[83]="54+2s14f10d3p";
      confs[84]="54+2s14f10d4p";
      confs[85]="54+2s14f10d5p";
      confs[86]="54+2s14f10d6p"; // Rn

      confs[87]="86+1s";
      confs[88]="86+2s";
      confs[89]="86+2s1d";
      confs[90]="86+2s2d";
      confs[91]="86+2s2f1d";
      confs[92]="86+2s3f1d";
      confs[93]="86+2s4f1d";
      confs[94]="86+2s6f";
      confs[95]="86+2s7f";
      confs[96]="86+2s7f1d";
      confs[97]="86+2s8f1d";
      confs[98]="86+2s10f";
      confs[99]="86+2s11f";
      confs[100]="86+2s12f";
      confs[101]="86+2s13f";
      confs[102]="86+2s14f";
      confs[103]="86+2s14f1d";
      confs[104]="86+2s14f2d";
      confs[105]="86+2s14f3d";
      confs[106]="86+2s14f4d";
      confs[107]="86+2s14f5d";
      confs[108]="86+2s14f6d";
      confs[109]="86+2s14f7d";
      confs[110]="86+1s14f9d";
      confs[111]="86+1s14f10d";
      confs[112]="86+2s14f10d";
      confs[113]="86+2s14f10d1p";
      confs[114]="86+2s14f10d2p";
      confs[115]="86+2s14f10d3p";
      confs[116]="86+2s14f10d4p";
      confs[117]="86+2s14f10d5p";
      confs[118]="86+2s14f10d6p"; // Og

      arma::ivec config(4);
      config.zeros();
      arma::ivec core(4);
      core.zeros();

      if(Z > (int) confs.size())
        throw std::logic_error("Unsupported element.\n");

      // Grab the string
      std::string cfg(confs[Z]);

      // Does it have a core?
      size_t pos(cfg.find('+'));
      if(pos != std::string::npos) {
        int Zcore;
        std::istringstream input(cfg.substr(0,pos));
        input >> Zcore;
        core=get_configuration(Zcore);

        // Drop the core part
        std::string newcfg(cfg.substr(pos+1));
        cfg=newcfg;
      }

      // Parse through the string
      for(size_t ipos=0;ipos<cfg.size();ipos++) {
        size_t jpos;
        for(jpos=ipos;jpos<cfg.size();jpos++)
          if(isalpha(cfg[jpos]))
            break;

        // Get the substring
        std::istringstream numel(cfg.substr(ipos,jpos-ipos));
        int N;
        numel >> N;
        switch(cfg[jpos]) {
        case('s'):
          config(0)+=N;
          break;
        case('p'):
          config(1)+=N;
          break;
        case('d'):
          config(2)+=N;
          break;
        case('f'):
          config(3)+=N;
          break;
        default:
          throw std::logic_error("Error!\n");
        }

        // Jump to the end
        ipos=jpos;
      }

      config+=core;
      if(arma::sum(config)!=Z) {
        std::ostringstream oss;
        oss << "Error for Z = " << Z << ": occupations sum to " << arma::sum(config) << "!\n";
        throw std::logic_error(oss.str());
      }

      return config;
    }
  }
}
