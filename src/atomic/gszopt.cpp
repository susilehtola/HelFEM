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
#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"
#include "../general/polynomial_basis.h"
#include "basis.h"
#include "dftgrid.h"
#include <cfloat>

using namespace helfem;

typedef std::pair<int, double> hfentry;

bool operator<(const hfentry & l, const hfentry & r) {
  return l.second < r.second;
}

std::vector<hfentry> get_energies(int Z) {
  // Energy vector
  std::vector<hfentry> Ev;

  switch(Z) {
    // H
  case(1):
    Ev.push_back(hfentry(0, 0.5));
    break;

    // He
  case(2):
    Ev.push_back(hfentry(0, 0.917956));
    break;

    // Li
  case(3):
    Ev.push_back(hfentry(0, 2.477741));
    Ev.push_back(hfentry(0, 0.196323));
    break;

    // Be
  case(4):
    Ev.push_back(hfentry(0, 4.732670));
    Ev.push_back(hfentry(0, 0.309270));
    break;

    // B
  case(5):
    Ev.push_back(hfentry(0, 7.695335));
    Ev.push_back(hfentry(0, 0.494706));
    Ev.push_back(hfentry(1, 0.309856));    
    break;

    // C
  case(6):
    Ev.push_back(hfentry(0, 11.325519));
    Ev.push_back(hfentry(0, 0.705627));
    Ev.push_back(hfentry(1, 0.433341));
    break;

    // N
  case(7):
    Ev.push_back(hfentry(0, 15.629060));
    Ev.push_back(hfentry(0, 0.945324));
    Ev.push_back(hfentry(1, 0.567589));
    break;

    // O
  case(8):
    Ev.push_back(hfentry(0, 20.668657));
    Ev.push_back(hfentry(0, 1.244315));
    Ev.push_back(hfentry(1, 0.631906));
    break;
    
    // F
  case(9):
    Ev.push_back(hfentry(0, 26.382760));
    Ev.push_back(hfentry(0, 1.572535));
    Ev.push_back(hfentry(1, 0.730018));
    break;

    // Ne
  case(10):
    Ev.push_back(hfentry(0, 32.772443));
    Ev.push_back(hfentry(0, 1.930391));
    Ev.push_back(hfentry(1, 0.850410));
    break;    

    // Na
  case(11):
    Ev.push_back(hfentry(0, 40.478501)); 
    Ev.push_back(hfentry(0, 2.797026));
    Ev.push_back(hfentry(0, 0.182103));
    Ev.push_back(hfentry(1, 1.518140));  
    break;
    
    // Mg
  case(12):
    Ev.push_back(hfentry(0, 49.031736));
    Ev.push_back(hfentry(0, 3.767721));
    Ev.push_back(hfentry(0, 0.253053)); 
    Ev.push_back(hfentry(1, 2.282226));
    break;
    
    // Al
  case(13):
    Ev.push_back(hfentry(0, 58.501027)); 
    Ev.push_back(hfentry(0, 4.910672));
    Ev.push_back(hfentry(0, 0.393420));
    Ev.push_back(hfentry(1, 3.218303));
    Ev.push_back(hfentry(1, 0.209951));
    break;
    
    // Si
  case(14):
    Ev.push_back(hfentry(0, 68.812456));
    Ev.push_back(hfentry(0, 6.156538));
    Ev.push_back(hfentry(0, 0.539842));
    Ev.push_back(hfentry(1, 4.256054));
    Ev.push_back(hfentry(1, 0.297115));
    break;

    // P
  case(15):
    Ev.push_back(hfentry(0, 79.969715));
    Ev.push_back(hfentry(0, 7.511096));
    Ev.push_back(hfentry(0, 0.696416));
    Ev.push_back(hfentry(1, 5.400959));
    Ev.push_back(hfentry(1, 0.391708));
    break;

    // S
  case(16):
    Ev.push_back(hfentry(0, 92.004450));
    Ev.push_back(hfentry(0, 9.004290));
    Ev.push_back(hfentry(0, 0.879527));
    Ev.push_back(hfentry(1, 6.682509));  
    Ev.push_back(hfentry(1, 0.437369));
    break;

    // Cl
  case(17):
    Ev.push_back(hfentry(0, 104.884421));
    Ev.push_back(hfentry(0, 10.607481));
    Ev.push_back(hfentry(0, 1.072912));
    Ev.push_back(hfentry(1, 8.072228));
    Ev.push_back(hfentry(1, 0.506400));
    break;

    // Ar
  case(18):
    Ev.push_back(hfentry(0, 118.610351));
    Ev.push_back(hfentry(0, 12.322153));
    Ev.push_back(hfentry(0, 1.277353));
    Ev.push_back(hfentry(1, 9.571466));
    Ev.push_back(hfentry(1, 0.591017));
    break;

    // K
  case(19):
    Ev.push_back(hfentry(0, 133.533050));
    Ev.push_back(hfentry(0, 14.489958));
    Ev.push_back(hfentry(0, 1.748781));
    Ev.push_back(hfentry(0, 0.147475));
    Ev.push_back(hfentry(1, 11.519280));
    Ev.push_back(hfentry(1, 0.954423));
    break;

    // Ca
  case(20):
    Ev.push_back(hfentry(0, 149.363726));
    Ev.push_back(hfentry(0, 16.822744));
    Ev.push_back(hfentry(0, 2.245376));
    Ev.push_back(hfentry(0, 0.195530));
    Ev.push_back(hfentry(1, 13.629269));
    Ev.push_back(hfentry(1, 1.340707));
    break;
			 
    // Sc
  case(21):
    Ev.push_back(hfentry(0, 165.899899));
    Ev.push_back(hfentry(0, 19.080621));
    Ev.push_back(hfentry(0, 2.567324));
    Ev.push_back(hfentry(0, 0.210109));
    Ev.push_back(hfentry(1, 15.668247));
    Ev.push_back(hfentry(1, 1.574549));
    Ev.push_back(hfentry(2, 0.343712));
    break;

    // Ti
  case(22):
    Ev.push_back(hfentry(0, 183.272759));
    Ev.push_back(hfentry(0, 21.422913));
    Ev.push_back(hfentry(0, 2.873398));
    Ev.push_back(hfentry(0, 0.220788));
    Ev.push_back(hfentry(1, 17.791188));
    Ev.push_back(hfentry(1, 1.795088));
    Ev.push_back(hfentry(2, 0.440656));
    break;

    // V
  case(23):
    Ev.push_back(hfentry(0, 201.502832));
    Ev.push_back(hfentry(0, 23.874653));
    Ev.push_back(hfentry(0, 3.183186)); 
    Ev.push_back(hfentry(0, 0.230579));
    Ev.push_back(hfentry(1, 20.022491));
    Ev.push_back(hfentry(1, 2.019226));
    Ev.push_back(hfentry(2, 0.509620));
    break;

    // Cr
  case(24):
    Ev.push_back(hfentry(0, 220.386406));
    Ev.push_back(hfentry(0, 26.209636));
    Ev.push_back(hfentry(0, 3.285161));
    Ev.push_back(hfentry(0, 0.222050));
    Ev.push_back(hfentry(1, 22.139856));
    Ev.push_back(hfentry(1, 2.050932));
    Ev.push_back(hfentry(2, 0.373605));
    break;

    // Mn
  case(25):
    Ev.push_back(hfentry(0, 240.533991));
    Ev.push_back(hfentry(0, 29.109475));
    Ev.push_back(hfentry(0, 3.816646));
    Ev.push_back(hfentry(0, 0.247871));
    Ev.push_back(hfentry(1, 24.812589));
    Ev.push_back(hfentry(1, 2.479531));
    Ev.push_back(hfentry(2, 0.638847));
    break;

    // Fe
  case(26):
    Ev.push_back(hfentry(0, 261.373422));
    Ev.push_back(hfentry(0, 31.935522));
    Ev.push_back(hfentry(0, 4.169440));
    Ev.push_back(hfentry(0, 0.258181));
    Ev.push_back(hfentry(1, 27.413712));
    Ev.push_back(hfentry(1, 2.742199));
    Ev.push_back(hfentry(2, 0.646888));
    break;

    // Co
  case(27):
    Ev.push_back(hfentry(0, 283.065505));
    Ev.push_back(hfentry(0, 34.868335));
    Ev.push_back(hfentry(0, 4.524290));
    Ev.push_back(hfentry(0, 0.267421));
    Ev.push_back(hfentry(1, 30.120174));
    Ev.push_back(hfentry(1, 3.006245));
    Ev.push_back(hfentry(2, 0.675419));
    break;

    // Ni
  case(28):
    Ev.push_back(hfentry(0, 305.619035));
    Ev.push_back(hfentry(0, 37.917832));
    Ev.push_back(hfentry(0, 4.887837));
    Ev.push_back(hfentry(0, 0.276251));
    Ev.push_back(hfentry(1, 32.941737));
    Ev.push_back(hfentry(1, 3.277679));
    Ev.push_back(hfentry(2, 0.706932));
    break;

    // Cu
  case(29):
    Ev.push_back(hfentry(0, 328.792979));
    Ev.push_back(hfentry(0, 40.818957));
    Ev.push_back(hfentry(0, 5.011981));
    Ev.push_back(hfentry(0, 0.238494));
    Ev.push_back(hfentry(1, 35.617945));
    Ev.push_back(hfentry(1, 3.324822));
    Ev.push_back(hfentry(2, 0.491233));
    break;

    // Zn
  case(30):
    Ev.push_back(hfentry(0, 353.304540));
    Ev.push_back(hfentry(0, 44.361720));
    Ev.push_back(hfentry(0, 5.637816));
    Ev.push_back(hfentry(0, 0.292507));
    Ev.push_back(hfentry(1, 38.924839));
    Ev.push_back(hfentry(1, 3.839373));
    Ev.push_back(hfentry(2, 0.782537));
    break;

    // Ga
  case(31):
    Ev.push_back(hfentry(0, 378.818423));
    Ev.push_back(hfentry(0, 48.168423));
    Ev.push_back(hfentry(0, 6.394657));
    Ev.push_back(hfentry(0, 0.424592));
    Ev.push_back(hfentry(1, 42.494029));
    Ev.push_back(hfentry(1, 4.482368));
    Ev.push_back(hfentry(1, 0.208500));
    Ev.push_back(hfentry(2, 1.193370));
    break;

    // Ge
  case(32):
    Ev.push_back(hfentry(0, 405.244452));
    Ev.push_back(hfentry(0, 52.150337));
    Ev.push_back(hfentry(0, 7.190999));
    Ev.push_back(hfentry(0, 0.553364));
    Ev.push_back(hfentry(1, 46.236162));
    Ev.push_back(hfentry(1, 5.161600));
    Ev.push_back(hfentry(1, 0.287354));
    Ev.push_back(hfentry(2, 1.634898));
    break;

    // As
  case(33):
    Ev.push_back(hfentry(0, 432.586204));
    Ev.push_back(hfentry(0, 56.309828));
    Ev.push_back(hfentry(0, 8.029623));
    Ev.push_back(hfentry(0, 0.685896));
    Ev.push_back(hfentry(1, 50.153745));
    Ev.push_back(hfentry(1, 5.880694));
    Ev.push_back(hfentry(1, 0.369482));
    Ev.push_back(hfentry(2, 2.112657));
    break;
    
    // Se
  case(34):
    Ev.push_back(hfentry(0, 460.867406));
    Ev.push_back(hfentry(0, 60.668873));
    Ev.push_back(hfentry(0, 8.932103));
    Ev.push_back(hfentry(0, 0.837382));
    Ev.push_back(hfentry(1, 54.268901));
    Ev.push_back(hfentry(1, 6.661523));
    Ev.push_back(hfentry(1, 0.402855));
    Ev.push_back(hfentry(2, 2.649627));
    break;
    
    // Br
  case(35):
    Ev.push_back(hfentry(0, 490.060340));
    Ev.push_back(hfentry(0, 65.199960));
    Ev.push_back(hfentry(0, 9.871894));
    Ev.push_back(hfentry(0, 0.992680));
    Ev.push_back(hfentry(1, 58.554224));
    Ev.push_back(hfentry(1, 7.478209));
    Ev.push_back(hfentry(1, 0.457086));
    Ev.push_back(hfentry(2, 3.220175));
    break;
    
    // Kr
  case(36):
    Ev.push_back(hfentry(0, 520.165468));
    Ev.push_back(hfentry(0, 69.903082));
    Ev.push_back(hfentry(0, 10.849466));
    Ev.push_back(hfentry(0, 1.152935));
    Ev.push_back(hfentry(1, 63.009785));
    Ev.push_back(hfentry(1, 8.331501));
    Ev.push_back(hfentry(1, 0.524187));
    Ev.push_back(hfentry(2, 3.825234));
    break;

    // Rb
  case(37):
    Ev.push_back(hfentry(0, 551.457338));
    Ev.push_back(hfentry(0, 75.049347));
    Ev.push_back(hfentry(0, 12.133200));
    Ev.push_back(hfentry(0, 1.523549));
    Ev.push_back(hfentry(0, 0.137867)); 
    Ev.push_back(hfentry(1, 67.906232));
    Ev.push_back(hfentry(1, 9.487696));
    Ev.push_back(hfentry(1, 0.810071));
    Ev.push_back(hfentry(2, 4.732292));
    break;

    // Sr
  case(38):
    Ev.push_back(hfentry(0, 583.687885));
    Ev.push_back(hfentry(0, 80.390796));
    Ev.push_back(hfentry(0, 13.475029));
    Ev.push_back(hfentry(0, 1.896812));
    Ev.push_back(hfentry(0, 0.178456));
    Ev.push_back(hfentry(1, 72.996038));
    Ev.push_back(hfentry(1, 10.699976));
    Ev.push_back(hfentry(1, 1.098163));
    Ev.push_back(hfentry(2, 5.694396));
    break;

    // Y
  case(39):
    Ev.push_back(hfentry(0, 616.749351));
    Ev.push_back(hfentry(0, 85.810945));
    Ev.push_back(hfentry(0, 14.758916));
    Ev.push_back(hfentry(0, 2.168878));
    Ev.push_back(hfentry(0, 0.196143));
    Ev.push_back(hfentry(1, 78.164472));
    Ev.push_back(hfentry(1, 11.854193));
    Ev.push_back(hfentry(1, 1.301188));
    Ev.push_back(hfentry(2, 6.599480));
    Ev.push_back(hfentry(2, 0.249849));
    break;

    // Zr
  case(40):
    Ev.push_back(hfentry(0, 650.704986));
    Ev.push_back(hfentry(0, 91.377690));
    Ev.push_back(hfentry(0, 16.055029));
    Ev.push_back(hfentry(0, 2.419195));
    Ev.push_back(hfentry(0, 0.207293));
    Ev.push_back(hfentry(1, 83.478543));
    Ev.push_back(hfentry(1, 13.019768));
    Ev.push_back(hfentry(1, 1.487595)); 
    Ev.push_back(hfentry(2, 7.515841));
    Ev.push_back(hfentry(2, 0.336759));
    break;

    // Nb
  case(41):
    Ev.push_back(hfentry(0, 685.444022)); 
    Ev.push_back(hfentry(0, 96.974832));
    Ev.push_back(hfentry(0, 17.247068));
    Ev.push_back(hfentry(0, 2.537478));
    Ev.push_back(hfentry(0, 0.215594));
    Ev.push_back(hfentry(1, 88.823126));
    Ev.push_back(hfentry(1, 14.081429));
    Ev.push_back(hfentry(1, 1.556994));
    Ev.push_back(hfentry(2, 8.329295));
    Ev.push_back(hfentry(2, 0.300647));
    break;

    // Mo
  case(42):
    Ev.push_back(hfentry(0, 721.202198));
    Ev.push_back(hfentry(0, 102.850600));
    Ev.push_back(hfentry(0, 18.584536));
    Ev.push_back(hfentry(0, 2.762914));
    Ev.push_back(hfentry(0, 0.222729));
    Ev.push_back(hfentry(1, 94.444090));
    Ev.push_back(hfentry(1, 15.286564));
    Ev.push_back(hfentry(1, 1.723633));
    Ev.push_back(hfentry(2, 9.284225));
    Ev.push_back(hfentry(2, 0.357918));
    break;

    // Tc
  case(43):
    Ev.push_back(hfentry(0, 758.043047));
    Ev.push_back(hfentry(0, 109.069787));
    Ev.push_back(hfentry(0, 20.131835));
    Ev.push_back(hfentry(0, 3.152190));
    Ev.push_back(hfentry(0, 0.231274));
    Ev.push_back(hfentry(1, 100.406091));
    Ev.push_back(hfentry(1, 16.699577)); 
    Ev.push_back(hfentry(1, 2.041219));
    Ev.push_back(hfentry(2, 10.444638));
    Ev.push_back(hfentry(2, 0.543953));
    break;

    // Ru
  case(44):
    Ev.push_back(hfentry(0, 795.513471));
    Ev.push_back(hfentry(0, 115.158792));
    Ev.push_back(hfentry(0, 21.413885));
    Ev.push_back(hfentry(0, 3.257128));
    Ev.push_back(hfentry(0, 0.222428));
    Ev.push_back(hfentry(1, 106.239392));
    Ev.push_back(hfentry(1, 17.848751));
    Ev.push_back(hfentry(1, 2.101248));
    Ev.push_back(hfentry(2, 11.343342));
    Ev.push_back(hfentry(2, 0.412778));
    break;
    
    // Rh
  case(45):
    Ev.push_back(hfentry(0, 834.039473));
    Ev.push_back(hfentry(0, 121.563647));
    Ev.push_back(hfentry(0, 22.879612));
    Ev.push_back(hfentry(0, 3.503891));
    Ev.push_back(hfentry(0, 0.221618));
    Ev.push_back(hfentry(1, 112.386179));
    Ev.push_back(hfentry(1, 19.179593));
    Ev.push_back(hfentry(1, 2.291149));
    Ev.push_back(hfentry(2, 12.421289));
    Ev.push_back(hfentry(2, 0.450195));
    break;

    // Pd
  case(46):
    Ev.push_back(hfentry(0, 873.315934));
    Ev.push_back(hfentry(0, 127.966574));
    Ev.push_back(hfentry(0, 24.209104));
    Ev.push_back(hfentry(0, 3.587310));
    Ev.push_back(hfentry(1, 118.531104));
    Ev.push_back(hfentry(1, 20.374285));
    Ev.push_back(hfentry(1, 2.330090));
    Ev.push_back(hfentry(2, 13.363440));
    Ev.push_back(hfentry(2, 0.336002));
    break;

    // Ag
  case(47):
    Ev.push_back(hfentry(0, 913.835600));
    Ev.push_back(hfentry(0, 134.878416));
    Ev.push_back(hfentry(0, 25.917827));
    Ev.push_back(hfentry(0, 4.001498));
    Ev.push_back(hfentry(0, 0.219980));
    Ev.push_back(hfentry(1, 125.181588));
    Ev.push_back(hfentry(1, 21.945436));
    Ev.push_back(hfentry(1, 2.676820));
    Ev.push_back(hfentry(2, 14.678202));
    Ev.push_back(hfentry(2, 0.537400));
    break;

    // Cd
  case(48):
    Ev.push_back(hfentry(0, 955.315356));
    Ev.push_back(hfentry(0, 142.006827));
    Ev.push_back(hfentry(0, 27.708623));
    Ev.push_back(hfentry(0, 4.450535));
    Ev.push_back(hfentry(0, 0.264856));
    Ev.push_back(hfentry(1, 132.047020));
    Ev.push_back(hfentry(1, 23.597234));
    Ev.push_back(hfentry(1, 3.053504));
    Ev.push_back(hfentry(2, 16.071971));
    Ev.push_back(hfentry(2, 0.763658));
    break;

    // In
  case(49):
    Ev.push_back(hfentry(0, 997.800457));
    Ev.push_back(hfentry(0, 149.395446));
    Ev.push_back(hfentry(0, 29.624667));
    Ev.push_back(hfentry(0, 4.976702));
    Ev.push_back(hfentry(0, 0.372663));
    Ev.push_back(hfentry(1, 139.171929));
    Ev.push_back(hfentry(1, 25.374284));
    Ev.push_back(hfentry(1, 3.507216));
    Ev.push_back(hfentry(1, 0.197283));
    Ev.push_back(hfentry(2, 17.589565));
    Ev.push_back(hfentry(2, 1.063138));
    break;

    // Sn
  case(50):
    Ev.push_back(hfentry(0, 1041.223349));
    Ev.push_back(hfentry(0, 156.977583));
    Ev.push_back(hfentry(0, 31.598981));
    Ev.push_back(hfentry(0, 5.512501));
    Ev.push_back(hfentry(0, 0.476434));
    Ev.push_back(hfentry(1, 146.489269));
    Ev.push_back(hfentry(1, 27.209044));
    Ev.push_back(hfentry(1, 3.969055));
    Ev.push_back(hfentry(1, 0.265040));
    Ev.push_back(hfentry(2, 19.163358));
    Ev.push_back(hfentry(2, 1.369045)); 
    break;

    // Sb
  case(51):
    Ev.push_back(hfentry(0, 1085.589045));
    Ev.push_back(hfentry(0, 164.757961));
    Ev.push_back(hfentry(0, 33.636220)); 
    Ev.push_back(hfentry(0, 6.063185));
    Ev.push_back(hfentry(0, 0.581773)); 
    Ev.push_back(hfentry(1, 154.003792));
    Ev.push_back(hfentry(1, 29.106148));
    Ev.push_back(hfentry(1, 4.444718));
    Ev.push_back(hfentry(1, 0.334711));
    Ev.push_back(hfentry(2, 20.798076));
    Ev.push_back(hfentry(2, 1.687866)); 
    break;

    // Te
  case(52):
    Ev.push_back(hfentry(0, 1130.917005));
    Ev.push_back(hfentry(0, 172.755410));
    Ev.push_back(hfentry(0, 35.754892));
    Ev.push_back(hfentry(0, 6.647019));
    Ev.push_back(hfentry(0, 0.700561));
    Ev.push_back(hfentry(1, 161.734379));
    Ev.push_back(hfentry(1, 31.084050));
    Ev.push_back(hfentry(1, 4.952569));
    Ev.push_back(hfentry(1, 0.359832));
    Ev.push_back(hfentry(2, 22.512352));
    Ev.push_back(hfentry(2, 2.038289));
    break;

    // I
  case(53):
    Ev.push_back(hfentry(0, 1177.186297));
    Ev.push_back(hfentry(0, 180.949223));
    Ev.push_back(hfentry(0, 37.934467));
    Ev.push_back(hfentry(0, 7.244360));
    Ev.push_back(hfentry(0, 0.821117));
    Ev.push_back(hfentry(1, 169.660349));
    Ev.push_back(hfentry(1, 33.122310));
    Ev.push_back(hfentry(1, 5.473358));
    Ev.push_back(hfentry(1, 0.403181));
    Ev.push_back(hfentry(2, 24.285692));
    Ev.push_back(hfentry(2, 2.401205));
    break;

    // Xe
  case(54):
    Ev.push_back(hfentry(0, 1224.397777));
    Ev.push_back(hfentry(0, 189.340123));
    Ev.push_back(hfentry(0, 40.175663));
    Ev.push_back(hfentry(0, 7.856302));
    Ev.push_back(hfentry(0, 0.944414)); 
    Ev.push_back(hfentry(1, 177.782449));
    Ev.push_back(hfentry(1, 35.221662));
    Ev.push_back(hfentry(1, 6.008338));
    Ev.push_back(hfentry(1, 0.457290));
    Ev.push_back(hfentry(2, 26.118869));
    Ev.push_back(hfentry(2, 2.777881));
    break;

    // Cs
  case(55):
    Ev.push_back(hfentry(0, 1272.768831));
    Ev.push_back(hfentry(0, 198.143782));
    Ev.push_back(hfentry(0, 42.693034));
    Ev.push_back(hfentry(0, 8.695490));
    Ev.push_back(hfentry(0, 1.231607));
    Ev.push_back(hfentry(0, 0.123668));
    Ev.push_back(hfentry(1, 186.316176));
    Ev.push_back(hfentry(1, 37.595936));
    Ev.push_back(hfentry(1, 6.768534));
    Ev.push_back(hfentry(1, 0.683475));
    Ev.push_back(hfentry(2, 28.226223));
    Ev.push_back(hfentry(2, 3.379542));
    break;

    // Ba
  case(56):
    Ev.push_back(hfentry(0, 1322.093398));
    Ev.push_back(hfentry(0, 207.154465));
    Ev.push_back(hfentry(0, 45.280852));
    Ev.push_back(hfentry(0, 9.556400));
    Ev.push_back(hfentry(0, 1.512722));
    Ev.push_back(hfentry(0, 0.157528));
    Ev.push_back(hfentry(1, 195.055960));
    Ev.push_back(hfentry(1, 40.039790));
    Ev.push_back(hfentry(1, 7.549318));
    Ev.push_back(hfentry(1, 0.903863));
    Ev.push_back(hfentry(2, 30.402309));
    Ev.push_back(hfentry(2, 4.001496));
    break;

    // La
  case(57):
    Ev.push_back(hfentry(0, 1372.280305));
    Ev.push_back(hfentry(0, 216.276999));
    Ev.push_back(hfentry(0, 47.844413));
    Ev.push_back(hfentry(0, 10.345537));
    Ev.push_back(hfentry(0, 1.704429));
    Ev.push_back(hfentry(0, 0.170405));
    Ev.push_back(hfentry(1, 203.907402));
    Ev.push_back(hfentry(1, 42.459150));
    Ev.push_back(hfentry(1, 8.258908));
    Ev.push_back(hfentry(1, 1.049440));
    Ev.push_back(hfentry(2, 32.553617));
    Ev.push_back(hfentry(2, 4.553592));
    Ev.push_back(hfentry(2, 0.268850));
    break;

    // Ce
  case(58):
    Ev.push_back(hfentry(0, 1423.102730));
    Ev.push_back(hfentry(0, 225.252823));
    Ev.push_back(hfentry(0, 50.077318));
    Ev.push_back(hfentry(0, 10.807065));
    Ev.push_back(hfentry(0, 1.749455));
    Ev.push_back(hfentry(0, 0.171432));
    Ev.push_back(hfentry(1, 212.618113));
    Ev.push_back(hfentry(1, 44.550339));
    Ev.push_back(hfentry(1, 8.650582));
    Ev.push_back(hfentry(1, 1.073869));
    Ev.push_back(hfentry(2, 34.383960));
    Ev.push_back(hfentry(2, 4.814578));
    Ev.push_back(hfentry(2, 0.297944));
    Ev.push_back(hfentry(3, 0.713190));
    break;

    // Pr
  case(59):
    Ev.push_back(hfentry(0, 1474.559431));
    Ev.push_back(hfentry(0, 234.080284));
    Ev.push_back(hfentry(0, 51.978923));
    Ev.push_back(hfentry(0, 10.945202));
    Ev.push_back(hfentry(0, 1.656296));
    Ev.push_back(hfentry(0, 0.163924));
    Ev.push_back(hfentry(1, 221.186614));
    Ev.push_back(hfentry(1, 46.312714));
    Ev.push_back(hfentry(1, 8.727861));
    Ev.push_back(hfentry(1, 0.983827));
    Ev.push_back(hfentry(2, 35.892497));
    Ev.push_back(hfentry(2, 4.785764));
    Ev.push_back(hfentry(3, 0.549223));
    break;

    // Nd
  case(60):
    Ev.push_back(hfentry(0, 1527.212971));
    Ev.push_back(hfentry(0, 243.384418));
    Ev.push_back(hfentry(0, 54.263196));
    Ev.push_back(hfentry(0, 11.395776));
    Ev.push_back(hfentry(0, 1.698177));
    Ev.push_back(hfentry(0, 0.165783));
    Ev.push_back(hfentry(1, 230.225910));
    Ev.push_back(hfentry(1, 48.454628));
    Ev.push_back(hfentry(1, 9.108511));
    Ev.push_back(hfentry(1, 1.005847));
    Ev.push_back(hfentry(2, 37.772687));
    Ev.push_back(hfentry(2, 5.035966));
    Ev.push_back(hfentry(3, 0.595772));
    break;

    // Pm
  case(61):
    Ev.push_back(hfentry(0, 1580.789685));
    Ev.push_back(hfentry(0, 252.861313));
    Ev.push_back(hfentry(0, 56.583164));
    Ev.push_back(hfentry(0, 11.848184));
    Ev.push_back(hfentry(0, 1.739373));
    Ev.push_back(hfentry(0, 0.167593));
    Ev.push_back(hfentry(1, 239.437914));
    Ev.push_back(hfentry(1, 50.631812));
    Ev.push_back(hfentry(1, 9.490685));
    Ev.push_back(hfentry(1, 1.027256));
    Ev.push_back(hfentry(2, 39.687450));
    Ev.push_back(hfentry(2, 5.287197));
    Ev.push_back(hfentry(3, 0.628301));
    break;

    // Sm
  case(62):
    Ev.push_back(hfentry(0, 1635.284543));
    Ev.push_back(hfentry(0, 262.505469));
    Ev.push_back(hfentry(0, 58.933038));
    Ev.push_back(hfentry(0, 12.298023));
    Ev.push_back(hfentry(0, 1.778680));
    Ev.push_back(hfentry(0, 0.169320));
    Ev.push_back(hfentry(1, 248.817219));
    Ev.push_back(hfentry(1, 52.838500));
    Ev.push_back(hfentry(1, 9.870087));
    Ev.push_back(hfentry(1, 1.047151));
    Ev.push_back(hfentry(2, 41.631055));
    Ev.push_back(hfentry(2, 5.535428));
    Ev.push_back(hfentry(3, 0.666415));
    break;

    // Eu
  case(63):
    Ev.push_back(hfentry(0, 1690.696544));
    Ev.push_back(hfentry(0, 272.315849));
    Ev.push_back(hfentry(0, 61.311967));
    Ev.push_back(hfentry(0, 12.744771));
    Ev.push_back(hfentry(0, 1.816028));
    Ev.push_back(hfentry(0, 0.170964));
    Ev.push_back(hfentry(1, 258.362798));
    Ev.push_back(hfentry(1, 55.073825));
    Ev.push_back(hfentry(1, 10.246171));
    Ev.push_back(hfentry(1, 1.065486));
    Ev.push_back(hfentry(2, 43.602595));
    Ev.push_back(hfentry(2, 5.780046));
    Ev.push_back(hfentry(3, 0.711724));
    break;

    // Gd
  case(64):
    Ev.push_back(hfentry(0, 1747.334818));
    Ev.push_back(hfentry(0, 282.631397));
    Ev.push_back(hfentry(0, 64.098941));
    Ev.push_back(hfentry(0, 13.530882));
    Ev.push_back(hfentry(0, 2.007425));
    Ev.push_back(hfentry(0, 0.183383));
    Ev.push_back(hfentry(1, 268.407828));
    Ev.push_back(hfentry(1, 57.714117));
    Ev.push_back(hfentry(1, 10.953493));
    Ev.push_back(hfentry(1, 1.212481));
    Ev.push_back(hfentry(2, 45.971402));
    Ev.push_back(hfentry(2, 6.334078));
    Ev.push_back(hfentry(2, 0.287278));
    Ev.push_back(hfentry(3, 1.043538));
    break;

    // Tb
  case(65):
    Ev.push_back(hfentry(0, 1804.342738));
    Ev.push_back(hfentry(0, 292.513139));
    Ev.push_back(hfentry(0, 66.242749));
    Ev.push_back(hfentry(0, 13.694011));
    Ev.push_back(hfentry(0, 1.901252));
    Ev.push_back(hfentry(0, 0.174492));
    Ev.push_back(hfentry(1, 278.029306));
    Ev.push_back(hfentry(1, 59.715597));
    Ev.push_back(hfentry(1, 11.051204));
    Ev.push_back(hfentry(1, 1.109516));
    Ev.push_back(hfentry(2, 47.713279));
    Ev.push_back(hfentry(2, 6.315975));
    Ev.push_back(hfentry(3, 0.695407));
    break;

    // Dy
  case(66):
    Ev.push_back(hfentry(0, 1862.544387));
    Ev.push_back(hfentry(0, 302.864302));
    Ev.push_back(hfentry(0, 68.755713));
    Ev.push_back(hfentry(0, 14.166828));
    Ev.push_back(hfentry(0, 1.941461));
    Ev.push_back(hfentry(0, 0.176148));
    Ev.push_back(hfentry(1, 288.115102));
    Ev.push_back(hfentry(1, 62.083368));
    Ev.push_back(hfentry(1, 11.451407));
    Ev.push_back(hfentry(1, 1.129563));
    Ev.push_back(hfentry(2, 49.814272));
    Ev.push_back(hfentry(2, 6.580807));
    Ev.push_back(hfentry(3, 0.703246));
    break;

    // Ho
  case(67):
    Ev.push_back(hfentry(0, 1921.671632));
    Ev.push_back(hfentry(0, 313.391050));
    Ev.push_back(hfentry(0, 71.308449));
    Ev.push_back(hfentry(0, 14.644518));
    Ev.push_back(hfentry(0, 1.981618));
    Ev.push_back(hfentry(0, 0.177781));
    Ev.push_back(hfentry(1, 298.376375));
    Ev.push_back(hfentry(1, 64.490395));
    Ev.push_back(hfentry(1, 11.855921));
    Ev.push_back(hfentry(1, 1.149437));
    Ev.push_back(hfentry(2, 51.953546));
    Ev.push_back(hfentry(2, 6.848877));
    Ev.push_back(hfentry(3, 0.708583));
    break;

    // Er
  case(68):
    Ev.push_back(hfentry(0, 1981.724675));
    Ev.push_back(hfentry(0, 324.093618));
    Ev.push_back(hfentry(0, 73.901311));
    Ev.push_back(hfentry(0, 15.127340));
    Ev.push_back(hfentry(0, 2.021781));
    Ev.push_back(hfentry(0, 0.179392));
    Ev.push_back(hfentry(1, 308.813356));
    Ev.push_back(hfentry(1, 66.937020));
    Ev.push_back(hfentry(1, 12.264984));
    Ev.push_back(hfentry(1, 1.169181));
    Ev.push_back(hfentry(2, 54.131415));
    Ev.push_back(hfentry(2, 7.120374));
    Ev.push_back(hfentry(3, 0.711720));
    break;

    // Tm
  case(69):
    Ev.push_back(hfentry(0, 2042.698792));
    Ev.push_back(hfentry(0, 334.966873));
    Ev.push_back(hfentry(0, 76.528887));
    Ev.push_back(hfentry(0, 15.611268));
    Ev.push_back(hfentry(0, 2.060938));
    Ev.push_back(hfentry(0, 0.180955));
    Ev.push_back(hfentry(1, 319.420993));
    Ev.push_back(hfentry(1, 69.417851));
    Ev.push_back(hfentry(1, 12.674672));
    Ev.push_back(hfentry(1, 1.188049));
    Ev.push_back(hfentry(2, 56.342527));
    Ev.push_back(hfentry(2, 7.391616));
    Ev.push_back(hfentry(3, 0.719265));
    break;

    // Yb
  case(70):
    Ev.push_back(hfentry(0, 2104.592407));
    Ev.push_back(hfentry(0, 346.009135));
    Ev.push_back(hfentry(0, 79.189503));
    Ev.push_back(hfentry(0, 16.095088));
    Ev.push_back(hfentry(0, 2.098804));
    Ev.push_back(hfentry(0, 0.182463));
    Ev.push_back(hfentry(1, 330.197628));
    Ev.push_back(hfentry(1, 71.931210));
    Ev.push_back(hfentry(1, 13.083780));
    Ev.push_back(hfentry(1, 1.205832));
    Ev.push_back(hfentry(2, 58.585193));
    Ev.push_back(hfentry(2, 7.661433));
    Ev.push_back(hfentry(3, 0.732406));
    break;

    // Lu
  case(71):
    Ev.push_back(hfentry(0, 2167.731477));
    Ev.push_back(hfentry(0, 357.572303));
    Ev.push_back(hfentry(0, 82.268768));
    Ev.push_back(hfentry(0, 16.937896));
    Ev.push_back(hfentry(0, 2.317030));
    Ev.push_back(hfentry(0, 0.198856));
    Ev.push_back(hfentry(1, 341.490124));
    Ev.push_back(hfentry(1, 74.860335));
    Ev.push_back(hfentry(1, 13.845079));
    Ev.push_back(hfentry(1, 1.375840));
    Ev.push_back(hfentry(2, 61.236416));
    Ev.push_back(hfentry(2, 8.264905));
    Ev.push_back(hfentry(2, 0.243353));
    Ev.push_back(hfentry(3, 1.076887));
    break;

    // Hf
  case(72):
    Ev.push_back(hfentry(0, 2231.811080));
    Ev.push_back(hfentry(0, 369.329801));
    Ev.push_back(hfentry(0, 85.411971));
    Ev.push_back(hfentry(0, 17.797724));
    Ev.push_back(hfentry(0, 2.519419));
    Ev.push_back(hfentry(0, 0.209002));
    Ev.push_back(hfentry(1, 352.976012));
    Ev.push_back(hfentry(1, 77.852505));
    Ev.push_back(hfentry(1, 14.621995));
    Ev.push_back(hfentry(1, 1.532359));
    Ev.push_back(hfentry(2, 63.948890));
    Ev.push_back(hfentry(2, 8.881065));
    Ev.push_back(hfentry(2, 0.324432));
    Ev.push_back(hfentry(3, 1.430271));
    break;

    // Ta
  case(73):
    Ev.push_back(hfentry(0, 2296.842183));
    Ev.push_back(hfentry(0, 381.292349));
    Ev.push_back(hfentry(0, 88.628813));
    Ev.push_back(hfentry(0, 18.684462));
    Ev.push_back(hfentry(0, 2.720391));
    Ev.push_back(hfentry(0, 0.217490));
    Ev.push_back(hfentry(1, 364.666082));
    Ev.push_back(hfentry(1, 80.917469));
    Ev.push_back(hfentry(1, 15.424583));
    Ev.push_back(hfentry(1, 1.688385));
    Ev.push_back(hfentry(2, 66.732486));
    Ev.push_back(hfentry(2, 9.520499));
    Ev.push_back(hfentry(2, 0.387780));
    Ev.push_back(hfentry(3, 1.804687));
    break;

    // W
  case(74):
    Ev.push_back(hfentry(0, 2362.823830));
    Ev.push_back(hfentry(0, 393.458738));
    Ev.push_back(hfentry(0, 91.917520));
    Ev.push_back(hfentry(0, 19.596328));
    Ev.push_back(hfentry(0, 2.921222));
    Ev.push_back(hfentry(0, 0.224809));
    Ev.push_back(hfentry(1, 376.559177));
    Ev.push_back(hfentry(1, 84.053488));
    Ev.push_back(hfentry(1, 16.251207));
    Ev.push_back(hfentry(1, 1.845081));
    Ev.push_back(hfentry(2, 69.585560));
    Ev.push_back(hfentry(2, 10.181981));
    Ev.push_back(hfentry(2, 0.446353));
    Ev.push_back(hfentry(3, 2.199662));
    break;

    // Re
  case(75):
    Ev.push_back(hfentry(0, 2429.751463));
    Ev.push_back(hfentry(0, 405.824202));
    Ev.push_back(hfentry(0, 95.273022));
    Ev.push_back(hfentry(0, 20.528342));
    Ev.push_back(hfentry(0, 3.119745));
    Ev.push_back(hfentry(0, 0.230673));
    Ev.push_back(hfentry(1, 388.650570));
    Ev.push_back(hfentry(1, 87.255518));
    Ev.push_back(hfentry(1, 17.097020));
    Ev.push_back(hfentry(1, 2.000529));
    Ev.push_back(hfentry(2, 72.503134));
    Ev.push_back(hfentry(2, 10.861012));
    Ev.push_back(hfentry(2, 0.514322));
    Ev.push_back(hfentry(3, 2.611121));
    break;

    // Os
  case(76):
    Ev.push_back(hfentry(0, 2497.646279));
    Ev.push_back(hfentry(0, 418.409954));
    Ev.push_back(hfentry(0, 98.715591));
    Ev.push_back(hfentry(0, 21.500388));
    Ev.push_back(hfentry(0, 3.333636));
    Ev.push_back(hfentry(0, 0.238767));
    Ev.push_back(hfentry(1, 400.961506));
    Ev.push_back(hfentry(1, 90.543869));
    Ev.push_back(hfentry(1, 17.981903));
    Ev.push_back(hfentry(1, 2.170386));
    Ev.push_back(hfentry(2, 75.505602));
    Ev.push_back(hfentry(2, 11.577449));
    Ev.push_back(hfentry(2, 0.535781));
    Ev.push_back(hfentry(3, 3.059248));
    break;

    // Ir
  case(77):
    Ev.push_back(hfentry(0, 2566.484813));
    Ev.push_back(hfentry(0, 431.192234));
    Ev.push_back(hfentry(0, 102.221814));
    Ev.push_back(hfentry(0, 22.489451));
    Ev.push_back(hfentry(0, 3.544825));
    Ev.push_back(hfentry(0, 0.245373));
    Ev.push_back(hfentry(1, 413.468249));
    Ev.push_back(hfentry(1, 93.895134));
    Ev.push_back(hfentry(1, 18.883022));
    Ev.push_back(hfentry(1, 2.338628));
    Ev.push_back(hfentry(2, 78.569583));
    Ev.push_back(hfentry(2, 12.308903));
    Ev.push_back(hfentry(2, 0.574001));
    Ev.push_back(hfentry(3, 3.521744));
    break;

    // Pt
  case(78):
    Ev.push_back(hfentry(0, 2636.112231));
    Ev.push_back(hfentry(0, 444.012794));
    Ev.push_back(hfentry(0, 105.633525));
    Ev.push_back(hfentry(0, 23.336611));
    Ev.push_back(hfentry(0, 3.605347));
    Ev.push_back(hfentry(0, 0.222830));
    Ev.push_back(hfentry(1, 426.013215));
    Ev.push_back(hfentry(1, 97.151661));
    Ev.push_back(hfentry(1, 19.642774));
    Ev.push_back(hfentry(1, 2.371651));
    Ev.push_back(hfentry(2, 81.537857));
    Ev.push_back(hfentry(2, 12.899237));
    Ev.push_back(hfentry(2, 0.475069));
    Ev.push_back(hfentry(3, 3.842165));
    break;

    // Au
  case(79):
    Ev.push_back(hfentry(0, 2706.832984));
    Ev.push_back(hfentry(0, 457.182524));
    Ev.push_back(hfentry(0, 109.260794));
    Ev.push_back(hfentry(0, 24.353765));
    Ev.push_back(hfentry(0, 3.808827));
    Ev.push_back(hfentry(0, 0.220778));
    Ev.push_back(hfentry(1, 438.906054));
    Ev.push_back(hfentry(1, 100.622567));
    Ev.push_back(hfentry(1, 20.570661));
    Ev.push_back(hfentry(1, 2.534336));
    Ev.push_back(hfentry(2, 84.718799));
    Ev.push_back(hfentry(2, 13.655497));
    Ev.push_back(hfentry(2, 0.521023));
    Ev.push_back(hfentry(3, 4.328347));
    break;

    // Hg
  case(80):
    Ev.push_back(hfentry(0, 2778.680221));
    Ev.push_back(hfentry(0, 470.735051));
    Ev.push_back(hfentry(0, 113.136600));
    Ev.push_back(hfentry(0, 25.573382));
    Ev.push_back(hfentry(0, 4.182017));
    Ev.push_back(hfentry(0, 0.261046));
    Ev.push_back(hfentry(1, 452.180310));
    Ev.push_back(hfentry(1, 104.340772));
    Ev.push_back(hfentry(1, 21.698933));
    Ev.push_back(hfentry(1, 2.850880));
    Ev.push_back(hfentry(2, 88.145357));
    Ev.push_back(hfentry(2, 14.609618));
    Ev.push_back(hfentry(2, 0.714203));
    Ev.push_back(hfentry(3, 5.012380));
    break;

    // Tl
  case(81):
    Ev.push_back(hfentry(0, 2851.547207));
    Ev.push_back(hfentry(0, 484.560177));
    Ev.push_back(hfentry(0, 117.150470));
    Ev.push_back(hfentry(0, 26.883439));
    Ev.push_back(hfentry(0, 4.618775));
    Ev.push_back(hfentry(0, 0.361112));
    Ev.push_back(hfentry(1, 465.726727));
    Ev.push_back(hfentry(1, 108.196801));
    Ev.push_back(hfentry(1, 22.917694));
    Ev.push_back(hfentry(1, 3.231369));
    Ev.push_back(hfentry(1, 0.192397));
    Ev.push_back(hfentry(2, 91.708496));
    Ev.push_back(hfentry(2, 15.652961));
    Ev.push_back(hfentry(2, 0.968275));
    Ev.push_back(hfentry(3, 5.785240));
    break;

    // Pb
  case(82):
    Ev.push_back(hfentry(0, 2925.369754));
    Ev.push_back(hfentry(0, 498.594155));
    Ev.push_back(hfentry(0, 121.238373));
    Ev.push_back(hfentry(0, 28.219947));
    Ev.push_back(hfentry(0, 5.055544));
    Ev.push_back(hfentry(0, 0.456226));
    Ev.push_back(hfentry(1, 479.481368));
    Ev.push_back(hfentry(1, 112.126368));
    Ev.push_back(hfentry(1, 24.162410));
    Ev.push_back(hfentry(1, 3.610692));
    Ev.push_back(hfentry(1, 0.255697));
    Ev.push_back(hfentry(2, 95.343943));
    Ev.push_back(hfentry(2, 16.720911));
    Ev.push_back(hfentry(2, 1.220429));
    Ev.push_back(hfentry(3, 6.582431));
    break;

    // Bi
  case(83):
    Ev.push_back(hfentry(0, 3000.152768));
    Ev.push_back(hfentry(0, 512.841770));
    Ev.push_back(hfentry(0, 125.404955));
    Ev.push_back(hfentry(0, 29.587497));
    Ev.push_back(hfentry(0, 5.497783));
    Ev.push_back(hfentry(0, 0.551690));
    Ev.push_back(hfentry(1, 493.449027));
    Ev.push_back(hfentry(1, 116.134115));
    Ev.push_back(hfentry(1, 25.437673));
    Ev.push_back(hfentry(1, 3.994714));
    Ev.push_back(hfentry(1, 0.320105));
    Ev.push_back(hfentry(2, 99.056387));
    Ev.push_back(hfentry(2, 17.818176));
    Ev.push_back(hfentry(2, 1.477157));
    Ev.push_back(hfentry(3, 7.408698));
    break;

    // Po
  case(84):
    Ev.push_back(hfentry(0, 3075.913471));
    Ev.push_back(hfentry(0, 527.319921));
    Ev.push_back(hfentry(0, 129.666880));
    Ev.push_back(hfentry(0, 31.002593));
    Ev.push_back(hfentry(0, 5.962098));
    Ev.push_back(hfentry(0, 0.658255));
    Ev.push_back(hfentry(1, 507.646630));
    Ev.push_back(hfentry(1, 120.236689));
    Ev.push_back(hfentry(1, 26.759958));
    Ev.push_back(hfentry(1, 4.400146));
    Ev.push_back(hfentry(1, 0.341467));
    Ev.push_back(hfentry(2, 102.862560));
    Ev.push_back(hfentry(2, 18.961385));
    Ev.push_back(hfentry(2, 1.755303));
    Ev.push_back(hfentry(3, 8.280689));
    break;

    // At
  case(85):
    Ev.push_back(hfentry(0, 3152.633219));
    Ev.push_back(hfentry(0, 542.010114));
    Ev.push_back(hfentry(0, 134.005628));
    Ev.push_back(hfentry(0, 32.446784));
    Ev.push_back(hfentry(0, 6.431043));
    Ev.push_back(hfentry(0, 0.765378));
    Ev.push_back(hfentry(1, 522.055686));
    Ev.push_back(hfentry(1, 124.415610));
    Ev.push_back(hfentry(1, 28.110895));
    Ev.push_back(hfentry(1, 4.809864));
    Ev.push_back(hfentry(1, 0.379865));
    Ev.push_back(hfentry(2, 106.743978));
    Ev.push_back(hfentry(2, 20.132178));
    Ev.push_back(hfentry(2, 2.037918));
    Ev.push_back(hfentry(3, 9.180077));
    break;

    // Rn
  case(86):
    Ev.push_back(hfentry(0, 3230.312838));
    Ev.push_back(hfentry(0, 556.913115));
    Ev.push_back(hfentry(0, 138.421866));
    Ev.push_back(hfentry(0, 33.920746));
    Ev.push_back(hfentry(0, 6.905818));
    Ev.push_back(hfentry(0, 0.873993));
    Ev.push_back(hfentry(1, 536.676971));
    Ev.push_back(hfentry(1, 128.671558));
    Ev.push_back(hfentry(1, 29.491183));
    Ev.push_back(hfentry(1, 5.225212));
    Ev.push_back(hfentry(1, 0.428007));
    Ev.push_back(hfentry(2, 110.701350));
    Ev.push_back(hfentry(2, 21.331318));
    Ev.push_back(hfentry(2, 2.326319));
    Ev.push_back(hfentry(3, 10.107635));
    break;

    // Fr
  case(87):
    Ev.push_back(hfentry(0, 3309.144859));
    Ev.push_back(hfentry(0, 572.220473));
    Ev.push_back(hfentry(0, 143.106572));
    Ev.push_back(hfentry(0, 35.615036));
    Ev.push_back(hfentry(0, 7.575756));
    Ev.push_back(hfentry(0, 1.127303));
    Ev.push_back(hfentry(0, 0.117912));
    Ev.push_back(hfentry(1, 551.701970));
    Ev.push_back(hfentry(1, 133.195189));
    Ev.push_back(hfentry(1, 31.090810));
    Ev.push_back(hfentry(1, 5.834205));
    Ev.push_back(hfentry(1, 0.628530));
    Ev.push_back(hfentry(2, 114.925563));
    Ev.push_back(hfentry(2, 22.749141));
    Ev.push_back(hfentry(2, 2.808270));
    Ev.push_back(hfentry(3, 11.253710));
    break;

    // Ra
  case(88):
    Ev.push_back(hfentry(0, 3388.941086));
    Ev.push_back(hfentry(0, 587.744241));
    Ev.push_back(hfentry(0, 147.871751));
    Ev.push_back(hfentry(0, 37.341614));
    Ev.push_back(hfentry(0, 8.253202));
    Ev.push_back(hfentry(0, 1.370703));
    Ev.push_back(hfentry(0, 0.148771));
    Ev.push_back(hfentry(1, 566.942804));
    Ev.push_back(hfentry(1, 137.798684));
    Ev.push_back(hfentry(1, 32.722041));
    Ev.push_back(hfentry(1, 6.449906));
    Ev.push_back(hfentry(1, 0.819838));
    Ev.push_back(hfentry(2, 119.228767));
    Ev.push_back(hfentry(2, 24.197831));
    Ev.push_back(hfentry(2, 3.296896));
    Ev.push_back(hfentry(3, 12.430525));
    break;

    // Ac
  case(89):
    Ev.push_back(hfentry(0, 3469.626572));
    Ev.push_back(hfentry(0, 603.407615));
    Ev.push_back(hfentry(0, 152.640817));
    Ev.push_back(hfentry(0, 39.023932));
    Ev.push_back(hfentry(0, 8.862401));
    Ev.push_back(hfentry(0, 1.535632));
    Ev.push_back(hfentry(0, 0.161013));
    Ev.push_back(hfentry(1, 582.323027));
    Ev.push_back(hfentry(1, 142.405714));
    Ev.push_back(hfentry(1, 34.308938));
    Ev.push_back(hfentry(1, 6.998077));
    Ev.push_back(hfentry(1, 0.945999));
    Ev.push_back(hfentry(2, 123.534804));
    Ev.push_back(hfentry(2, 25.602107));
    Ev.push_back(hfentry(2, 3.718988));
    Ev.push_back(hfentry(2, 0.251492));
    Ev.push_back(hfentry(3, 13.562616));
    break;

    // Th
  case(90):
    Ev.push_back(hfentry(0, 3551.263977));
    Ev.push_back(hfentry(0, 619.274498));
    Ev.push_back(hfentry(0, 157.477657));
    Ev.push_back(hfentry(0, 40.726162));
    Ev.push_back(hfentry(0, 9.468353));
    Ev.push_back(hfentry(0, 1.690361));
    Ev.push_back(hfentry(0, 0.169989));
    Ev.push_back(hfentry(1, 597.906324));
    Ev.push_back(hfentry(1, 147.080042));
    Ev.push_back(hfentry(1, 35.915381));
    Ev.push_back(hfentry(1, 7.542955));
    Ev.push_back(hfentry(1, 1.065187));
    Ev.push_back(hfentry(2, 127.907309));
    Ev.push_back(hfentry(2, 27.025469));
    Ev.push_back(hfentry(2, 4.138488));
    Ev.push_back(hfentry(2, 0.316901));
    Ev.push_back(hfentry(3, 14.713587));
    break;

    // Pa
  case(91):
    Ev.push_back(hfentry(0, 3633.460561));
    Ev.push_back(hfentry(0, 634.924072));
    Ev.push_back(hfentry(0, 161.935825));
    Ev.push_back(hfentry(0, 42.011351));
    Ev.push_back(hfentry(0, 9.666735));
    Ev.push_back(hfentry(0, 1.633846));
    Ev.push_back(hfentry(0, 0.163733));
    Ev.push_back(hfentry(1, 613.277519));
    Ev.push_back(hfentry(1, 151.376996));
    Ev.push_back(hfentry(1, 37.104790));
    Ev.push_back(hfentry(1, 7.693145));
    Ev.push_back(hfentry(1, 1.005681));
    Ev.push_back(hfentry(2, 131.905677));
    Ev.push_back(hfentry(2, 28.031813));
    Ev.push_back(hfentry(2, 4.203950));
    Ev.push_back(hfentry(2, 0.291102));
    Ev.push_back(hfentry(3, 15.451392));
    Ev.push_back(hfentry(3, 0.611767));
    break;

    // U
  case(92):
    Ev.push_back(hfentry(0, 3716.781476));
    Ev.push_back(hfentry(0, 650.960245));
    Ev.push_back(hfentry(0, 166.654983));
    Ev.push_back(hfentry(0, 43.508322));
    Ev.push_back(hfentry(0, 10.049791));
    Ev.push_back(hfentry(0, 1.674926));
    Ev.push_back(hfentry(0, 0.165176));
    Ev.push_back(hfentry(1, 629.032775));
    Ev.push_back(hfentry(1, 155.933735));
    Ev.push_back(hfentry(1, 38.505419));
    Ev.push_back(hfentry(1, 8.022882));
    Ev.push_back(hfentry(1, 1.029528));
    Ev.push_back(hfentry(2, 136.161260));
    Ev.push_back(hfentry(2, 29.248497));
    Ev.push_back(hfentry(2, 4.432602));
    Ev.push_back(hfentry(2, 0.300379));
    Ev.push_back(hfentry(3, 16.397546));
    Ev.push_back(hfentry(3, 0.701559));
    break;

    // Np
  case(93):
    Ev.push_back(hfentry(0, 3801.047108));
    Ev.push_back(hfentry(0, 667.191187));
    Ev.push_back(hfentry(0, 171.432036));
    Ev.push_back(hfentry(0, 45.017315));
    Ev.push_back(hfentry(0, 10.429141));
    Ev.push_back(hfentry(0, 1.713918));
    Ev.push_back(hfentry(0, 0.166567));
    Ev.push_back(hfentry(1, 644.982706));
    Ev.push_back(hfentry(1, 160.547961));
    Ev.push_back(hfentry(1, 39.917702));
    Ev.push_back(hfentry(1, 8.349355));
    Ev.push_back(hfentry(1, 1.051827));
    Ev.push_back(hfentry(2, 140.473594));
    Ev.push_back(hfentry(2, 30.476245));
    Ev.push_back(hfentry(2, 4.659317));
    Ev.push_back(hfentry(2, 0.309935));
    Ev.push_back(hfentry(3, 17.354532));
    Ev.push_back(hfentry(3, 0.770080));
    break;

    // Pu
  case(94):
    Ev.push_back(hfentry(0, 3886.048279));
    Ev.push_back(hfentry(0, 683.394155));
    Ev.push_back(hfentry(0, 176.032343));
    Ev.push_back(hfentry(0, 46.307145));
    Ev.push_back(hfentry(0, 10.587780));
    Ev.push_back(hfentry(0, 1.636153));
    Ev.push_back(hfentry(0, 0.159929));
    Ev.push_back(hfentry(1, 660.907270));
    Ev.push_back(hfentry(1, 164.985869));
    Ev.push_back(hfentry(1, 41.110645));
    Ev.push_back(hfentry(1, 8.461448));
    Ev.push_back(hfentry(1, 0.975678));
    Ev.push_back(hfentry(2, 144.610821));
    Ev.push_back(hfentry(2, 31.484373));
    Ev.push_back(hfentry(2, 4.691281));
    Ev.push_back(hfentry(3, 18.093399));
    Ev.push_back(hfentry(3, 0.654375));
    break;

    // Am
  case(95):
    Ev.push_back(hfentry(0, 3972.193926));
    Ev.push_back(hfentry(0, 700.004792));
    Ev.push_back(hfentry(0, 180.915313));
    Ev.push_back(hfentry(0, 47.830984));
    Ev.push_back(hfentry(0, 10.952653));
    Ev.push_back(hfentry(0, 1.668161));
    Ev.push_back(hfentry(0, 0.161309));
    Ev.push_back(hfentry(1, 677.236792));
    Ev.push_back(hfentry(1, 169.705203));
    Ev.push_back(hfentry(1, 42.537061));
    Ev.push_back(hfentry(1, 8.774365));
    Ev.push_back(hfentry(1, 0.992523));
    Ev.push_back(hfentry(2, 149.026799));
    Ev.push_back(hfentry(2, 32.725076));
    Ev.push_back(hfentry(2, 4.907078));
    Ev.push_back(hfentry(3, 19.062694));
    Ev.push_back(hfentry(3, 0.722194));
    break;

    // Cm
  case(96):
    Ev.push_back(hfentry(0, 4059.515709));
    Ev.push_back(hfentry(0, 717.055916));
    Ev.push_back(hfentry(0, 186.114113));
    Ev.push_back(hfentry(0, 49.621354));
    Ev.push_back(hfentry(0, 11.552378));
    Ev.push_back(hfentry(0, 1.824316));
    Ev.push_back(hfentry(0, 0.171577));
    Ev.push_back(hfentry(1, 694.003880));
    Ev.push_back(hfentry(1, 174.739115));
    Ev.push_back(hfentry(1, 44.229462));
    Ev.push_back(hfentry(1, 9.316185));
    Ev.push_back(hfentry(1, 1.114342));
    Ev.push_back(hfentry(2, 153.754620));
    Ev.push_back(hfentry(2, 34.230821));
    Ev.push_back(hfentry(2, 5.333598));
    Ev.push_back(hfentry(2, 0.300695));
    Ev.push_back(hfentry(3, 20.294897));
    Ev.push_back(hfentry(3, 0.968851));
    break;

    // Bk
  case(97):
    Ev.push_back(hfentry(0, 4147.580550));
    Ev.push_back(hfentry(0, 734.087216));
    Ev.push_back(hfentry(0, 191.144846));
    Ev.push_back(hfentry(0, 51.200532));
    Ev.push_back(hfentry(0, 11.938003));
    Ev.push_back(hfentry(0, 1.865158));
    Ev.push_back(hfentry(0, 0.173886));
    Ev.push_back(hfentry(1, 710.753631));
    Ev.push_back(hfentry(1, 179.605363));
    Ev.push_back(hfentry(1, 45.710503));
    Ev.push_back(hfentry(1, 9.649728));
    Ev.push_back(hfentry(1, 1.138552));
    Ev.push_back(hfentry(2, 158.315920));
    Ev.push_back(hfentry(2, 35.524917));
    Ev.push_back(hfentry(2, 5.570008));
    Ev.push_back(hfentry(2, 0.283443));
    Ev.push_back(hfentry(3, 21.316714));
    Ev.push_back(hfentry(3, 0.988592));
    break;

    // Cf
  case(98):
    Ev.push_back(hfentry(0, 4236.351589));
    Ev.push_back(hfentry(0, 751.061089));
    Ev.push_back(hfentry(0, 195.970133));
    Ev.push_back(hfentry(0, 52.531850));
    Ev.push_back(hfentry(0, 12.076210));
    Ev.push_back(hfentry(0, 1.769886));
    Ev.push_back(hfentry(0, 0.165547));
    Ev.push_back(hfentry(1, 727.448558));
    Ev.push_back(hfentry(1, 184.266562));
    Ev.push_back(hfentry(1, 46.943550));
    Ev.push_back(hfentry(1, 9.741983));
    Ev.push_back(hfentry(1, 1.047447));
    Ev.push_back(hfentry(2, 162.673303));
    Ev.push_back(hfentry(2, 36.570823));
    Ev.push_back(hfentry(2, 5.583823));
    Ev.push_back(hfentry(3, 22.091422));
    Ev.push_back(hfentry(3, 0.806768));
    break;

    // Es
  case(99):
    Ev.push_back(hfentry(0, 4326.294960));
    Ev.push_back(hfentry(0, 768.470955));
    Ev.push_back(hfentry(0, 201.106192));
    Ev.push_back(hfentry(0, 54.125634));
    Ev.push_back(hfentry(0, 12.446889));
    Ev.push_back(hfentry(0, 1.800654));
    Ev.push_back(hfentry(0, 0.166842));
    Ev.push_back(hfentry(1, 744.576698));
    Ev.push_back(hfentry(1, 189.237321));
    Ev.push_back(hfentry(1, 48.438531));
    Ev.push_back(hfentry(1, 10.061071));
    Ev.push_back(hfentry(1, 1.063364));
    Ev.push_back(hfentry(2, 167.337593));
    Ev.push_back(hfentry(2, 37.877709));
    Ev.push_back(hfentry(2, 5.807280));
    Ev.push_back(hfentry(3, 23.124979));
    Ev.push_back(hfentry(3, 0.843070));
    break;

    // Fm
  case(100):
    Ev.push_back(hfentry(0, 4417.185990));
    Ev.push_back(hfentry(0, 786.078755));
    Ev.push_back(hfentry(0, 206.303632));
    Ev.push_back(hfentry(0, 55.735276));
    Ev.push_back(hfentry(0, 12.817797));
    Ev.push_back(hfentry(0, 1.830638));
    Ev.push_back(hfentry(0, 0.168104));
    Ev.push_back(hfentry(1, 761.902641));
    Ev.push_back(hfentry(1, 194.269050));
    Ev.push_back(hfentry(1, 49.949034));
    Ev.push_back(hfentry(1, 10.380470));
    Ev.push_back(hfentry(1, 1.078675));
    Ev.push_back(hfentry(2, 172.062084));
    Ev.push_back(hfentry(2, 39.199520));
    Ev.push_back(hfentry(2, 6.031402));
    Ev.push_back(hfentry(3, 24.172891));
    Ev.push_back(hfentry(3, 0.878134));
    break;

    // Md
  case(101):
    Ev.push_back(hfentry(0, 4509.022098));
    Ev.push_back(hfentry(0, 803.881765));
    Ev.push_back(hfentry(0, 211.559685));
    Ev.push_back(hfentry(0, 57.358286));
    Ev.push_back(hfentry(0, 13.187003));
    Ev.push_back(hfentry(0, 1.859205));
    Ev.push_back(hfentry(0, 0.169317));
    Ev.push_back(hfentry(1, 779.423690));
    Ev.push_back(hfentry(1, 199.358987));
    Ev.push_back(hfentry(1, 51.472575));
    Ev.push_back(hfentry(1, 10.698300));
    Ev.push_back(hfentry(1, 1.092903));
    Ev.push_back(hfentry(2, 176.844018));
    Ev.push_back(hfentry(2, 40.533767));
    Ev.push_back(hfentry(2, 6.254452));
    Ev.push_back(hfentry(3, 25.232636));
    Ev.push_back(hfentry(3, 0.917868));
    break;

    // No
  case(102):
    Ev.push_back(hfentry(0, 4601.802521));
    Ev.push_back(hfentry(0, 821.879184));
    Ev.push_back(hfentry(0, 216.873560));
    Ev.push_back(hfentry(0, 58.994019));
    Ev.push_back(hfentry(0, 13.554080));
    Ev.push_back(hfentry(0, 1.886259));
    Ev.push_back(hfentry(0, 0.170482));
    Ev.push_back(hfentry(1, 797.139052));
    Ev.push_back(hfentry(1, 204.506342));
    Ev.push_back(hfentry(1, 53.008505));
    Ev.push_back(hfentry(1, 11.014131));
    Ev.push_back(hfentry(1, 1.105981));
    Ev.push_back(hfentry(2, 181.682604));
    Ev.push_back(hfentry(2, 41.879795));
    Ev.push_back(hfentry(2, 6.475998));
    Ev.push_back(hfentry(3, 26.303590));
    Ev.push_back(hfentry(3, 0.963492));
    break;

    // Lr
  case(103):
    Ev.push_back(hfentry(0, 4695.804658));
    Ev.push_back(hfentry(0, 840.363165));
    Ev.push_back(hfentry(0, 222.548914));
    Ev.push_back(hfentry(0, 60.942845));
    Ev.push_back(hfentry(0, 14.199976));
    Ev.push_back(hfentry(0, 2.064767));
    Ev.push_back(hfentry(0, 0.183297));
    Ev.push_back(hfentry(1, 815.337993));
    Ev.push_back(hfentry(1, 210.013960));
    Ev.push_back(hfentry(1, 54.856988));
    Ev.push_back(hfentry(1, 11.602624));
    Ev.push_back(hfentry(1, 1.247405));
    Ev.push_back(hfentry(2, 186.878762));
    Ev.push_back(hfentry(2, 43.537332));
    Ev.push_back(hfentry(2, 6.951918));
    Ev.push_back(hfentry(2, 0.267398));
    Ev.push_back(hfentry(3, 27.683911));
    Ev.push_back(hfentry(3, 1.248362));
    break;

    // Rf
  case(104):
    Ev.push_back(hfentry(0, 4790.761257));
    Ev.push_back(hfentry(0, 859.053569));
    Ev.push_back(hfentry(0, 228.295547));
    Ev.push_back(hfentry(0, 62.916090));
    Ev.push_back(hfentry(0, 14.848505));
    Ev.push_back(hfentry(0, 2.230989));
    Ev.push_back(hfentry(0, 0.192030));
    Ev.push_back(hfentry(1, 833.742853));
    Ev.push_back(hfentry(1, 215.592378));
    Ev.push_back(hfentry(1, 56.729599));
    Ev.push_back(hfentry(1, 12.193057));
    Ev.push_back(hfentry(1, 1.378908));
    Ev.push_back(hfentry(2, 192.144824));
    Ev.push_back(hfentry(2, 45.218587));
    Ev.push_back(hfentry(2, 7.428230));
    Ev.push_back(hfentry(2, 0.341986));
    Ev.push_back(hfentry(3, 29.087197));
    Ev.push_back(hfentry(3, 1.532670));
    break;

    // Db
  case(105):
    Ev.push_back(hfentry(0, 4886.680834));
    Ev.push_back(hfentry(0, 877.958854));
    Ev.push_back(hfentry(0, 234.121659));
    Ev.push_back(hfentry(0, 64.921953));
    Ev.push_back(hfentry(0, 15.507954));
    Ev.push_back(hfentry(0, 2.394741));
    Ev.push_back(hfentry(0, 0.199440));
    Ev.push_back(hfentry(1, 852.362114));
    Ev.push_back(hfentry(1, 221.249812));
    Ev.push_back(hfentry(1, 58.634527));
    Ev.push_back(hfentry(1, 12.793788));
    Ev.push_back(hfentry(1, 1.509113));
    Ev.push_back(hfentry(2, 197.489036));
    Ev.push_back(hfentry(2, 46.931707));
    Ev.push_back(hfentry(2, 7.913593));
    Ev.push_back(hfentry(2, 0.400170));
    Ev.push_back(hfentry(3, 30.521648));
    Ev.push_back(hfentry(3, 1.825614));
    break;

    // Sg
  case(106):
    Ev.push_back(hfentry(0, 4983.563644));
    Ev.push_back(hfentry(0, 897.079200));
    Ev.push_back(hfentry(0, 240.027297));
    Ev.push_back(hfentry(0, 66.960473));
    Ev.push_back(hfentry(0, 16.178397));
    Ev.push_back(hfentry(0, 2.557374));
    Ev.push_back(hfentry(0, 0.205899));
    Ev.push_back(hfentry(1, 871.195973));
    Ev.push_back(hfentry(1, 226.986321));
    Ev.push_back(hfentry(1, 60.571812));
    Ev.push_back(hfentry(1, 13.404960));
    Ev.push_back(hfentry(1, 1.639159));
    Ev.push_back(hfentry(2, 202.911483));
    Ev.push_back(hfentry(2, 48.676733));
    Ev.push_back(hfentry(2, 8.408375));
    Ev.push_back(hfentry(2, 0.453946));
    Ev.push_back(hfentry(3, 31.987330));
    Ev.push_back(hfentry(3, 2.127830));
    break;

    // Bh
  case(107):
    Ev.push_back(hfentry(0, 5081.406844));
    Ev.push_back(hfentry(0, 916.411703));
    Ev.push_back(hfentry(0, 246.009513));
    Ev.push_back(hfentry(0, 69.028728));
    Ev.push_back(hfentry(0, 16.857026));
    Ev.push_back(hfentry(0, 2.717509));
    Ev.push_back(hfentry(0, 0.211181));
    Ev.push_back(hfentry(1, 890.241537));
    Ev.push_back(hfentry(1, 232.798963));
    Ev.push_back(hfentry(1, 62.538541));
    Ev.push_back(hfentry(1, 14.023840));
    Ev.push_back(hfentry(1, 1.767792));
    Ev.push_back(hfentry(2, 208.409234));
    Ev.push_back(hfentry(2, 50.450767));
    Ev.push_back(hfentry(2, 8.910044));
    Ev.push_back(hfentry(2, 0.516506));
    Ev.push_back(hfentry(3, 33.481347));
    Ev.push_back(hfentry(3, 2.436949));
    break;

    // Hs
  case(108):
    Ev.push_back(hfentry(0, 5180.227370));
    Ev.push_back(hfentry(0, 935.973286));
    Ev.push_back(hfentry(0, 252.084838));
    Ev.push_back(hfentry(0, 71.143071));
    Ev.push_back(hfentry(0, 17.559799));
    Ev.push_back(hfentry(0, 2.888838));
    Ev.push_back(hfentry(0, 0.218225));
    Ev.push_back(hfentry(1, 909.515753));
    Ev.push_back(hfentry(1, 238.704293));
    Ev.push_back(hfentry(1, 64.551041));
    Ev.push_back(hfentry(1, 14.666337));
    Ev.push_back(hfentry(1, 1.907031));
    Ev.push_back(hfentry(2, 213.998892));
    Ev.push_back(hfentry(2, 52.270064));
    Ev.push_back(hfentry(2, 9.434450));
    Ev.push_back(hfentry(2, 0.534843));
    Ev.push_back(hfentry(3, 35.020049));
    Ev.push_back(hfentry(3, 2.768813));
    break;

    // Mt
  case(109):
    Ev.push_back(hfentry(0, 5280.007240));
    Ev.push_back(hfentry(0, 955.745884));
    Ev.push_back(hfentry(0, 258.235442));
    Ev.push_back(hfentry(0, 73.285831));
    Ev.push_back(hfentry(0, 18.269481));
    Ev.push_back(hfentry(0, 3.057598));
    Ev.push_back(hfentry(0, 0.224082));
    Ev.push_back(hfentry(1, 929.000552));
    Ev.push_back(hfentry(1, 244.684474));
    Ev.push_back(hfentry(1, 66.591674));
    Ev.push_back(hfentry(1, 15.315353));
    Ev.push_back(hfentry(1, 2.044736));
    Ev.push_back(hfentry(2, 219.662607));
    Ev.push_back(hfentry(2, 54.117073));
    Ev.push_back(hfentry(2, 9.964785));
    Ev.push_back(hfentry(2, 0.569105));
    Ev.push_back(hfentry(3, 36.585816));
    Ev.push_back(hfentry(3, 3.106745));
    break;

    // Ds
  case(110):
    Ev.push_back(hfentry(0, 5380.604208));
    Ev.push_back(hfentry(0, 975.585459));
    Ev.push_back(hfentry(0, 264.317778));
    Ev.push_back(hfentry(0, 75.313381));
    Ev.push_back(hfentry(0, 18.841985));
    Ev.push_back(hfentry(0, 3.089990));
    Ev.push_back(hfentry(0, 0.202432));
    Ev.push_back(hfentry(1, 948.552228));
    Ev.push_back(hfentry(1, 250.596178));
    Ev.push_back(hfentry(1, 68.517327));
    Ev.push_back(hfentry(1, 15.827926));
    Ev.push_back(hfentry(1, 2.060692));
    Ev.push_back(hfentry(2, 225.257152));
    Ev.push_back(hfentry(2, 55.849279));
    Ev.push_back(hfentry(2, 10.359073));
    Ev.push_back(hfentry(2, 0.480983));
    Ev.push_back(hfentry(3, 38.035935));
    Ev.push_back(hfentry(3, 3.308616));
    break;

    // Rg
  case(111):
    Ev.push_back(hfentry(0, 5482.299830));
    Ev.push_back(hfentry(0, 995.776977));
    Ev.push_back(hfentry(0, 270.615664));
    Ev.push_back(hfentry(0, 77.509748));
    Ev.push_back(hfentry(0, 19.562481));
    Ev.push_back(hfentry(0, 3.252602));
    Ev.push_back(hfentry(0, 0.201006));
    Ev.push_back(hfentry(1, 968.455127));
    Ev.push_back(hfentry(1, 256.722820));
    Ev.push_back(hfentry(1, 70.611031));
    Ev.push_back(hfentry(1, 16.487131));
    Ev.push_back(hfentry(1, 2.194268));
    Ev.push_back(hfentry(2, 231.065778));
    Ev.push_back(hfentry(2, 57.748535));
    Ev.push_back(hfentry(2, 10.898725));
    Ev.push_back(hfentry(2, 0.524288));
    Ev.push_back(hfentry(3, 39.652685));
    Ev.push_back(hfentry(3, 3.656138));
    break;

    // Cn
  case(112):
    Ev.push_back(hfentry(0, 5585.118173));
    Ev.push_back(hfentry(0, 1016.344743));
    Ev.push_back(hfentry(0, 277.153044));
    Ev.push_back(hfentry(0, 79.898632));
    Ev.push_back(hfentry(0, 20.454185));
    Ev.push_back(hfentry(0, 3.563905));
    Ev.push_back(hfentry(0, 0.238168));
    Ev.push_back(hfentry(1, 988.733509));
    Ev.push_back(hfentry(1, 263.088318));
    Ev.push_back(hfentry(1, 72.896386));
    Ev.push_back(hfentry(1, 17.315951));
    Ev.push_back(hfentry(1, 2.461834));
    Ev.push_back(hfentry(2, 237.112440));
    Ev.push_back(hfentry(2, 59.838343));
    Ev.push_back(hfentry(2, 11.606420));
    Ev.push_back(hfentry(2, 0.696094));
    Ev.push_back(hfentry(3, 41.459662));
    Ev.push_back(hfentry(3, 4.171938));
    break;

    // Nh
  case(113):
    Ev.push_back(hfentry(0, 5688.953835));
    Ev.push_back(hfentry(0, 1037.181523));
    Ev.push_back(hfentry(0, 283.823034));
    Ev.push_back(hfentry(0, 82.372799));
    Ev.push_back(hfentry(0, 21.408723));
    Ev.push_back(hfentry(0, 3.922116));
    Ev.push_back(hfentry(0, 0.326582));
    Ev.push_back(hfentry(1, 1009.280640));
    Ev.push_back(hfentry(1, 269.586265));
    Ev.push_back(hfentry(1, 75.267059));
    Ev.push_back(hfentry(1, 18.207827));
    Ev.push_back(hfentry(1, 2.776992));
    Ev.push_back(hfentry(1, 0.184086));
    Ev.push_back(hfentry(2, 243.290836));
    Ev.push_back(hfentry(2, 62.012936));
    Ev.push_back(hfentry(2, 12.376408));
    Ev.push_back(hfentry(2, 0.913580));
    Ev.push_back(hfentry(3, 43.350937));
    Ev.push_back(hfentry(3, 4.749946));
    break;

    // Fl
  case(114):
    Ev.push_back(hfentry(0, 5793.753853));
    Ev.push_back(hfentry(0, 1058.234593));
    Ev.push_back(hfentry(0, 290.572759));
    Ev.push_back(hfentry(0, 84.879361));
    Ev.push_back(hfentry(0, 22.373353));
    Ev.push_back(hfentry(0, 4.275164));
    Ev.push_back(hfentry(0, 0.410516));
    Ev.push_back(hfentry(1, 1030.043684));
    Ev.push_back(hfentry(1, 276.163652));
    Ev.push_back(hfentry(1, 77.669916));
    Ev.push_back(hfentry(1, 19.109537));
    Ev.push_back(hfentry(1, 3.086296));
    Ev.push_back(hfentry(1, 0.241721));
    Ev.push_back(hfentry(2, 249.547973));
    Ev.push_back(hfentry(2, 64.219125));
    Ev.push_back(hfentry(2, 13.155431));
    Ev.push_back(hfentry(2, 1.125263));
    Ev.push_back(hfentry(3, 45.273367));
    Ev.push_back(hfentry(3, 5.336976));
    break;

    // Mc
  case(115):
    Ev.push_back(hfentry(0, 5899.523013));
    Ev.push_back(hfentry(0, 1079.508693));
    Ev.push_back(hfentry(0, 297.406919));
    Ev.push_back(hfentry(0, 87.422990));
    Ev.push_back(hfentry(0, 23.352750));
    Ev.push_back(hfentry(0, 4.628409));
    Ev.push_back(hfentry(0, 0.494316));
    Ev.push_back(hfentry(1, 1051.027380));
    Ev.push_back(hfentry(1, 282.825175));
    Ev.push_back(hfentry(1, 80.109620));
    Ev.push_back(hfentry(1, 20.025749));
    Ev.push_back(hfentry(1, 3.395389));
    Ev.push_back(hfentry(1, 0.300122));
    Ev.push_back(hfentry(2, 255.888563));
    Ev.push_back(hfentry(2, 66.461595));
    Ev.push_back(hfentry(2, 13.948224));
    Ev.push_back(hfentry(2, 1.337111));
    Ev.push_back(hfentry(3, 47.231643));
    Ev.push_back(hfentry(3, 5.937787));
    break;

    // Lv
  case(116):
    Ev.push_back(hfentry(0, 6006.276636));
    Ev.push_back(hfentry(0, 1101.018964));
    Ev.push_back(hfentry(0, 304.340559));
    Ev.push_back(hfentry(0, 90.018651));
    Ev.push_back(hfentry(0, 24.361772));
    Ev.push_back(hfentry(0, 4.996692));
    Ev.push_back(hfentry(0, 0.587215));
    Ev.push_back(hfentry(1, 1072.246885));
    Ev.push_back(hfentry(1, 289.585866));
    Ev.push_back(hfentry(1, 82.601113));
    Ev.push_back(hfentry(1, 20.971284));
    Ev.push_back(hfentry(1, 3.719157));
    Ev.push_back(hfentry(1, 0.318112));
    Ev.push_back(hfentry(2, 262.327677));
    Ev.push_back(hfentry(2, 68.755341));
    Ev.push_back(hfentry(2, 14.769704));
    Ev.push_back(hfentry(2, 1.563991));
    Ev.push_back(hfentry(3, 49.240774));
    Ev.push_back(hfentry(3, 6.567278));
    break;

    // Ts
  case(117):
    Ev.push_back(hfentry(0, 6113.998867));
    Ev.push_back(hfentry(0, 1122.749661));
    Ev.push_back(hfentry(0, 311.357940));
    Ev.push_back(hfentry(0, 92.650623));
    Ev.push_back(hfentry(0, 25.384797));
    Ev.push_back(hfentry(0, 5.365249));
    Ev.push_back(hfentry(0, 0.680180));
    Ev.push_back(hfentry(1, 1093.686450));
    Ev.push_back(hfentry(1, 296.430008));
    Ev.push_back(hfentry(1, 85.128705));
    Ev.push_back(hfentry(1, 21.930581));
    Ev.push_back(hfentry(1, 4.043072));
    Ev.push_back(hfentry(1, 0.351830));
    Ev.push_back(hfentry(2, 268.849589));
    Ev.push_back(hfentry(2, 71.084655));
    Ev.push_back(hfentry(2, 15.604312));
    Ev.push_back(hfentry(2, 1.791485));
    Ev.push_back(hfentry(3, 51.285055));
    Ev.push_back(hfentry(3, 7.209941));
    break;

    // Og
  case(118):
    Ev.push_back(hfentry(0, 6222.690665));
    Ev.push_back(hfentry(0, 1144.701722));
    Ev.push_back(hfentry(0, 318.459965));
    Ev.push_back(hfentry(0, 95.319797));
    Ev.push_back(hfentry(0, 26.422739));
    Ev.push_back(hfentry(0, 5.735380));
    Ev.push_back(hfentry(0, 0.774032));
    Ev.push_back(hfentry(1, 1115.347016));
    Ev.push_back(hfentry(1, 303.358508));
    Ev.push_back(hfentry(1, 87.693285));
    Ev.push_back(hfentry(1, 22.904566));
    Ev.push_back(hfentry(1, 4.368537));
    Ev.push_back(hfentry(1, 0.394359));
    Ev.push_back(hfentry(2, 275.455220));
    Ev.push_back(hfentry(2, 73.450439));
    Ev.push_back(hfentry(2, 16.453015));
    Ev.push_back(hfentry(2, 2.020924));
    Ev.push_back(hfentry(3, 53.365396));
    Ev.push_back(hfentry(3, 7.866750));
    break;
    
  default:
    std::ostringstream oss;
    oss << "Energies for Z = " << Z << " not implemented!\n";
    throw std::logic_error(oss.str());
  }

  // Fix signs
  for(size_t i=0;i<Ev.size();i++) {
    Ev[i].second=-std::abs(Ev[i].second);
  }

  const char shtypes[]="spdfgh";

  printf("Reference energies for Z=%i\n",Z);
  int lold=-1, iold=-1;
  for(size_t i=0;i<Ev.size();i++) {
    if(lold != Ev[i].first) {
      lold=Ev[i].first;
      iold=Ev[i].first+1; // 1s, 2p, etc
    }
    printf("%i%c %12.6f\n",iold,shtypes[lold],Ev[i].second);
    iold++;
  }

  return Ev;
}

arma::vec getE(const std::vector< std::pair<int, double> > & en) {
  std::vector<double> E;
  for(size_t i=0;i<en.size();i++) {
    for(int m=0;m<2*en[i].first+1;m++)
      E.push_back(en[i].second);
  }

  return arma::sort(arma::conv_to<arma::vec>::from(E),"ascend");
}

arma::vec compute_energies(const arma::mat & T, const arma::mat & Sinvh, const std::vector<arma::uvec> & dsym, const atomic::basis::TwoDBasis & basis, const arma::vec & p) {

  double dz=exp(p(0));
  double Hz=exp(p(1));
  
  // Form nuclear attraction energy matrix
  arma::mat Vnuc(basis.gsz(dz,Hz));
  
  // Form Hamiltonian
  arma::mat H0(T+Vnuc);
    
  // Solve energies
  arma::vec E;
  arma::mat C;
  scf::eig_gsym_sub(E,C,H0,Sinvh,dsym,false);

  return E;
}
  
double compute_value(const arma::mat & T, const arma::mat & Sinvh, const std::vector<arma::uvec> & dsym, const atomic::basis::TwoDBasis & basis, const arma::vec & Eref, const arma::vec & p) {
  arma::vec E=compute_energies(T, Sinvh, dsym, basis, p);
  // Grab interesting part
  E=E.subvec(0,Eref.n_elem-1);

  // chi2 value
  return arma::sum(arma::square(E-Eref)/(-Eref));
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<double>("Rmax", 0, "practical infinity in au", false, 40.0);
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for logarithmic", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("niter", 0, "number of iterations", false, 4);
  parser.add<int>("ntrials", 0, "number of trials in each direction", false, 4);
  parser.add<double>("dz", 0, "d_Z parameter", false, 0.8);
  parser.add<double>("Hz", 0, "H_Z parameter", false, 0.0);
  parser.parse_check(argc, argv);

  // Get parameters
  double Rmax(parser.get<double>("Rmax"));
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int primbas(parser.get<int>("primbas"));
  // Number of elements
  int Nelem(parser.get<int>("nelem"));
  // Number of nodes
  int Nnodes(parser.get<int>("nnodes"));

  // Order of quadrature rule
  int Nquad(parser.get<int>("nquad"));

  // Nuclear charge
  int Z(get_Z(parser.get<std::string>("Z")));

  int niter(parser.get<int>("niter"));
  int ntrials(parser.get<int>("ntrials"));
  
  double dz(parser.get<double>("dz"));
  double Hz(parser.get<double>("Hz"));

  // Get primitive basis
  polynomial_basis::PolynomialBasis *poly(polynomial_basis::get_basis(primbas,Nnodes));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");

  // Get energies
  std::vector< std::pair<int, double> > en(get_energies(Z));
  arma::vec Eref(getE(en));

  // Angular grid
  int lmax=0;
  for(size_t i=0;i<en.size();i++)
    lmax=std::max(lmax,en[i].first);
  
  printf("Using %i point quadrature rule.\n",Nquad);

  atomic::basis::TwoDBasis basis=atomic::basis::TwoDBasis(Z, poly, Nquad, Nelem, Rmax, lmax, lmax, igrid, zexp);
  printf("Basis set consists of %i angular shells composed of %i radial functions, totaling %i basis functions\n",(int) basis.Nang(), (int) basis.Nrad(), (int) basis.Nbf());

  // Symmetry indices
  int symm=2;
  bool diag=true;
  std::vector<arma::uvec> dsym;
  if(symm)
    dsym=basis.get_sym_idx(symm);

  // Get half-inverse
  arma::mat Sinvh(basis.Sinvh(!diag,symm));

  // Form kinetic energy matrix
  arma::mat T(basis.kinetic());

  // Parameters
  arma::vec p(2);
  p.zeros();

  // Initial guess
  if(dz!=0.0) // d_z
    p(0)=log(dz);
  if(Hz!=0.0) // H_z
    p(1)=log(Hz);
  else
    // GSZ expression
    p(1)=p(0) + 0.4*log(Z-1);

  // Step size
  const double maxhh=0.3;
  double hh(maxhh);
  
  // Loop
  double E0=compute_value(T,Sinvh,dsym,basis,Eref,p);
  const double Einit(E0);
  printf("Initial value %e\n",E0);
  printf("Initial parameters: dz = % e, Hz = %e\n",exp(p(0)),exp(p(1)));
  {
    arma::mat Etes(Eref.n_elem,3);
    Etes.col(0)=Eref;
    Etes.col(1)=compute_energies(T,Sinvh,dsym,basis,p).subvec(0,Eref.n_elem-1);
    Etes.col(2)=(Etes.col(1)-Etes.col(0))/Eref;
    Etes.print("GWH energies");
    printf("\n");
  }
  
  for(int iit=0;iit<niter;iit++) {
    double Enew;

    // Trials
    arma::ivec ixs(arma::linspace<arma::ivec>(-ntrials,ntrials,2*ntrials+1));
    arma::vec xs(arma::linspace<arma::vec>(-ntrials,ntrials,2*ntrials+1)*hh);

    printf("Iteration %i\n",iit);
    xs.t().print("xs");
    printf("Current values dz = %e Hz = %e\n",exp(p(0)),exp(p(1)));
    printf("Scanning in window dz = %e .. %e Hz = %e .. %e\n",exp(p(0)+xs(0)),exp(p(0)+xs(xs.n_elem-1)),exp(p(1)+xs(0)),exp(p(1)+xs(xs.n_elem-1)));
    
    // Brute-force search window
    arma::mat window(2*ntrials+1,2*ntrials+1);
    for(size_t i=0;i<window.n_rows;i++) {
      for(size_t j=0;j<window.n_cols;j++) {
	arma::vec pt(p);
	pt(0)+=xs(i);
	pt(1)+=xs(j);
	
	window(i,j)=compute_value(T,Sinvh,dsym,basis,Eref,pt);
      }
    }

    // Find minimum
    arma::uword imin, jmin;
    Enew=window.min(imin,jmin);
    p(0)+=xs(imin);
    p(1)+=xs(jmin);
    printf("Minimum found at (%e,%e)\n",xs(imin),xs(jmin));
    //printf("imin=%i, jmin=%i\n",(int) imin,(int) jmin);
    // Update search window?
    if(abs(ixs(imin))<=1 && abs(ixs(jmin))<=1) {
      hh/=exp(1);
      printf("Decreased step size to %e\n",hh);
    } else if(abs(ixs(imin))==ntrials || abs(ixs(jmin))==ntrials) {
      hh*=exp(1);
      if(hh>maxhh)
	hh=maxhh;
      printf("Increased step size to %e\n",hh);
    }
      
      /*
      // Calculate Hessian
      arma::mat hess(h.n_elem, h.n_elem);
      for(size_t i=0;i<h.n_elem;i++) {
	for(size_t j=0;j<h.n_elem;j++) {
	  arma::vec ih(h);
	  ih.zeros();
	  arma::vec jh(h);
	  jh.zeros();
	
	  ih(i)=h(i);
	  jh(j)=h(j);

	  double Eij=compute_value(T,Sinvh,dsym,basis,Eref,p+ih+jh);
	  double Ei=compute_value(T,Sinvh,dsym,basis,Eref,p+ih);
	  double Ej=compute_value(T,Sinvh,dsym,basis,Eref,p+jh);
	  hess(i,j)=(Eij-Ei-Ej+E0)/(ih(i)*jh(j));
	}
      }
      hess.print("Hessian");
      hess=(hess+hess.t())/2;

      arma::vec eh;
      arma::mat ch;
      arma::eig_sym(eh,ch,hess);
      eh.t().print("Hessian eigenvalues");

      // Shift eigenvalues
      if(eh(0)<0.0) {
	hess+=(10-eh(0))*arma::eye<arma::mat>(hess.n_rows,hess.n_cols);
	arma::eig_sym(eh,ch,hess);
	eh.t().print("Shifted Hessian eigenvalues");
      }
      
      // E = E0 + gx + 1/2 x^t H x
      // -> E' = g + H x = 0 when x = - H^-1 g
      
      arma::vec x = - arma::inv(hess)*g;
      p+=x;
      Enew=compute_value(T,Sinvh,dsym,basis,Eref,p);
    }
      */

    double dE=Enew-E0;
    printf("Error %e changed by %e\n", Enew, dE);
    E0=Enew;
    printf("Current parameters: dz = % e, Hz = %e\n",exp(p(0)),exp(p(1)));

    arma::mat Etes(Eref.n_elem,3);
    Etes.col(0)=Eref;
    Etes.col(1)=compute_energies(T,Sinvh,dsym,basis,p).subvec(0,Eref.n_elem-1);
    Etes.col(2)=(Etes.col(1)-Etes.col(0))/Eref;
    Etes.print("GWH energies");
    printf("\n");

    if(hh<1e-6) {
      printf("Converged.\n");
      break;
    }
  }

  printf("Optimization decreased error from %e to %e i.e. by %e (%e %%)\n",Einit,E0,E0-Einit,(E0-Einit)*100.0/Einit); 
  printf("Final values for d_Z and H_Z are % .6f % .6f\n",exp(p(0)),exp(p(1)));
  
  return 0;
}
