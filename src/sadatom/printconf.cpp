#include "configurations.h"
#include "../general/elements.h"
#include <cstdio>
#include <sstream>

std::vector<occ_t> get_occ(int Z) {
  std::vector<occ_t> occs(get_configuration(Z));
  int ntot=0;
  for(size_t i=0;i<occs.size();i++)
    ntot+=occs[i].second;
  if(ntot!=Z) {
    std::ostringstream oss;
    oss << "Mismatch neutral: " << ntot << " vs " << Z << "!\n";
    throw std::logic_error(oss.str());
  }

  static const int nobles[]={2,10,18,36,54,86};
  size_t N=sizeof(nobles)/sizeof(nobles[0]);

  std::vector<occ_t> core;
  for(size_t i=N-1;i<N;i--)
    if(Z>nobles[i]) {
      core=get_configuration(nobles[i]);
      break;
    }
  if(core.size())
    occs.erase(occs.begin(),occs.begin()+core.size()-1);

  return occs;
}

std::vector<occ_t> get_cocc(int Z) {
  std::vector<occ_t> occs(get_cationic_configuration(Z));
  int ntot=0;
  for(size_t i=0;i<occs.size();i++)
    ntot+=occs[i].second;
  if(ntot!=Z-1) {
    std::ostringstream oss;
    oss << "Mismatch cation: " << ntot << " vs " << Z << "!\n";
    throw std::logic_error(oss.str());
  }

  static const int nobles[]={2,10,18,36,54,86};
  size_t N=sizeof(nobles)/sizeof(nobles[0]);

  std::vector<occ_t> core;
  for(size_t i=N-1;i<N;i--)
    if(Z>nobles[i]) {
      core=get_configuration(nobles[i]);
      break;
    }
  if(core.size())
    occs.erase(occs.begin(),occs.begin()+core.size()-1);

  return occs;
}

int main(void) {
  for(int Z=1;Z<=92;Z++) {
    std::vector<occ_t> occs(get_occ(Z));
    std::vector<occ_t> coccs(get_cocc(Z));
    printf("%3i\t%-2s\t",Z,element_symbols[Z].c_str());

    const char shs[]="spdf";
    {
      int nums[]={1, 2, 3, 4};
      for(size_t i=0;i<occs.size();i++) {
        printf(" %i%c^%i",nums[occs[i].first]++,shs[occs[i].first],occs[i].second);
      }
    }
    printf("\t");
    {
      int nums[]={1, 2, 3, 4};
      for(size_t i=0;i<coccs.size();i++) {
        printf(" %i%c^%i",nums[coccs[i].first]++,shs[coccs[i].first],coccs[i].second);
      }
    }
    printf("\n");
  }

  return 0;
}
