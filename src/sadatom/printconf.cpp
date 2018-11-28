#include "configurations.h"
#include "../general/elements.h"
#include <cstdio>


int main(void) {
  for(int Z=1;Z<=92;Z++) {
    std::vector<occ_t> occs(get_configuration(Z));
    printf("%3i\t%-2s\t",Z,element_symbols[Z].c_str());

    int nums[]={1, 2, 3, 4};
    const char shs[]="spdf";

    int ntot=0;
    for(size_t i=0;i<occs.size();i++) {
      printf(" %i%c^%i",nums[occs[i].first]++,shs[occs[i].first],occs[i].second);
      ntot+=occs[i].second;
    }
    printf("\n");
    if(ntot!=Z)
      printf("Mismatch: %i vs %i\n",ntot,Z);
  }

  return 0;
}

