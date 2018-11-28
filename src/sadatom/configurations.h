#ifndef CONFIGURATIONS_H
#define CONFIGURATIONS_H

#include <vector>

/// Occupation for a shell: (l, nelectrons)
typedef std::pair<int, int> occ_t;

/// Get configuration (l, nelectrons) for atom with charge Z
std::vector<occ_t> get_configuration(int Z);

#endif
