#
#                This source code is part of
#
#                          HelFEM
#                             -
# Finite element methods for electronic structure calculations on small systems
#
# Written by Susi Lehtola, 2018-
# Copyright (c) 2018- Susi Lehtola
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# Code to generate sources for arbitrary order derivatives of LIP
# basis functions.

for order in range(0,6):
    fname = 'f'
    if order == 1:
        fname = 'df'
    if order > 1:
        fname = 'd{}f'.format(order)

    print('void LIPBasis::eval_{}_raw(const arma::vec & x, arma::mat & {}) const {{'.format(fname, fname))

    print('// Allocate memory\n{}.zeros(x.n_elem, x0.n_elem);'.format(fname))

    print('// Loop over points\nfor(size_t ix=0; ix<x.n_elem; ix++) {')
    print('// Loop over polynomials\nfor(size_t fi=0; fi<x0.n_elem; fi++) {')

    # Derivative loops
    for ider in range(1,order+1):
        print('// Derivative {0} acting on index\nfor(size_t d{0}=0; d{0}<x0.n_elem; d{0}++) {{'.format(ider))

        # Skip these terms
        check_terms = ['d{}'.format(o) for o in range(1,ider)]
        check_terms.append('fi')
        for term in check_terms:
            print('if(d{0} == {1}) continue;'.format(ider, term))
            
    # Now do the product
    print('// Form the LIP product\ndouble {}val = 1.0;'.format(fname))
    print('for(size_t ip=0; ip<x0.n_elem; ip++) {')
    print('// Skip terms which have been acted upon by a derivative')
    check_terms = ['d{}'.format(o) for o in range(1,order+1)] 
    check_terms.append('fi')
    for term in check_terms:
        print('if(ip == {}) continue;'.format(term))
    print('{}val *= (x(ix)-x0(ip))/(x0(fi)-x0(ip));'.format(fname))
    print('}')

    # Form the divider
    div_terms = ['(x0(fi)-x0(d{}))'.format(o) for o in range(1,order+1)]
    if len(div_terms)>0:
        print('// Apply derivative denominators\n{}val /= '.format(fname) + '*'.join(div_terms) + ';')
        
    print('// Store the computed value')
    print('{0}(ix,fi) += {0}val;'.format(fname))

    for ider in range(1,order+1):
        print('}')
    # End segment
    print('}\n}\n}')