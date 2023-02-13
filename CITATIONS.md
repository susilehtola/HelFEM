The core of the program has been described in two articles focusing on the atomic and diatomic parts, respectively:
* S. Lehtola, Fully numerical Hartree‐Fock and density functional calculations. I. Atoms. Int. J. Quantum Chem. 2019, e25945. doi:10.1002/qua.25945. arXiv:1810.11651
* S. Lehtola, Fully numerical Hartree‐Fock and density functional calculations. II. Diatomic molecules. Int. J. Quantum Chem. 2019, e25944. doi:10.1002/qua.25944. arXiv:1810.11653

There's also an article that discusses calculations with spherically averaged densities at the HF, LDA and GGA levels of theory:
* S. Lehtola, Fully numerical calculations on atoms with fractional occupations and range-separated exchange functionals, Phys. Rev. A 101, 012516 (2020). doi:10.1103/PhysRevA.101.012516
The formulation for meta-GGA functionals has been discussed in
* S. Lehtola, Meta-GGA density functional calculations on atoms with spherically symmetric densities in the finite element formalism, to appear on the arXiv

The Lagrange interpolating basis was discussed in the first article, doi:10.1002/qua.25945. High-order Hermite interpolating functions and their importance in meta-GGA calculations has been discussed in
* S. Lehtola, Atomic electronic structure calculations with Hermite interpolating polynomials. arXiv:2302.00440

The calculations in both the atomic and diatomic programs are started by default from atomic potentials as described in
* S. Lehtola, Assessment of initial guesses for self-consistent field calculations. Superposition of Atomic Potentials: simple yet efficient, J. Chem. Theory Comput. 15, 1593 (2019). doi:10.1021/acs.jctc.8b01089

which leads to fast convergence of the self-consistent field calculations. The program employs local exchange potentials determined with HelFEM, according to the procedure laid out in the fractional occupation paper.

Calculations in finite magnetic fields have been described in:
* S. Lehtola, M. Dimitrova, and D. Sundholm, Fully numerical electronic structure calculations on diatomic molecules in weak to strong magnetic fields, Mol. Phys. (2019), doi:10.1080/00268976.2019.1597989. arXiv:1812.06274

There is also a review paper discussing fully numerical calculations on atoms and diatomic molecules in general:
* S. Lehtola, A review on non-relativistic fully numerical electronic structure calculations on atoms and diatomic molecules, Int. J. Quantum Chem. 2019, e25968. doi:10.1002/qua.25968. arXiv:1902.01431

The diatomic program relies on a library for the calculation of Legendre functions that has been described in
* B. Schneider et al, Comput. Phys. Commun. 2010, 181, 2091–2097
* B. Schneider et al, Comput. Phys. Commun. 2018, 225, 192–193

Density functionals are evaluated in both the atomic and diatomic program with the libxc library, which has been described in
* S. Lehtola et al, SoftwareX 2018, 7, 1–5.

In addition to the above papers, thorough calculations of atomic energies have been presented in
* S. Lehtola, L. Visscher, and E. Engel, Efficient implementation of the superposition of atomic potentials initial guess for electronic structure calculations in Gaussian basis sets, J. Chem. Phys. 152, 144105 (2020). doi:10.1063/5.0004046
* S. Lehtola, Polarized Gaussian basis sets from one-electron ions, J. Chem. Phys. 152, 134108 (2020). doi:10.1063/1.5144964
