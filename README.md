HelFEM -- Helsinki Finite Element Suite for atoms and diatomic molecules
------------------------------------------------------------------------

HelFEM is a suite of programs for finite element calculations on atoms
and diatomic molecules at the Hartree-Fock or density-functional
levels of theory. Hundreds of functionals at the local spin density
approximation (LDA), generalized gradient approximation (GGA), and
meta-GGA levels of theory are supported.

The program has been described in the following three articles:
* S. Lehtola, [Fully numerical Hartree‐Fock and density functional calculations. I. Atoms](http://doi.org/10.1002/qua.25945). Int. J. Quantum Chem. 2019, e25945. doi:10.1002/qua.25945. arXiv:1810.11651
* S. Lehtola, [Fully numerical Hartree‐Fock and density functional calculations. II. Diatomic molecules](http://doi.org/10.1002/qua.25944). Int. J. Quantum Chem. 2019, e25944. doi:10.1002/qua.25944. arXiv:1810.11653
* S. Lehtola, [Fully numerical calculations on atoms with fractional occupations and range-separated exchange functionals](http://doi.org/10.1002/qua.25944), Phys. Rev. A 101, 012516 (2020). doi:10.1103/PhysRevA.101.012516

The program can also be used to form starting potentials for molecular electronic structure calculations with the superposition of atomic potentials method described in [J. Chem. Theory Comput. 15, 1593 (2019)](http://doi.org/10.1021/acs.jctc.8b01089), as discussed in [J. Chem. Phys. 152, 144105 (2020)](http://doi.org/10.1063/5.0004046).

There is also a general review paper on fully numerical calculations on atoms and diatomic molecules that should be of interest
* S. Lehtola, [A review on non-relativistic fully numerical electronic structure calculations on atoms and diatomic molecules](http://doi.org/10.1002/qua.25968), Int. J. Quantum Chem. 119, e25968 (2019). doi:10.1002/qua.25968

Compilation is straightforward with CMake. To compile, you have to set
some variables in CMake.system, such as the directories where the
Armadillo headers, and libxc headers and libraries reside, and how to
link against LAPACK.

Susi Lehtola
2022-10-20
