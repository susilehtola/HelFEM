HelFEM -- Helsinki Finite Element Suite for atoms and diatomic molecules
------------------------------------------------------------------------

HelFEM is a suite of programs for finite element calculations on atoms
and diatomic molecules at the Hartree-Fock or density-functional
levels of theory. Hundreds of functionals at the local spin density
approximation (LDA), generalized gradient approximation (GGA), and
meta-GGA levels of theory are supported.

Compilation is straightforward with CMake. To compile, you have to set
some variables in CMake.system, such as the directories where the
Armadillo headers, and libxc headers and libraries reside, and how to
link against LAPACK.

Susi Lehtola
2018-10-24
