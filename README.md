HelFEM -- Helsinki Finite Element Suite for atoms and diatomic molecules
------------------------------------------------------------------------

HelFEM is a suite of programs for finite element calculations on atoms
and diatomic molecules at the Hartree-Fock or density-functional
levels of theory. Hundreds of functionals at the local spin density
approximation (LDA), generalized gradient approximation (GGA), and
meta-GGA levels of theory are supported.

The program has been described in two articles focusing on the atomic and diatomic parts, respectively
* S. Lehtola, Hartree--Fock and hybrid density functional theory calculations of static properties at the complete basis set limit via finite elements. I. Atoms. [arXiv:1810.11651](http://arxiv.org/abs/1810.11651)
* S. Lehtola, Hartree--Fock and hybrid density functional theory calculations of static properties at the complete basis set limit via finite elements. II. Diatomic molecules. [arXiv:1810.11653](http://arxiv.org/abs/1810.11653)

Compilation is straightforward with CMake. To compile, you have to set
some variables in CMake.system, such as the directories where the
Armadillo headers, and libxc headers and libraries reside, and how to
link against LAPACK.

Susi Lehtola
2018-10-24
