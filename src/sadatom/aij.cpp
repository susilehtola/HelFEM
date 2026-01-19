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
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "utils.h"
#include "dftgrid.h"
#include "solver.h"
#include "configurations.h"
#include <cfloat>

using namespace helfem;

int consistent_lmax(int njellium) {
  // Consistent pairing of lmax, given the number of jellium electrons
  const std::vector<int> magic({2, 8, 19, 36, 59, 89, 118, 163, 215, 269, 341, 425, 516, 612, 731, 859, 994, 1131, 1300, 1424, 1625, 1820, 2043, 2295, 2543, 2821, 3080, 3337, 3665, 3987, 4373, 4741, 5155, 5611, 5961, 6407, 6909, 7406, 7964, 8512, 9058, 9591, 10147, 10792, 11440, 12116, 12825, 13611, 14431, 15158, 15988, 16805, 17706, 18621, 19645, 20657, 21501, 22408, 23489, 24637, 25833, 26930, 28248, 29434, 30645, 31904, 33212, 34657, 36113, 37655, 39105, 40511, 41843, 43494, 45069, 46840, 48404, 50040, 51814, 53644, 55608, 57488, 59540, 61712, 63658, 65554, 67735, 69649, 71971, 74347, 76766, 78827, 81288, 83721, 86369, 89041, 91671, 94286, 96899, 99284});
  for(size_t i=0;i<magic.size();i++) {
    if(magic[i] > njellium)
      return (int) i;
  }

  std::ostringstream oss;
  oss << "Magic numbers not tabulated for njellium=" << njellium << "!\n";
  throw std::logic_error(oss.str());
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<double>("nufreq", 0, "frequency of uniform elements", false, 0.5);
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<int>("taylor_order", 0, "order of Taylor expansion near the nucleus", false, -1);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<int>("njellium", 0,"number of jellium electrons", true);
  parser.add<double>("rs", 0, "Wigner-Seitz radius for jellium", true);
  parser.add<double>("convthr", 0, "Convergence threshold", false, 1e-7);
  parser.add<bool>("vacancy", 0, "Jellium vacancy model?", false, false);
  parser.add<int>("maxiter", 0, "maximum number of iterations", false, 1024);
  parser.add<bool>("zeroright", 0, "Zero the right-hand function value", false, false);
  parser.add<double>("Rmax", 0, "Size of vacuum region", false, 40.0);
  parser.add<int>("M", 0, "spin multiplicity", true);
  parser.add<int>("lmax", 0, "maximum angular momentum", false, 4);
  parser.add<bool>("oda", 0, "Run optimal damping?", false, true);
  parser.parse_check(argc, argv);

  // Get parameters
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int Nelem(parser.get<int>("nelem"));
  double Nufreq(parser.get<double>("nufreq"));

  int Z(get_Z(parser.get<std::string>("Z")));
  int Q(parser.get<int>("Q"));
  int M(parser.get<int>("M"));

  int Nnodes(parser.get<int>("nnodes"));
  int Nquad(parser.get<int>("nquad"));
  std::string method(parser.get<std::string>("method"));
  double dftthr(parser.get<double>("dftthr"));

  int primbas(parser.get<int>("primbas"));
  int taylor_order(parser.get<int>("taylor_order"));
  std::string xparf(parser.get<std::string>("x_pars"));
  std::string cparf(parser.get<std::string>("c_pars"));

  int njellium(parser.get<int>("njellium"));
  double rs(parser.get<double>("rs"));
  double Rmax(parser.get<double>("Rmax"));
  bool vacancy(parser.get<bool>("vacancy"));

  int maxiter(parser.get<int>("maxiter"));
  double convthr(parser.get<double>("convthr"));

  bool zeroright(parser.get<bool>("zeroright"));
  int lmax(parser.get<int>("lmax"));
  bool oda(parser.get<bool>("oda"));

  // Parse xc parameters
  arma::vec x_pars, c_pars;
  if(xparf.size()) {
    x_pars = scf::parse_xc_params(xparf);
    x_pars.t().print("Exchange functional parameters");
  }
  if(cparf.size()) {
    c_pars = scf::parse_xc_params(cparf);
    c_pars.t().print("Correlation functional parameters");
  }

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->get_nbf();
  else if(Nquad<2*poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  // Order of quadrature rule
  printf("Using %i point quadrature rule.\n",Nquad);

  // Set default order of Taylor expansion
  if(taylor_order==-1)
    taylor_order = poly->get_nprim()-1;

  // Functional
  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if(!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if(!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");

  // Determine box size
  double R = 0.0, r_inner = 0.0, r_outer = 0.0;
  if(vacancy) {
    // Inner cavity has zero charge for a radius that matches the density of the background charge.
    // Outer cavity has constant background charge. Altogether, this leads to
    R = cbrt(Z-Q+njellium)*rs;
    r_inner = cbrt(Z)*rs;
    r_outer = R - r_inner;
  } else {
    R = cbrt(njellium)*rs;
    r_inner = 0.0;
    r_outer = R;
  }

  lmax = std::max(lmax, consistent_lmax(njellium));

  // Suitable uniform spacing to account for Friedel oscillations is pi/k_F
  double friedel_period = std::cbrt(4*M_PI*M_PI/9)*rs;
  // Number of uniform elements is thus
  int Nuelem = (rs>0) ? std::ceil(R/friedel_period*Nufreq) : 0;

  if(vacancy) {
    printf("%i jellium electrons with rs = % .3f and vacancy model leads to r_inner = % .10f r_outer = % .10f lmax = %i\n",njellium,rs,r_inner,r_outer,lmax);
  } else {
    printf("%i jellium electrons with rs = % .3f leads to R = % .10f lmax = %i\n",njellium,rs,R,lmax);
  }
  printf("Friedel period is % .3f, using %i uniform elements.\n", friedel_period, Nuelem);
  double Erep = (rs > 0 ) ? 3*std::pow(R,5)/(5*std::pow(rs,6)) : 0.0;

  // Uniform part of grid
  arma::vec bval_unif;
  if(Nuelem)
    bval_unif = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nuelem, R, 1, 0.0, 0, 0, 0, Z, 0, 0, 0.0, false, 0.0);

  arma::vec bval;
  if(Nelem>0) {
    // Atomic grid
    double rinfty = (Nuelem>0) ? bval_unif(1) : Rmax;
    arma::vec bval_atom = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nelem, rinfty, igrid, zexp, 0, 0, 0.0, Z, 0, 0, 0.0, false, 0.0);

    // Glue grids together
    if(bval_unif.n_elem) {
      bval.zeros(bval_atom.n_elem+bval_unif.n_elem-2);
      bval.subvec(0,bval_atom.n_elem-1) = bval_atom;
      if(bval_atom(bval_atom.n_elem-1) != bval_unif(1)) {
        std::ostringstream oss;
        oss << "Grids don't coincide: difference " << bval_atom(bval_atom.n_elem-1) - bval_unif(1) << "!\n";
        throw std::logic_error(oss.str());
      }
      if(bval_unif.n_elem>2) {
        bval.subvec(bval_atom.n_elem,bval.n_elem-1) = bval_unif.subvec(2,bval_unif.n_elem-1);
      }
    } else {
      bval = bval_atom;
    }
  } else {
    bval=bval_unif;
  }

  // Handle vacancy case
  if(vacancy) {
    arma::vec vbval(bval.n_elem+1);
    vbval.subvec(0,bval.n_elem-1)=bval;
    vbval(bval.n_elem) = r_inner;
    bval = arma::sort(vbval, "ascend");
  }

  // Add vacuum region
  if(Nuelem>0 and Rmax>0.0) {
    int Nvelem = std::ceil(Rmax/friedel_period*Nufreq);

    // Points in vacuum region
    arma::vec bval_vac = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nvelem, Rmax, 1, 0.0, 0, 0, 0, Z, 0, 0, 0.0, false, 0.0);
    // offset
    bval_vac += bval(bval.n_elem-1);

    arma::vec bval_new(bval.n_elem+bval_vac.n_elem-1);
    bval_new.subvec(0,bval.n_elem-1) = bval;
    bval_new.subvec(bval.n_elem,bval_new.n_elem-1) = bval_vac.subvec(1,bval_vac.n_elem-1);
    if(bval_new(bval.n_elem-1) != bval_vac(0)) {
      std::ostringstream oss;
      oss << "Grids don't coincide: difference " << bval_new(bval.n_elem-1) - bval_vac(0) << "!\n";
      throw std::logic_error(oss.str());
    }
    bval=bval_new;
  }
  bval.print("Final grid for calculation");

  bool zeroder = false;
  auto basis = sadatom::basis::TwoDBasis(Z, modelpotential::POINT_NUCLEUS, 0.0, poly, zeroder, Nquad, bval, taylor_order, lmax, zeroright);
  printf("Basis set has %i radial functions\n",(int) basis.Nbf());
  printf("%ith order Taylor series used to evaluate basis functions for r <= %e, error %e\n",taylor_order, basis.get_small_r_taylor_cutoff(), basis.get_taylor_diff());

  std::function<double(double, double)> sphere_pot = [&](double r, double R) {
    const double prefac = std::pow(R/rs,3);
    if(r<R) {
      return prefac*(3.0 - std::pow(r/R,2))/(2*R);
    } else {
      return prefac/r;
    }
  };

  std::function<double(double)> potfunc = [&](double r) {
    // The background potential is the difference between the
    // uniform background charge, and the vacancy in the
    // middle. Since the potential is attractive for electrons,
    // we flip the sign here.
    if(r_inner>0)
      return -sphere_pot(r,r_outer)+sphere_pot(r,r_inner);
    else if(r_outer>0)
      return -sphere_pot(r,r_outer);
    else
      return 0.0;
  };

  // Energy of nucleus in external field
  double Enucfield = -Z*potfunc(0);
  printf("potfunc(0) = %e Enucfield = %e\n",potfunc(0),Enucfield);
  fflush(stdout);

  // Form overlap matrix
  arma::mat S=basis.overlap();
  // Get half-inverse
  arma::mat Sinvh=basis.Sinvh();
  // Form kinetic energy matrix
  arma::mat T=basis.kinetic();
  // Form kinetic energy matrix
  arma::mat Tl=basis.kinetic_l();
  // Form nuclear attraction energy matrix
  arma::mat Vnuc=basis.nuclear();
  // Uniform background potential
  arma::mat Vunif=basis.potential(potfunc);

  // Form DFT grid
  auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
  // Compute two-electron integrals
  basis.compute_tei();

  // Jellium Hamiltonian
  OpenOrbitalOptimizer::FockMatrix<double> H_jellium(lmax+1);
  for(int l=0;l<=lmax;l++) {
    // Uniform potential is cancelled out by the jellium density
    H_jellium[l] = Sinvh.t() * (T + l*(l+1)*Tl) * Sinvh;
  }

  // Compute the jellium eigenvalues
  std::vector<arma::mat> Cjellium(lmax+1);
  std::vector<arma::vec> Ejellium(lmax+1);
  std::vector<std::tuple<double,int,int>> jellium_energies;
  for(int l=0;l<=lmax;l++) {
    arma::eig_sym(Ejellium[l], Cjellium[l], H_jellium[l]);
    // Convert to non-orthogonal basis
    Cjellium[l] = Sinvh*Cjellium[l];

    printf("l = %i eigenvalues\n",l);
    Ejellium[l].t().print();

    for(size_t io=0;io<Ejellium[l].n_elem;io++)
      jellium_energies.push_back(std::make_tuple(Ejellium[l](io),l,io));
  }
  std::sort(jellium_energies.begin(), jellium_energies.end(), [](std::tuple<double,int,int> const & t1, std::tuple<double,int,int> const & t2) {
    return std::get<0>(t1) < std::get<0>(t2);
  });


  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Kinetic energy
    double Ekin=0.0;
    // Radial density matrix
    arma::mat Prad(Sinvh.n_rows, Sinvh.n_rows, arma::fill::zeros);
    arma::cube Pl(Sinvh.n_rows, Sinvh.n_rows, lmax+1, arma::fill::zeros);
    for(int l=0;l<=lmax;l++) {
      // Nothing to do
      if(arma::max(arma::abs(occupations[l]))==0.0)
        continue;

      // Same radial basis for all l!
      arma::mat C = Sinvh*orbitals[l];
      arma::mat P = C*arma::diagmat(occupations[l])*C.t();
      Pl.slice(l) = P;
      Prad += P;

      // Kinetic energy
      Ekin += arma::trace(P*T) + l*(l+1)*arma::trace(P*Tl);
    }

    double Enuc=arma::trace(Prad*Vnuc);
    double Eunif=arma::trace(Prad*Vunif);

    // Coulomb matrix
    double angfac(4.0*M_PI);
    arma::mat J(basis.coulomb(Prad/angfac));

    double Exc=0.0;
    arma::cube XC;
    double nelnum;
    if(x_func > 0 || c_func > 0) {
      grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pl/angfac, XC, Exc, nelnum, dftthr);
      // Potential needs to be divided as well
      XC/=angfac;
      if(verbose) {
        printf("DFT energy %.10e\n",Exc);
        printf("Error in integrated number of electrons % e\n",nelnum-(Z-Q+njellium));
        fflush(stdout);
      }
    }

    double Ecoul = 0.5*arma::trace(Prad*J);
    double Etot = Ekin + Enuc + Enucfield + Eunif + Erep + Ecoul + Exc;

    if(true) {
      printf("kinetic energy         % .10f\n",Ekin);
      printf("nuclear attraction     % .10f\n",Enuc);
      printf("nucleus-field term     % .10f\n",Enucfield);
      printf("background repulsion   % .10f\n",Erep);
      printf("background attraction  % .10f\n",Eunif);
      printf("Coulomb repulsion      % .10f\n",Ecoul);
      printf("exchange-correlation   % .10f\n",Exc);
      printf("total energy           % .10f\n",Etot);
    }

    OpenOrbitalOptimizer::FockMatrix<double> fock(lmax+1);
    for(int l=0;l<=lmax;l++) {
      fock[l] = T + l*(l+1)*Tl + Vnuc + Vunif + J;
      if(x_func>0 || c_func>0)
        fock[l] += XC.slice(l);
      fock[l] = Sinvh.t() * fock[l] * Sinvh;
    }
    return std::make_pair(Etot,fock);
  };

  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Kinetic energy
    double Ekin=0.0;
    // Radial density matrix
    arma::mat Prad(Sinvh.n_rows, Sinvh.n_rows, arma::fill::zeros);

    arma::cube Pal(Sinvh.n_rows, Sinvh.n_rows, lmax+1, arma::fill::zeros);
    arma::cube Pbl(Sinvh.n_rows, Sinvh.n_rows, lmax+1, arma::fill::zeros);
    for(int l=0;l<=lmax;l++) {
      // Nothing to do
      if(arma::max(arma::abs(occupations[l]))==0.0)
        continue;

      // Same radial basis for all l!
      arma::mat Ca = Sinvh*orbitals[l];
      arma::mat Cb = Sinvh*orbitals[l+lmax+1];
      arma::mat Pa = Ca*arma::diagmat(occupations[l])*Ca.t();
      arma::mat Pb = Cb*arma::diagmat(occupations[l+lmax+1])*Cb.t();
      Pal.slice(l) = Pa;
      Pbl.slice(l) = Pb;
      Prad += Pa+Pb;

      // Kinetic energy
      Ekin += arma::trace((Pa+Pb)*T) + l*(l+1)*arma::trace((Pa+Pb)*Tl);
    }

    double Enuc=arma::trace(Prad*Vnuc);
    double Eunif=arma::trace(Prad*Vunif);

    // Coulomb matrix
    double angfac(4.0*M_PI);
    arma::mat J(basis.coulomb(Prad/angfac));

    double Exc=0.0;
    arma::cube XCa, XCb;
    double nelnum;
    if(x_func > 0 || c_func > 0) {
      grid.eval_Fxc(x_func, x_pars, c_func, c_pars, Pal/angfac, Pbl/angfac, XCa, XCb, Exc, nelnum, true, dftthr);
      // Potential needs to be divided as well
      XCa/=angfac;
      XCb/=angfac;
      if(verbose) {
        printf("DFT energy %.10e\n",Exc);
        printf("Error in integrated number of electrons % e\n",nelnum-(Z-Q+njellium));
        fflush(stdout);
      }
    }

    double Ecoul = 0.5*arma::trace(Prad*J);
    double Etot = Ekin + Enuc + Enucfield + Eunif + Erep + Ecoul + Exc;

    if(true) {
      printf("kinetic energy         % .10f\n",Ekin);
      printf("nuclear attraction     % .10f\n",Enuc);
      printf("nucleus-field term     % .10f\n",Enucfield);
      printf("background repulsion   % .10f\n",Erep);
      printf("background attraction  % .10f\n",Eunif);
      printf("Coulomb repulsion      % .10f\n",Ecoul);
      printf("exchange-correlation   % .10f\n",Exc);
      printf("total energy           % .10f\n",Etot);
    }

    OpenOrbitalOptimizer::FockMatrix<double> fock(2*lmax+2);
    for(int l=0;l<=lmax;l++) {
      fock[l] = T + l*(l+1)*Tl + Vnuc + Vunif + J;
      if(x_func>0 || c_func>0)
        fock[l] += XCa.slice(l);
      fock[l] = Sinvh.t() * fock[l] * Sinvh;

      fock[l+lmax+1] = T + l*(l+1)*Tl + Vnuc + Vunif + J;
      if(x_func>0 || c_func>0)
        fock[l+lmax+1] += XCb.slice(l);
      fock[l+lmax+1] = Sinvh.t() * fock[l+lmax+1] * Sinvh;
    }
    return std::make_pair(Etot,fock);
  };

  std::function<arma::mat(const OpenOrbitalOptimizer::DensityMatrix<double, double> &)> radial_density_matrix = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;
    // Radial density matrix
    arma::mat Prad(Sinvh.n_rows, Sinvh.n_rows, arma::fill::zeros);
    for(int l=0;l<=lmax;l++) {
      // Nothing to do
      if(arma::max(arma::abs(occupations[l]))==0.0)
        continue;

      // Same radial basis for all l!
      arma::mat C = Sinvh*orbitals[l];
      arma::mat P = C*arma::diagmat(occupations[l])*C.t();
      Prad += P;
    }
    return Prad;
  };

  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double, double> &, const std::string &)> save_density = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const std::string & fname) {
    auto Prad = radial_density_matrix(dm);
    // Remove the angular factor
    Prad /= 4.0*M_PI;

    arma::vec r(basis.radii());
    arma::vec density(basis.electron_density(Prad, false));
    size_t Npoints(r.n_elem);
    // and pack it for libxc
    arma::mat rho_arr(Npoints,2);
    rho_arr.col(0)=r;
    rho_arr.col(1)=density;
    rho_arr.save(fname, arma::raw_ascii);
  };

  std::string density_name;
  if(Z-Q==0 and njellium>0)
    density_name = "density_jellium.dat";
  else if(Z-Q>0 and njellium==0)
    density_name = "density_atom.dat";
  else if(Z-Q>0 and njellium>0)
    density_name = "density_aij.dat";
  else
    throw std::logic_error("Nothing to calculate: Z-Q <=0 and njellium == 0!\n");

  // Parse number of spin-up and spin-down electrons
  int nelec = Z-Q+njellium;

  if(M==0) {
    // OOO data
    arma::uvec number_of_blocks_per_particle_type({(arma::uword) (lmax+1)});
    arma::vec maximum_occupation(lmax+1);
    std::vector<std::string> block_descriptions(lmax+1);
    for(int l=0;l<=lmax;l++) {
      maximum_occupation(l) = 2*(2*l+1);

      std::ostringstream oss;
      oss << "l=" << l;
      block_descriptions[l] = oss.str();
    }
    maximum_occupation.t().print("Max occ");

    arma::vec number_of_particles({(double) nelec});

    // Core guess
    OpenOrbitalOptimizer::FockMatrix<double> coreH(lmax+1);
    for(int l=0;l<=lmax;l++)
      coreH[l] = Sinvh.t() * (T + l*(l+1)*Tl + Vnuc + Vunif) * Sinvh;

    OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, restricted_builder, block_descriptions);
    scfsolver.maximum_iterations(maxiter);
    scfsolver.convergence_threshold(convthr);
    scfsolver.initialize_with_fock(coreH);
    if(oda) {
      try {
        scfsolver.run_optimal_damping();
      } catch(...) {};
    } else {
      scfsolver.run();
    }
    save_density(scfsolver.get_solution(), density_name);

  } else {
    int nela=0, nelb=0;
    scf::parse_nela_nelb(nela,nelb,Q,M,Z+njellium);

    // OOO data
    arma::uvec number_of_blocks_per_particle_type({(arma::uword) (lmax+1), (arma::uword) (lmax+1)});
    arma::vec maximum_occupation(2*lmax+2);
    std::vector<std::string> block_descriptions(2*lmax+2);
    for(int l=0;l<=lmax;l++) {
      maximum_occupation(l) = 2*l+1;
      maximum_occupation(l+lmax+1) = 2*l+1;

      std::ostringstream oss;
      oss << "l=" << l;

      block_descriptions[l] = oss.str() + " alpha";
      block_descriptions[l+lmax+1] = oss.str() + " beta";
    }
    maximum_occupation.t().print("Max occ");

    arma::vec number_of_particles({(double) nela, (double) nelb});

    // Core guess
    OpenOrbitalOptimizer::FockMatrix<double> coreH(2*lmax+2);
    for(int l=0;l<=lmax;l++) {
      coreH[l] = Sinvh.t() * (T + l*(l+1)*Tl + Vnuc + Vunif) * Sinvh;
      coreH[l+lmax+1] = coreH[l];
    }

    OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, unrestricted_builder, block_descriptions);
    scfsolver.maximum_iterations(maxiter);
    scfsolver.convergence_threshold(convthr);
    scfsolver.initialize_with_fock(coreH);
    if(oda) {
      try {
        scfsolver.run_optimal_damping();
      } catch(...) {};
    } else {
      scfsolver.run();
    }
    save_density(scfsolver.get_solution(), density_name);
  }

  return 0;
}
