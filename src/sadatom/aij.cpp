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
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file at the root of this source distribution
 * for the full license text.
 */
#include "../general/cmdline.h"
#include "../general/constants.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/eigen_io.h"
#include "../general/scf_driver_common.h"
#include "../general/trustregion_scf.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include "basis.h"
#include "dftgrid.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cfloat>
#include <fstream>
#include <sstream>

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
  // Solver-facing scalar type. The sadatom chemistry layer (TwoDBasis,
  // DFTGrid, libxc) is double-only, so this is not yet wired beyond
  // double; the libhelfem radial machinery underneath is fully
  // templated to eps(T).
  using OOO_Real = double;

  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<double>("nufreq", 0, "frequency of uniform elements", false, 0.5);
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 8);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 5);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");
  parser.add<int>("njellium", 0,"number of jellium electrons", true);
  parser.add<double>("rs", 0, "Wigner-Seitz radius for jellium", true);
  parser.add<double>("convthr", 0, "Convergence threshold", false, 1e-7);
  parser.add<double>("savethr", 0, "Threshold for nonzero occupation", false, 1e-6);
  parser.add<bool>("vacancy", 0, "Jellium vacancy model?", false, false);
  parser.add<int>("maxiter", 0, "maximum number of iterations", false, 1024);
  parser.add<bool>("zeroright", 0, "Zero the right-hand function value", false, false);
  parser.add<double>("Rmax", 0, "Size of vacuum region", false, 40.0);
  parser.add<int>("M", 0, "spin multiplicity", true);
  parser.add<int>("lmax", 0, "maximum angular momentum", false, 4);
  parser.add<std::string>("scfmethods", 0, "SCF convergence methods: '+' separated subset of DIIS, ODA, CG, LBFGS", false, "DIIS + ODA + CG");
  parser.add<std::string>("loadfock", 0, "file to load guess fock matrix from", false, "");
  parser.add<std::string>("savefock", 0, "file to save fock matrix to", false, "");
  parser.add<bool>("saveorb", 0, "save radial orbitals to disk?", false, false);
  parser.add<int>("verbosity", 0, "output detail: 0 silent, 1 setup and energies, 5 also per-iteration diagnostics and Fock timings; also passed to the SCF solver", false, 5);
  parser.add<bool>("secondorder", 0, "follow the first-order SCF with second-order trust-region optimization of the orbitals and the fractional occupations", false, false);
  parser.add<int>("preiter", 0, "first-order iterations to run before switching to the second-order optimizer", false, 100);
  parser.add<double>("soconvthr", 0, "second-order convergence threshold on the RMS gradient", false, 1e-8);
  parser.add<int>("somacro", 0, "maximum second-order macroiterations", false, 150);
  parser.add<int>("somicro", 0, "maximum second-order microiterations", false, 50);
  parser.add<double>("soredfac", 0, "how far the second-order microiterations must reduce the residual before a step is accepted; a step whose subproblem misses this is rejected however good it is, so tightening it can stall an ill-conditioned case rather than sharpen it", false, 3e-1);
  parser.add<int>("somaxhess", 0, "ceiling on Hessian-vector products in one second-order solve; 0 uses somacro*somicro, which only a runaway step-rejection loop can reach", false, 0);
  parser.add<std::string>("sosolver", 0, "second-order microiteration solver: davidson, jacobi-davidson or tcg", false, "davidson");
  parser.add<bool>("soprecond", 0, "precondition with the exactly built Hessian subspace; off falls back to OpenTrustRegion's diagonal, which is blind to the occupation coupling", false, true);
  parser.add<double>("sotest", 0, "instead of optimizing, check the analytic gradient and Hessian against finite differences with this step size", false, 0.0);
  parser.parse_check(argc, argv);

  // Get parameters
  int igrid(parser.get<int>("grid"));
  double zexp(parser.get<double>("zexp"));
  int Nelem(parser.get<int>("nelem"));
  double Nufreq(parser.get<double>("nufreq"));

  int Z(element_Z(parser.get<std::string>("Z")));
  int Q(parser.get<int>("Q"));
  int M(parser.get<int>("M"));

  int Nnodes(parser.get<int>("nnodes"));
  int Nquad(parser.get<int>("nquad"));
  const int verbosity(parser.get<int>("verbosity"));
  // The shared basis/grid/functional code reports through this flag.
  helfem::set_verbosity(verbosity >= 1);
  std::string method(parser.get<std::string>("method"));
  double dftthr(parser.get<double>("dftthr"));

  int primbas(parser.get<int>("primbas"));
  std::string xparf(parser.get<std::string>("x_pars"));
  std::string cparf(parser.get<std::string>("c_pars"));

  int njellium(parser.get<int>("njellium"));
  double rs(parser.get<double>("rs"));
  double Rmax(parser.get<double>("Rmax"));
  bool vacancy(parser.get<bool>("vacancy"));

  int maxiter(parser.get<int>("maxiter"));
  double convthr(parser.get<double>("convthr"));
  double savethr(parser.get<double>("savethr"));

  bool zeroright(parser.get<bool>("zeroright"));
  int lmax(parser.get<int>("lmax"));
  std::string scfmethods(parser.get<std::string>("scfmethods"));

  std::string loadfock(parser.get<std::string>("loadfock"));
  std::string savefock(parser.get<std::string>("savefock"));
  bool saveorb(parser.get<bool>("saveorb"));

  bool secondorder(parser.get<bool>("secondorder"));
  int preiter(parser.get<int>("preiter"));
  double soconvthr(parser.get<double>("soconvthr"));
  int somacro(parser.get<int>("somacro"));
  int somicro(parser.get<int>("somicro"));
  int somaxhess(parser.get<int>("somaxhess"));
  double soredfac(parser.get<double>("soredfac"));
  std::string sosolver(parser.get<std::string>("sosolver"));
  bool soprecond(parser.get<bool>("soprecond"));
  double sotest(parser.get<double>("sotest"));
  if(sotest > 0.0)
    secondorder = true;
  if(secondorder && !helfem::otr::available())
    throw std::logic_error("This HelFEM was built without OpenTrustRegion, so --secondorder is unavailable. Configure with -DHELFEM_OPENTRUSTREGION=ON.\n");

  // Parse xc parameters
  helfem::Vector x_pars, c_pars;
  if(xparf.size()) {
    x_pars = scf::parse_xc_params(xparf);
    helfem::io::print_matrix("Exchange functional parameters", helfem::Matrix(x_pars.transpose()));
  }
  if(cparf.size()) {
    c_pars = scf::parse_xc_params(cparf);
    helfem::io::print_matrix("Correlation functional parameters", helfem::Matrix(c_pars.transpose()));
  }

  // Get primitive basis
  auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::make_basis(primbas,Nnodes)));

  if(Nquad==0)
    // Set default value
    Nquad=5*poly->nbf();
  else if(Nquad<2*poly->nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  // Order of quadrature rule
  if(verbosity >= 1)
    printf("Using %i point quadrature rule.\n",Nquad);

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
    r_inner = cbrt(Z-Q)*rs;
    // The background occupies the shell [r_inner, r_outer] at the
    // uniform density 3/(4 pi rs^3), so its charge is
    // (r_outer/rs)^3 - (r_inner/rs)^3. Requiring that to equal njellium
    // gives r_outer^3 = (njellium + Z - Q) rs^3, i.e. r_outer = R.
    r_outer = R;
  } else {
    // cavity size is determined by number of jellium electrons and the background density
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
    if(verbosity >= 1)
      printf("%i jellium electrons with rs = % .3f and vacancy model leads to r_inner = % .10f r_outer = % .10f lmax = %i\n",njellium,rs,r_inner,r_outer,lmax);
  } else {
    if(verbosity >= 1)
      printf("%i jellium electrons with rs = % .3f leads to R = % .10f lmax = %i\n",njellium,rs,R,lmax);
  }
  // The background charge follows from the shell geometry; report it so
  // a mismatch with njellium is visible rather than silent.
  if(rs > 0) {
    const double bgcharge = std::pow(r_outer/rs,3) - std::pow(r_inner/rs,3);
    if(verbosity >= 1)
      printf("Background charge is % .10f, should be %i\n", bgcharge, njellium);
    if(std::abs(bgcharge - njellium) > 1e-8*std::max(1,njellium))
      throw std::logic_error("Background charge does not match the number of jellium electrons!\n");
  }
  if(verbosity >= 1)
    printf("Friedel period is % .3f, using %i uniform elements.\n", friedel_period, Nuelem);
  // Self-repulsion of the positive background. The background is the
  // shell [r_inner, r_outer], i.e. ball(r_outer) minus ball(r_inner) at
  // the same density, so its self-energy is E_out + E_in - W with
  // E_X = (3/5) Q_X^2 / X the ball self-energies and
  // W = (3 Q_in Q_out / 2 r_outer) (1 - r_inner^2 / 5 r_outer^2) their
  // mutual interaction. With Q_X = (X/rs)^3 this collapses to the
  // expression below, which reduces to the full-sphere 3 R^5/(5 rs^6)
  // when r_inner = 0.
  double Erep = (rs > 0 ) ? (3*std::pow(r_outer,5) + 4.5*std::pow(r_inner,5)
                             - 7.5*std::pow(r_inner,3)*std::pow(r_outer,2))
                            / (5*std::pow(rs,6)) : 0.0;

  // Uniform part of grid
  helfem::Vector bval_unif;
  if(Nuelem)
    bval_unif = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nuelem, R, 1, 0.0, 0, 0, 0, Z, 0, 0, 0.0, false, 0.0);

  helfem::Vector bval;
  if(Nelem>0) {
    // Atomic grid
    double rinfty = (Nuelem>0) ? bval_unif(1) : Rmax;
    helfem::Vector bval_atom = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nelem, rinfty, igrid, zexp, 0, 0, 0.0, Z, 0, 0, 0.0, false, 0.0);

    // Glue grids together
    if(bval_unif.size()) {
      bval = helfem::Vector::Zero(bval_atom.size()+bval_unif.size()-2);
      bval.head(bval_atom.size()) = bval_atom;
      if(bval_atom(bval_atom.size()-1) != bval_unif(1)) {
        std::ostringstream oss;
        oss << "Grids don't coincide: difference " << bval_atom(bval_atom.size()-1) - bval_unif(1) << "!\n";
        throw std::logic_error(oss.str());
      }
      if(bval_unif.size()>2) {
        bval.tail(bval_unif.size()-2) = bval_unif.segment(2,bval_unif.size()-2);
      }
    } else {
      bval = bval_atom;
    }
  } else {
    bval=bval_unif;
  }

  // Handle vacancy case
  if(vacancy) {
    helfem::Vector vbval(bval.size()+1);
    vbval.head(bval.size())=bval;
    vbval(bval.size()) = r_inner;
    std::sort(vbval.data(), vbval.data()+vbval.size());
    bval = vbval;
  }

  // Add vacuum region
  if(Nuelem>0 and Rmax>0.0) {
    int Nvelem = std::ceil(Rmax/friedel_period*Nufreq);

    // Points in vacuum region
    helfem::Vector bval_vac = atomic::basis::form_grid(modelpotential::POINT_NUCLEUS, 0.0, Nvelem, Rmax, 1, 0.0, 0, 0, 0, Z, 0, 0, 0.0, false, 0.0);
    // offset
    bval_vac.array() += bval(bval.size()-1);

    helfem::Vector bval_new(bval.size()+bval_vac.size()-1);
    bval_new.head(bval.size()) = bval;
    bval_new.tail(bval_vac.size()-1) = bval_vac.segment(1,bval_vac.size()-1);
    if(bval_new(bval.size()-1) != bval_vac(0)) {
      std::ostringstream oss;
      oss << "Grids don't coincide: difference " << bval_new(bval.size()-1) - bval_vac(0) << "!\n";
      throw std::logic_error(oss.str());
    }
    bval=bval_new;
  }
  if(verbosity >= 1)
    helfem::io::print_matrix("Final grid for calculation", helfem::Matrix(bval));

  bool zeroder = false;
  auto basis = sadatom::basis::TwoDBasis(Z, modelpotential::POINT_NUCLEUS, 0.0, poly, zeroder, Nquad, bval, lmax, zeroright);
  if(verbosity >= 1)
    printf("Basis set has %i radial functions\n",(int) basis.Nbf());

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

  // The background potential has kinks at the sphere radii
  std::vector<double> potential_breakpoints;
  if(r_inner>0) potential_breakpoints.push_back(r_inner);
  if(r_outer>0) potential_breakpoints.push_back(r_outer);

  // Energy of nucleus in external field
  double Enucfield = -Z*potfunc(0);
  if(verbosity >= 1)
    printf("potfunc(0) = %e Enucfield = %e\n",potfunc(0),Enucfield);
  fflush(stdout);

  // Form overlap matrix
  helfem::Matrix S=basis.overlap();
  // Get half-inverse
  helfem::Matrix Sinvh=basis.Sinvh();
  // Form kinetic energy matrix
  helfem::Matrix T=basis.kinetic();
  // Form kinetic energy matrix
  helfem::Matrix Tl=basis.kinetic_l();
  // Form nuclear attraction energy matrix
  helfem::Matrix Vnuc=basis.nuclear();
  // Uniform background potential
  helfem::Matrix Vunif=basis.potential(potfunc, potential_breakpoints);

  // Form DFT grid
  auto grid = helfem::sadatom::dftgrid::DFTGrid(&basis);
  // Compute two-electron integrals
  basis.compute_tei();

  // Jellium Hamiltonian
  OpenOrbitalOptimizer::FockMatrix<OOO_Real> H_jellium(lmax+1);
  for(int l=0;l<=lmax;l++) {
    // Uniform potential is cancelled out by the jellium density
    H_jellium[l] = Sinvh.transpose() * (T + l*(l+1)*Tl) * Sinvh;
  }

  // Compute the jellium eigenvalues
  std::vector<helfem::Matrix> Cjellium(lmax+1);
  std::vector<helfem::Vector> Ejellium(lmax+1);
  std::vector<std::tuple<double,int,int>> jellium_energies;
  for(int l=0;l<=lmax;l++) {
    Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(H_jellium[l]);
    Ejellium[l] = es.eigenvalues();
    // Convert to non-orthogonal basis
    Cjellium[l] = Sinvh*es.eigenvectors();

    if(verbosity >= 1)
      printf("l = %i eigenvalues\n",l);
    if(verbosity >= 1)
      helfem::io::print_matrix("", helfem::Matrix(Ejellium[l].transpose()));

    for(Eigen::Index io=0;io<Ejellium[l].size();io++)
      jellium_energies.push_back(std::make_tuple(Ejellium[l](io),l,(int) io));
  }
  std::sort(jellium_energies.begin(), jellium_energies.end(), [](std::tuple<double,int,int> const & t1, std::tuple<double,int,int> const & t2) {
    return std::get<0>(t1) < std::get<0>(t2);
  });

  const Eigen::Index Nrad = Sinvh.rows();
  const double angfac = 4.0*M_PI;

  // Divide each slice of a cube by a scalar (helfem::Cube has no
  // whole-object arithmetic)
  auto divided_cube = [](const helfem::Cube & C, double f) {
    helfem::Cube out(C.size());
    for (size_t l = 0; l < C.size(); ++l) out[l] = C[l] / f;
    return out;
  };

  // Fock builder
  // Per-component wall clock for the Fock build, plus the time spent
  // outside it in the SCF solver. Shared by both builders: only one of
  // them is ever handed to the solver.
  helfem::scf_driver::FockTimer ftimer;

  // Energy and per-block Fock matrices in the FEM basis, given the
  // per-block FEM densities. This is the whole energy expression, lifted
  // out of the OpenOrbitalOptimizer builders below so that the
  // second-order optimizer -- which works from densities rather than from
  // orbitals and occupations -- shares it instead of restating it.
  //
  // Blocks are l = 0..lmax for the restricted case, and alpha l = 0..lmax
  // followed by beta l = 0..lmax for the unrestricted one, matching the
  // block layout OpenOrbitalOptimizer is given.
  auto fock_fem = [&](const helfem::Cube & P, helfem::Cube & F, bool report) -> double {
    const bool unrestricted = ((int) P.size() == 2*lmax+2);
    if(!unrestricted && (int) P.size() != lmax+1)
      throw std::logic_error("Density has an unexpected number of blocks.\n");

    ftimer.enter();
    helfem::scf_driver::FockTimer::Components tc;
    Timer tcomp;

    // Kinetic energy and total radial density
    double Ekin=0.0;
    helfem::Matrix Prad = helfem::Matrix::Zero(Nrad, Nrad);
    for(size_t b=0;b<P.size();b++) {
      const int l = (int)(b % (lmax+1));
      Prad += P[b];
      Ekin += (P[b]*T).trace() + l*(l+1)*(P[b]*Tl).trace();
    }
    tc.density += tcomp.get();

    double Enuc=(Prad*Vnuc).trace();
    double Eunif=(Prad*Vunif).trace();

    // Coulomb matrix
    tcomp.set();
    helfem::Matrix J(basis.coulomb(Prad/angfac));
    tc.coulomb += tcomp.get();

    double Exc=0.0;
    helfem::Cube XCa, XCb;
    double nelnum = 0.0;
    if(x_func > 0 || c_func > 0) {
      tcomp.set();
      if(unrestricted) {
        helfem::Cube Pa(P.begin(), P.begin()+lmax+1);
        helfem::Cube Pb(P.begin()+lmax+1, P.end());
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, divided_cube(Pa,angfac), divided_cube(Pb,angfac), XCa, XCb, Exc, nelnum, true, dftthr);
        for(size_t l=0;l<XCb.size();l++) XCb[l]/=angfac;
      } else {
        grid.eval_Fxc(x_func, x_pars, c_func, c_pars, divided_cube(P,angfac), XCa, Exc, nelnum, dftthr);
      }
      for(size_t l=0;l<XCa.size();l++) XCa[l]/=angfac;
      tc.xc += tcomp.get();
      if(report && verbosity >= 5) {
        printf("DFT energy %.10e\n",Exc);
        printf("Error in integrated number of electrons % e\n",nelnum-(Z-Q+njellium));
        fflush(stdout);
      }
    }

    double Ecoul = 0.5*(Prad*J).trace();
    double Etot = Ekin + Enuc + Enucfield + Eunif + Erep + Ecoul + Exc;

    if(report && verbosity >= 1) {
      printf("kinetic energy         % .10f\n",Ekin);
      printf("nuclear attraction     % .10f\n",Enuc);
      printf("nucleus-field term     % .10f\n",Enucfield);
      printf("background repulsion   % .10f\n",Erep);
      printf("background attraction  % .10f\n",Eunif);
      printf("Coulomb repulsion      % .10f\n",Ecoul);
      printf("exchange-correlation   % .10f\n",Exc);
      printf("total energy           % .10f\n",Etot);
    }

    F.assign(P.size(), helfem::Matrix());
    for(size_t b=0;b<P.size();b++) {
      const int l = (int)(b % (lmax+1));
      F[b] = T + l*(l+1)*Tl + Vnuc + Vunif + J;
      if(x_func>0 || c_func>0)
        F[b] += (unrestricted && b > (size_t) lmax) ? XCb[l] : XCa[l];
    }

    tc.total = ftimer.build_elapsed();
    ftimer.add_build(tc);
    if(report && verbosity >= 5) ftimer.print_build(x_func > 0 || c_func > 0, false);
    ftimer.leave();
    return Etot;
  };

  // Linear response of those Fock matrices to a batch of density
  // perturbations. The Coulomb half is linear, so its response is exact
  // and costs one more Coulomb build; the XC half is the density-density
  // kernel block, exact for an LDA and a deliberate approximation beyond
  // it (see DFTGridWorkerBase::compute_fxc). The batch is passed through
  // to the grid, which shares the basis values, the reference density and
  // the kernel evaluation across all of the perturbations.
  auto response_fem = [&](const helfem::Cube & P, const std::vector<helfem::Cube> & dP, std::vector<helfem::Cube> & dF) {
    const bool unrestricted = ((int) P.size() == 2*lmax+2);
    const size_t nt = dP.size();
    dF.assign(nt, helfem::Cube());

    std::vector<helfem::Cube> dXCa, dXCb;
    if(x_func > 0 || c_func > 0) {
      if(unrestricted) {
        helfem::Cube Pa(P.begin(), P.begin()+lmax+1);
        helfem::Cube Pb(P.begin()+lmax+1, P.end());
        std::vector<helfem::Cube> dPa(nt), dPb(nt);
        for(size_t it=0;it<nt;it++) {
          dPa[it] = divided_cube(helfem::Cube(dP[it].begin(), dP[it].begin()+lmax+1), angfac);
          dPb[it] = divided_cube(helfem::Cube(dP[it].begin()+lmax+1, dP[it].end()), angfac);
        }
        grid.eval_Fxc_response(x_func, x_pars, c_func, c_pars, divided_cube(Pa,angfac), divided_cube(Pb,angfac), dPa, dPb, dXCa, dXCb, dftthr);
        for(size_t it=0;it<nt;it++)
          for(size_t l=0;l<dXCb[it].size();l++) dXCb[it][l]/=angfac;
      } else {
        std::vector<helfem::Cube> dPs(nt);
        for(size_t it=0;it<nt;it++) dPs[it] = divided_cube(dP[it], angfac);
        grid.eval_Fxc_response(x_func, x_pars, c_func, c_pars, divided_cube(P,angfac), dPs, dXCa, dftthr);
      }
      for(size_t it=0;it<nt;it++)
        for(size_t l=0;l<dXCa[it].size();l++) dXCa[it][l]/=angfac;
    }

    for(size_t it=0;it<nt;it++) {
      helfem::Matrix dPrad = helfem::Matrix::Zero(Nrad, Nrad);
      for(size_t b=0;b<dP[it].size();b++)
        dPrad += dP[it][b];
      const helfem::Matrix dJ(basis.coulomb(dPrad/angfac));

      dF[it].assign(P.size(), helfem::Matrix());
      for(size_t b=0;b<P.size();b++) {
        const int l = (int)(b % (lmax+1));
        dF[it][b] = dJ;
        if(x_func>0 || c_func>0)
          dF[it][b] += (unrestricted && b > (size_t) lmax) ? dXCb[it][l] : dXCa[it][l];
      }
    }
  };

  // Build the per-block FEM densities of an OpenOrbitalOptimizer solution
  auto blocked_density = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;
    helfem::Cube P(orbitals.size(), helfem::Matrix::Zero(Nrad, Nrad));
    for(size_t b=0;b<orbitals.size();b++) {
      if(occupations[b].cwiseAbs().maxCoeff()==0.0)
        continue;
      // Same radial basis for all l!
      const helfem::Matrix C = Sinvh*orbitals[b];
      P[b] = C*occupations[b].asDiagonal()*C.transpose();
    }
    return P;
  };

  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> restricted_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    helfem::Cube F;
    const double Etot = fock_fem(blocked_density(dm), F, true);
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(lmax+1);
    for(int l=0;l<=lmax;l++)
      fock[l] = Sinvh.transpose() * F[l] * Sinvh;
    return std::make_pair(Etot,fock);
  };

  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<OOO_Real, OOO_Real> unrestricted_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    helfem::Cube F;
    const double Etot = fock_fem(blocked_density(dm), F, true);
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> fock(2*lmax+2);
    for(int b=0;b<2*lmax+2;b++)
      fock[b] = Sinvh.transpose() * F[b] * Sinvh;
    return std::make_pair(Etot,fock);
  };

  // The same two builders, in the form the second-order optimizer wants:
  // densities in, energy and Fock matrices out, with no reporting -- the
  // trust-region solver prints its own iteration table, and the energy
  // decomposition is printed once at the end instead of on every one of
  // the many objective evaluations a microiteration sweep makes.
  helfem::trscf::FockBuilder so_fock = [&](const helfem::Cube & P, helfem::Cube & F) {
    return fock_fem(P, F, false);
  };
  helfem::trscf::ResponseBuilder so_response = [&](const helfem::Cube & P, const std::vector<helfem::Cube> & dP, std::vector<helfem::Cube> & dF) {
    response_fem(P, dP, dF);
  };

  // Drive the second-order phase from a converged (or partly converged)
  // first-order solution, and hand back the refined orbitals and
  // occupations. Shared by the restricted and unrestricted branches; they
  // differ only in how the blocks divide between particle types.
  auto second_order_phase = [&](OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm,
                                const Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> & maximum_occupation,
                                const std::vector<size_t> & blocks_per_particle) {
    std::vector<double> maxocc(maximum_occupation.size());
    for(Eigen::Index i=0;i<maximum_occupation.size();i++)
      maxocc[(size_t) i] = maximum_occupation(i);

    helfem::trscf::Optimizer opt(Sinvh, maxocc, blocks_per_particle, so_fock, so_response);
    opt.set_reference(dm.first, dm.second);

    if(sotest > 0.0) {
      // Derivative check only: the analytic gradient and Hessian are the
      // whole content of the second-order method, and nothing downstream
      // can tell a wrong Hessian from a hard problem.
      // The response kernel is the true one only for an LDA; beyond that
      // the Hessian is deliberately approximate, so it is measured rather
      // than tested.
      bool xg=false, xt=false, xl=false, cg=false, ct=false, cl=false;
      if(x_func > 0) ::is_gga_mgga(x_func, xg, xt, xl);
      if(c_func > 0) ::is_gga_mgga(c_func, cg, ct, cl);
      const bool exact_kernel = !(xg||xt||xl||cg||ct||cl);
      if(!opt.verify(sotest, 1e-4, verbosity, exact_kernel))
        throw std::runtime_error("The analytic derivatives disagree with finite differences.\n");
      return;
    }

    helfem::trscf::Options sopt;
    sopt.otr.conv_tol = soconvthr;
    sopt.otr.n_macro = somacro;
    sopt.otr.n_micro = somicro;
    sopt.max_hessian = somaxhess;
    sopt.otr.global_red_factor = soredfac;
    sopt.otr.local_red_factor = 0.1*soredfac;
    sopt.otr.subsystem_solver = sosolver;
    // OpenTrustRegion prints its iteration table at level 3.
    sopt.otr.verbose = (verbosity >= 1) ? std::max(3, verbosity) : 0;
    sopt.exact_precond = soprecond;
    sopt.verbosity = verbosity;

    const helfem::trscf::Result res = opt.run(sopt);
    dm = std::make_pair(opt.orbitals(), opt.occupations());

    // Announce convergence the way every other driver does. Which optimizer
    // reached the solution is not the reader's problem, and it is not the
    // test harness's either: a run with no such line is rejected as
    // non-converged, which is exactly right, and is why this is printed
    // only when the gradient really did come down.
    if(res.converged)
      printf("Converged to energy %.10f!\n", res.energy);

    if(verbosity >= 1) {
      printf("\nSecond-order optimization reached energy %.10f with RMS gradient %.3e\n",
             res.energy, res.grad_rms);
      printf("  %i reference updates, %i objective evaluations, %i Hessian-vector "
             "products in %i response builds\n",
             (int) res.n_update, (int) res.n_objective, (int) res.n_hessian,
             (int) res.n_response);
      fflush(stdout);
    }
  };

  std::function<helfem::Matrix(const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> &)> radial_density_matrix = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm) {
    // Every block, which for an unrestricted calculation means alpha AND
    // beta. Summing only l = 0..lmax, as this did, is the alpha density
    // alone -- half of what the saved density file is supposed to hold.
    const helfem::Cube P = blocked_density(dm);
    helfem::Matrix Prad = helfem::Matrix::Zero(Nrad, Nrad);
    for(size_t b=0;b<P.size();b++)
      Prad += P[b];
    return Prad;
  };

  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> &, const std::string &)> save_density = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm, const std::string & fname) {
    helfem::Matrix Prad = radial_density_matrix(dm);
    // Remove the angular factor
    Prad /= angfac;

    helfem::Vector r(basis.radii());
    helfem::Vector density(basis.electron_density(Prad, false));
    Eigen::Index Npoints(r.size());
    // and pack it for libxc
    helfem::Matrix rho_arr(Npoints,2);
    rho_arr.col(0)=r;
    rho_arr.col(1)=density;
    helfem::io::write_raw_ascii(fname, rho_arr);
  };

  // Fock matrices are saved as ASCII: a "nblocks nrows ncols" header
  // followed by the stacked blocks, one matrix row per line.
  std::function<void(const OpenOrbitalOptimizer::FockMatrix<OOO_Real> &, const std::string &)> save_fock = [&](const OpenOrbitalOptimizer::FockMatrix<OOO_Real> & fock, const std::string & fname) {
    std::ofstream out(fname);
    if(!out)
      throw std::runtime_error("Could not open " + fname + " for writing.\n");
    out << fock.size() << " " << fock[0].rows() << " " << fock[0].cols() << "\n";
    out.precision(17);
    out.setf(std::ios::scientific);
    for(size_t i=0;i<fock.size();i++)
      for(Eigen::Index r=0;r<fock[i].rows();r++) {
        for(Eigen::Index c=0;c<fock[i].cols();c++)
          out << " " << fock[i](r,c);
        out << "\n";
      }
  };

  std::function<helfem::Cube(const std::string &)> load_fock = [&](const std::string & fname) {
    std::ifstream in(fname);
    if(!in)
      throw std::runtime_error("Could not open " + fname + " for reading.\n");
    size_t nblocks;
    Eigen::Index nrows, ncols;
    in >> nblocks >> nrows >> ncols;
    helfem::Cube fockmat(nblocks, helfem::Matrix(nrows,ncols));
    for(size_t i=0;i<nblocks;i++)
      for(Eigen::Index r=0;r<nrows;r++)
        for(Eigen::Index c=0;c<ncols;c++)
          in >> fockmat[i](r,c);
    if(!in)
      throw std::runtime_error("Error reading Fock matrix from " + fname + ".\n");
    return fockmat;
  };

  // Diagonalize a Fock matrix block to obtain orbital energies and orbital
  // coefficients in the non-orthonormal basis.
  auto diagonalize_blocks = [&](const OpenOrbitalOptimizer::FockMatrix<OOO_Real> & fock,
                                std::vector<helfem::Vector> & Eblock,
                                std::vector<helfem::Matrix> & Cblock) {
    Eblock.resize(fock.size());
    Cblock.resize(fock.size());
    for(size_t b=0; b<fock.size(); b++) {
      helfem::Matrix fsym = 0.5*(fock[b] + fock[b].transpose());
      Eigen::SelfAdjointEigenSolver<helfem::Matrix> es(fsym);
      Eblock[b] = es.eigenvalues();
      Cblock[b] = Sinvh * es.eigenvectors();
    }
  };

  // Print orbital information (occupation, energy, <r>, r(max)) for the
  // requested range of blocks. The block index modulo lmax+1 gives l.
  auto print_orbitals = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm,
                            const std::vector<helfem::Vector> & Eblock,
                            const std::vector<helfem::Matrix> & Cblock,
                            const std::vector<std::string> & block_descriptions,
                            size_t bstart, size_t bend) {
    static const char shtype[] = "spdfgh";
    const auto & occupations = dm.second;

    std::vector< std::pair<int, helfem::Matrix> > rmat(basis.Rmatrices());

    for(size_t b=bstart; b<bend; b++) {
      int l = (int)(b % (lmax+1));

      printf("\n%s orbitals\n", block_descriptions[b].c_str());
      printf("%3s %8s %16s","nl","occ","E");
      for(size_t ir=0;ir<rmat.size();ir++) {
        std::ostringstream oss;
        oss << "<r>(" << rmat[ir].first << ")";
        printf(" %12s",oss.str().c_str());
      }
      printf(" %12s\n","r(max)");

      for(Eigen::Index io=0;io<Eblock[b].size();io++) {
        double occ = occupations[b](io);
        if(std::abs(occ) < savethr)
          continue;

        helfem::Vector orb = Cblock[b].col(io);
        helfem::Matrix P = orb*orb.transpose();

        int n = (int)io + l + 1;
        char ltag = (l < 6) ? shtype[l] : '?';
        printf("%2i%c % 8.4f % 16.9f", n, ltag, occ, Eblock[b](io));
        for(size_t ir=0;ir<rmat.size();ir++) {
          double rpos = std::pow((P*rmat[ir].second).trace(), 1.0/rmat[ir].first);
          printf(" %12e", rpos);
        }
        printf(" %12e\n", basis.electron_density_maximum_radius(P));
      }
    }
  };

  // Save the radial values of all occupied orbitals from the requested block
  // range to {prefix}_orbs.dat together with their energies, occupations and
  // angular momenta. Companion files store the first and second derivatives.
  auto save_orbitals = [&](const OpenOrbitalOptimizer::DensityMatrix<OOO_Real, OOO_Real> & dm,
                           const std::vector<helfem::Vector> & Eblock,
                           const std::vector<helfem::Matrix> & Cblock,
                           size_t bstart, size_t bend,
                           const std::string & prefix) {
    const auto & occupations = dm.second;

    helfem::Vector r(basis.radii());
    helfem::Vector wt(basis.quadrature_weights());

    // Collect occupied orbital indices
    std::vector< std::vector<Eigen::Index> > occ_idx(bend-bstart);
    size_t norb = 0;
    for(size_t b=bstart; b<bend; b++) {
      for(Eigen::Index io=0; io<occupations[b].size(); io++) {
        if(std::abs(occupations[b](io)) > savethr) {
          occ_idx[b-bstart].push_back(io);
          norb++;
        }
      }
    }

    // Evaluate orbitals, derivatives and second derivatives
    std::vector<helfem::Matrix> orbval(bend-bstart), orbdval(bend-bstart), orblval(bend-bstart);
    for(size_t b=bstart; b<bend; b++) {
      const auto & idx = occ_idx[b-bstart];
      if(idx.empty())
        continue;
      helfem::Matrix Cl(Cblock[b].rows(), idx.size());
      for(size_t i=0;i<idx.size();i++)
        Cl.col(i) = Cblock[b].col(idx[i]);
      orbval[b-bstart]  = basis.orbitals(Cl);
      orbdval[b-bstart] = basis.orbitals_derivative(Cl);
      orblval[b-bstart] = basis.orbitals_second_derivative(Cl);

      // Fix the phases: the largest density value should be at a positive amplitude
      for(Eigen::Index io=0; io<orbval[b-bstart].cols(); io++) {
        Eigen::Index imax;
        orbval[b-bstart].col(io).cwiseAbs2().maxCoeff(&imax);
        if(orbval[b-bstart](imax,io) < 0.0) {
          orbval[b-bstart].col(io)  *= -1;
          orbdval[b-bstart].col(io) *= -1;
          orblval[b-bstart].col(io) *= -1;
        }
      }
    }

    auto save_data = [&](const std::string & fname, const std::vector<helfem::Matrix> & data) {
      FILE *out = fopen(fname.c_str(),"w");
      if(!out) {
        fprintf(stderr,"Failed to open %s for writing\n", fname.c_str());
        return;
      }
      // Header: number of radial points and orbitals
      fprintf(out,"%i %i\n",(int) r.size(),(int) norb);
      // Orbital angular momenta
      for(size_t b=bstart; b<bend; b++) {
        int l = (int)(b % (lmax+1));
        for(size_t i=0; i<occ_idx[b-bstart].size(); i++)
          fprintf(out," %i", l);
      }
      fprintf(out,"\n");
      // Orbital occupations
      for(size_t b=bstart; b<bend; b++)
        for(Eigen::Index i : occ_idx[b-bstart])
          fprintf(out," %e", occupations[b](i));
      fprintf(out,"\n");
      // Orbital energies
      for(size_t b=bstart; b<bend; b++)
        for(Eigen::Index i : occ_idx[b-bstart])
          fprintf(out," %e", Eblock[b](i));
      fprintf(out,"\n");
      // Radial data
      for(Eigen::Index ir=0; ir<r.size(); ir++) {
        fprintf(out,"%e", r(ir));
        fprintf(out," % e", wt(ir));
        for(size_t b=bstart; b<bend; b++) {
          const auto & block = data[b-bstart];
          for(Eigen::Index ic=0; ic<block.cols(); ic++)
            fprintf(out," % e", block(ir,ic));
        }
        fprintf(out,"\n");
      }
      fclose(out);
    };

    save_data(prefix + "_orbs.dat",      orbval);
    save_data(prefix + "_orbs_der.dat",  orbdval);
    save_data(prefix + "_orbs_2der.dat", orblval);
  };

  std::string density_name, orb_prefix;
  if(Z-Q==0 and njellium>0) {
    density_name = "density_jellium.dat";
    orb_prefix = "jellium";
  } else if(Z-Q>0 and njellium==0) {
    density_name = "density_atom.dat";
    orb_prefix = "atom";
  } else if(Z-Q>0 and njellium>0) {
    density_name = "density_aij.dat";
    orb_prefix = "aij";
  } else
    throw std::logic_error("Nothing to calculate: Z-Q <=0 and njellium == 0!\n");

  // Parse number of spin-up and spin-down electrons
  int nelec = Z-Q+njellium;

  if(M==0) {
    // OOO data
    OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(1);
    number_of_blocks_per_particle_type(0) = lmax+1;
    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(lmax+1);
    std::vector<std::string> block_descriptions(lmax+1);
    for(int l=0;l<=lmax;l++) {
      maximum_occupation(l) = 2*(2*l+1);

      std::ostringstream oss;
      oss << "l=" << l;
      block_descriptions[l] = oss.str();
    }
    if(verbosity >= 1)
      helfem::io::print_matrix("Max occ", helfem::Matrix(maximum_occupation.transpose()));

    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(1);
    number_of_particles(0) = nelec;

    // Core guess
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> coreH(lmax+1);
    for(int l=0;l<=lmax;l++)
      coreH[l] = Sinvh.transpose() * (T + l*(l+1)*Tl + Vnuc + Vunif) * Sinvh;
    if(loadfock != "") {
      helfem::Cube fock = load_fock(loadfock);
      if((int) fock.size() == lmax+1 or (int) fock.size() == 2*lmax+2) {
        for(int l=0;l<=lmax;l++)
          coreH[l]=fock[l];
      } else {
        throw std::logic_error("Guess Fock matrix has unexpected angular dimensions!\n");
      }
    }

    OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, restricted_builder, block_descriptions);
    scfsolver.set("verbosity", verbosity);
    // The first-order phase only has to find the occupation pattern -- which
    // shells are fractionally occupied -- and get close enough for a local
    // quadratic model to mean something. Squeezing the last digits out of it
    // is exactly what it is bad at, so it is cut short when the second-order
    // optimizer is going to take over.
    scfsolver.set("maximum_iterations", secondorder ? std::min(maxiter, preiter) : maxiter);
    scfsolver.set("convergence_threshold", convthr);
    scfsolver.set("methods", scfmethods);
    scfsolver.print_citation();
    scfsolver.initialize_with_fock(coreH);
    scfsolver.run();
    if(verbosity >= 5) ftimer.print_summary(x_func > 0 || c_func > 0, false);

    auto dm = std::make_pair(scfsolver.get_orbitals(), scfsolver.get_orbital_occupations());
    auto fock = scfsolver.get_fock_matrix();

    if(secondorder) {
      second_order_phase(dm, maximum_occupation, std::vector<size_t>(1, (size_t)(lmax+1)));
      // Rebuild through the ordinary path, which also prints the energy
      // decomposition of the final solution.
      fock = restricted_builder(dm).second;
    }
    save_density(dm, density_name);
    if(savefock != "") {
      save_fock(fock, savefock);
    }

    {
      std::vector<helfem::Vector> Eblock;
      std::vector<helfem::Matrix> Cblock;
      diagonalize_blocks(fock, Eblock, Cblock);
      if(verbosity >= 1)
        print_orbitals(dm, Eblock, Cblock, block_descriptions, 0, fock.size());
      if(saveorb)
        save_orbitals(dm, Eblock, Cblock, 0, fock.size(), orb_prefix);
    }

  } else {
    int nela=0, nelb=0;
    scf::parse_nela_nelb(nela,nelb,Q,M,Z+njellium);

    // OOO data
    OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(2);
    number_of_blocks_per_particle_type(0) = lmax+1;
    number_of_blocks_per_particle_type(1) = lmax+1;
    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> maximum_occupation(2*lmax+2);
    std::vector<std::string> block_descriptions(2*lmax+2);
    for(int l=0;l<=lmax;l++) {
      maximum_occupation(l) = 2*l+1;
      maximum_occupation(l+lmax+1) = 2*l+1;

      std::ostringstream oss;
      oss << "l=" << l;

      block_descriptions[l] = oss.str() + " alpha";
      block_descriptions[l+lmax+1] = oss.str() + " beta";
    }
    if(verbosity >= 1)
      helfem::io::print_matrix("Max occ", helfem::Matrix(maximum_occupation.transpose()));

    Eigen::Matrix<OOO_Real, Eigen::Dynamic, 1> number_of_particles(2);
    number_of_particles(0) = nela;
    number_of_particles(1) = nelb;

    // Core guess
    OpenOrbitalOptimizer::FockMatrix<OOO_Real> coreH(2*lmax+2);
    for(int l=0;l<=lmax;l++) {
      coreH[l] = Sinvh.transpose() * (T + l*(l+1)*Tl + Vnuc + Vunif) * Sinvh;
      coreH[l+lmax+1] = coreH[l];
    }
    if(loadfock != "") {
      helfem::Cube fock = load_fock(loadfock);
      for(int l=0;l<=lmax;l++) {
        coreH[l]=fock[l];

        if((int) fock.size() == lmax+1) {
          coreH[l+lmax+1] = coreH[l];
        } else if((int) fock.size() == 2*lmax+2) {
          coreH[l+lmax+1] = fock[l+lmax+1];
        } else {
          throw std::logic_error("Guess Fock matrix has unexpected angular dimensions!\n");
        }
      }
    }

    OpenOrbitalOptimizer::SCFSolver<OOO_Real, OOO_Real> scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, unrestricted_builder, block_descriptions);
    scfsolver.set("verbosity", verbosity);
    scfsolver.set("maximum_iterations", secondorder ? std::min(maxiter, preiter) : maxiter);
    scfsolver.set("convergence_threshold", convthr);
    scfsolver.set("methods", scfmethods);
    scfsolver.print_citation();
    scfsolver.initialize_with_fock(coreH);
    scfsolver.run();
    if(verbosity >= 5) ftimer.print_summary(x_func > 0 || c_func > 0, false);

    auto dm = std::make_pair(scfsolver.get_orbitals(), scfsolver.get_orbital_occupations());
    auto fock = scfsolver.get_fock_matrix();

    if(secondorder) {
      // Alpha and beta are separate particle types: their electron counts
      // are conserved separately, so the occupation transfers stay within
      // a spin.
      second_order_phase(dm, maximum_occupation, std::vector<size_t>(2, (size_t)(lmax+1)));
      fock = unrestricted_builder(dm).second;
    }
    save_density(dm, density_name);
    if(savefock != "") {
      save_fock(fock, savefock);
    }

    {
      std::vector<helfem::Vector> Eblock;
      std::vector<helfem::Matrix> Cblock;
      diagonalize_blocks(fock, Eblock, Cblock);
      if(verbosity >= 1)
        printf("\nAlpha orbitals\n");
      if(verbosity >= 1)
        print_orbitals(dm, Eblock, Cblock, block_descriptions, 0, lmax+1);
      if(verbosity >= 1)
        printf("\nBeta orbitals\n");
      if(verbosity >= 1)
        print_orbitals(dm, Eblock, Cblock, block_descriptions, lmax+1, 2*lmax+2);
      if(saveorb) {
        save_orbitals(dm, Eblock, Cblock, 0,        lmax+1,   orb_prefix + "_alpha");
        save_orbitals(dm, Eblock, Cblock, lmax+1,   2*lmax+2, orb_prefix + "_beta");
      }
    }
  }

  return 0;
}
