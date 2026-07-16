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

// Sadatom driver (binary: gensap) -- OpenOrbitalOptimizer based.
// Thin CLI wrapper around helfem::sadatom::scf::run_atomic_scf which
// is also called from src/diatomic/twodquadrature.cpp for the
// atomic-orbital-guess sub-SCF.
//
// HF (x_func == 0 && c_func == 0) is deferred; picking any HF method throws.

#include "../general/cmdline.h"
#include "../general/dftfuncs.h"
#include "../general/constants.h"
#include "../general/elements.h"
#include "../general/scf_helpers.h"
#include "../general/scf_driver_common.h"
#include "../general/lcao.h"
#include "../atomic/basis.h"
#include "scf.h"
#include "../general/eigen_io.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>

using namespace helfem;

// Write a matrix to disk in exactly arma::mat::save(..., arma::raw_ascii)
// format: no header, each element in scientific notation with precision
// 16, right-justified in a field of width 24, preceded by one space.
// Replicating the arma format keeps the emitted .dat files byte-identical
// to the pre-migration output.
// Assemble the radial effective-potential table for the converged
// density in `res`, using exchange/correlation ids (xp_func, cp_func)
// for the xc screening. Mirrors the bespoke SCFSolver::{Restricted,
// Unrestricted}Potential. Columns:
//   0 r   1 rho   2 grad(rho)   3 lapl(rho)   4 tau
//   5 v_coul(screening)   6 v_xc(screening)   7 quad weight
//   8 Zeff = charge - (v_coul + v_xc)   <-- the SAP effective charge
static helfem::Matrix effective_potential_table(
    const helfem::sadatom::basis::TwoDBasis & basis,
    const helfem::sadatom::scf::AtomicSCFResult & res,
    bool restricted, int xp_func, int cp_func) {

  const helfem::Matrix P = res.Prad;
  // Total per-l density cube (for tau). Restricted: Pl_a already holds
  // the full per-l density; unrestricted: sum alpha + beta.
  helfem::Cube Pl = res.Pl_a;
  if (!restricted)
    for (size_t l = 0; l < res.Pl_b.size(); ++l) Pl[l] += res.Pl_b[l];

  const helfem::Vector r    = basis.radii();
  const helfem::Vector wt   = basis.quadrature_weights();
  const helfem::Vector vcoul = basis.coulomb_screening(P);

  helfem::Vector vxc;
  if (restricted) {
    vxc = basis.xc_screening(P, xp_func, cp_func);
  } else {
    // Averaged spin-unrestricted xc screening.
    helfem::Matrix Pa = helfem::Matrix::Zero(P.rows(), P.cols());
    helfem::Matrix Pb = helfem::Matrix::Zero(P.rows(), P.cols());
    for (size_t l = 0; l < res.Pl_a.size(); ++l) Pa += res.Pl_a[l];
    for (size_t l = 0; l < res.Pl_b.size(); ++l) Pb += res.Pl_b[l];
    // xc_screening returns (Npoints, 2); rowwise mean averages the spin
    // channels, matching arma::mean(., 1).
    vxc = basis.xc_screening(Pa, Pb, xp_func, cp_func).rowwise().mean();
  }

  const helfem::Vector Zeff = vcoul + vxc;
  const helfem::Vector rho  = basis.electron_density(P);
  const helfem::Vector grho = basis.electron_density_gradient(P);
  const helfem::Vector lrho = basis.electron_density_laplacian(P);
  const helfem::Vector tau  = basis.kinetic_energy_density(Pl);

  helfem::Matrix result(r.size(), 9);
  result.col(0) = r;
  result.col(1) = rho;
  result.col(2) = grho;
  result.col(3) = lrho;
  result.col(4) = tau;
  result.col(5) = vcoul;
  result.col(6) = vxc;
  result.col(7) = wt;
  result.col(8) = basis.charge() * helfem::Vector::Ones(Zeff.size()) - Zeff;

  printf("Electron density by quadrature: %.10e\n",
         (wt.array() * rho.array() * r.array() * r.array()).sum());
  printf("Quadrature of tabulated Coulomb potential yields Coulomb energy %.10e\n",
         (0.5 * r.array() * rho.array() * wt.array() * vcoul.array()).sum());
  return result;
}

// Completeness profile Y(alpha, l): for each trial exponent alpha and
// angular momentum l, the norm of the projection of the radial AO
// (STO/GTO) onto the orthonormal FE basis -- i.e. how completely the FE
// basis reproduces that AO (1 = fully represented). Column 0 is alpha;
// columns 1..lmax+1 are the per-l profiles. Mirrors the bespoke
// SCFSolver::ao_completeness_profile.
static helfem::Matrix ao_completeness_profile(
    const helfem::sadatom::basis::TwoDBasis & basis, const helfem::Matrix & Sinvh, int lmax,
    const helfem::Vector & expn,
    const std::function<helfem::Matrix(int l, const helfem::Vector & r)> & eval_ao) {
  helfem::Matrix Y = helfem::Matrix::Zero(expn.size(), lmax + 2);
  Y.col(0) = expn;
  for (int l = 0; l <= lmax; l++) {
    helfem::Matrix ao_projection = helfem::Matrix::Zero(expn.size(), basis.Nbf());
    for (size_t iel = 0; iel < basis.get_rad_Nel(); iel++) {
      const helfem::Vector r  = basis.get_r(iel);
      const helfem::Vector wr = basis.get_wrad(iel);
      const helfem::Matrix ao = eval_ao(l, r);          // (npts x nexp)
      const helfem::Matrix bf = basis.eval_bf(iel);     // (npts x nbf_el)
      const std::vector<Eigen::Index> bfl = basis.bf_list(iel);
      const helfem::Vector wgt = wr.array() * r.array() * r.array();
      const helfem::Matrix contrib = ao.transpose() * wgt.asDiagonal() * bf;
      for (size_t j = 0; j < bfl.size(); ++j)
        ao_projection.col(bfl[j]) += contrib.col(j);
    }
    // Into the orthonormal basis, then per-exponent norm.
    const helfem::Matrix projT = (ao_projection * Sinvh).transpose();
    for (Eigen::Index ix = 0; ix < expn.size(); ix++)
      Y(ix, l + 1) = projT.col(ix).norm();
  }
  return Y;
}

// Importance profile: like the completeness profile, but projects the
// trial AOs onto the OCCUPIED SCF orbitals instead of the full FE basis
// -- how important each exponent is for the converged density. Mirrors
// SCFSolver::ao_importance_profile.
static helfem::Matrix ao_importance_profile(
    const helfem::sadatom::basis::TwoDBasis & basis,
    const helfem::Cube & C, const Eigen::VectorXi & occs, int lmax,
    const helfem::Vector & expn,
    const std::function<helfem::Matrix(int l, const helfem::Vector & r)> & eval_ao) {
  helfem::Matrix I = helfem::Matrix::Zero(expn.size(), lmax + 2);
  I.col(0) = expn;
  for (int l = 0; l <= lmax; l++) {
    if (occs(l) <= 0) continue;
    const int nocc = (int) std::ceil(occs(l) / (2.0 * (2.0 * l + 1.0)));
    const helfem::Matrix Cocc = C[l].leftCols(nocc);
    helfem::Matrix ao_projection = helfem::Matrix::Zero(nocc, expn.size());
    for (size_t iel = 0; iel < basis.get_rad_Nel(); iel++) {
      const helfem::Vector r  = basis.get_r(iel);
      const helfem::Vector wr = basis.get_wrad(iel);
      const helfem::Matrix ao = eval_ao(l, r);
      const helfem::Matrix bf = basis.eval_bf(iel);
      const std::vector<Eigen::Index> bfl = basis.bf_list(iel);
      const helfem::Matrix orbs = bf * Cocc(bfl, Eigen::all);   // (npts x nocc)
      const helfem::Vector wgt = wr.array() * r.array() * r.array();
      ao_projection += orbs.transpose() * wgt.asDiagonal() * ao;
    }
    for (Eigen::Index ix = 0; ix < expn.size(); ix++)
      I(ix, l + 1) = ao_projection.col(ix).norm();
  }
  return I;
}

// Write GTO and STO completeness/importance profiles for the converged
// density to <El>_{gto,sto}_{completeness,importance}.dat, over a
// log-spaced exponent grid.
static void write_profiles(
    const helfem::sadatom::scf::AtomicSCFResult & result, int Z, int lmax,
    double minexp = 1e-5, double maxexp = 1e10, size_t nexp = 501) {
  const helfem::Vector logexp = helfem::Vector::LinSpaced(
      nexp, std::log10(minexp), std::log10(maxexp));
  const helfem::Vector expn = logexp.array().unaryExpr(
      [](double x) { return std::pow(10.0, x); });
  const helfem::Matrix Sinvh = result.basis.Sinvh();
  auto eval_gto = [&](int l, const helfem::Vector & r) {
    return helfem::lcao::radial_GTO(r, l, expn); };
  auto eval_sto = [&](int l, const helfem::Vector & r) {
    return helfem::lcao::radial_STO(r, l, expn); };

  const std::string el = element_symbols[Z];
  io::write_raw_ascii(el + "_gto_completeness.dat",
                      ao_completeness_profile(result.basis, Sinvh, lmax, expn, eval_gto));
  io::write_raw_ascii(el + "_sto_completeness.dat",
                      ao_completeness_profile(result.basis, Sinvh, lmax, expn, eval_sto));
  io::write_raw_ascii(el + "_gto_importance.dat",
                      ao_importance_profile(result.basis, result.orbs_a, result.occs_a, lmax, expn, eval_gto));
  io::write_raw_ascii(el + "_sto_importance.dat",
                      ao_importance_profile(result.basis, result.orbs_a, result.occs_a, lmax, expn, eval_sto));
  printf("Wrote GTO/STO completeness and importance profiles for %s.\n", el.c_str());
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<int>("grid", 0, "type of grid: 1 for linear, 2 for quadratic, 3 for polynomial, 4 for exponential", false, 4);
  parser.add<double>("zexp", 0, "parameter in radial grid", false, 2.0);
  parser.add<double>("Rmax", 0, "practical infinity", false, 40.0);
  parser.add<int>("lmax", 0, "maximum angular momentum", true);
  parser.add<int>("nelem", 0, "number of elements", true);
  parser.add<std::string>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "charge of system", false, 0);
  parser.add<int>("M", 0, "spin multiplicity (2S+1); mutually exclusive with nela/nelb", false, 0);
  parser.add<int>("nela", 0, "number of alpha electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("nelb", 0, "number of beta electrons (leave 0 to derive from Q/M)", false, 0);
  parser.add<int>("restricted", 0, "spin-restricted: 1 restricted, 0 unrestricted, -1 auto from nela/nelb", false, -1);
  parser.add<int>("nnodes", 0, "number of nodes per element", false, 15);
  parser.add<int>("nquad", 0, "number of quadrature points", false, 0);
  parser.add<std::string>("method", 0, "method to use", false, "lda_x");
  parser.add<double>("dftthr", 0, "density threshold for dft", false, 1e-12);
  parser.add<int>("primbas", 0, "primitive radial basis", false, 4);
  parser.add<std::string>("x_pars", 0, "file for parameters for exchange functional", false, "");
  parser.add<std::string>("c_pars", 0, "file for parameters for correlation functional", false, "");

  // Confinement potential (parity with bespoke sadatom driver).
  parser.add<int>   ("iconf",        0, "Confinement potential: 1 polynomial, 2 exponential, 3 barrier, 4 Junquera et al.", false, 0);
  parser.add<int>   ("conf_N",       0, "Exponent in confinement potential", false, 0);
  parser.add<double>("conf_R",       0, "Confinement radius",                false, 0.0);
  parser.add<double>("conf_barrier", 0, "Confinement barrier height",        false, 0.0);
  parser.add<double>("shift_conf",   0, "Where does confinement start?",     false, 0.0);
  parser.add<bool>  ("add_conf",     0, "Add element boundary at shifted confinement radius?", false, true);

  parser.add<int>("iguess", 0, "initial guess: 0 core Hamiltonian, 1 GSZ, 2 SAP, 3 Thomas-Fermi", false, 2);
  parser.add<std::string>("occs", 0, "occupations: 'auto' (Aufbau), or space-separated per-l counts (lmax+1 totals; unrestricted also accepts 2*(lmax+1) as alpha then beta)", false, "auto");

  parser.add<std::string>("load", 0, "load orbital guess from checkpoint file", false, "");
  parser.add<std::string>("save", 0, "save results to checkpoint file",       false, "");

  // SAP / effective-potential generation (parity with bespoke gensap).
  // With a functional active, gensap tabulates the radial effective
  // charge Zeff(r) = Z - r*(V_Hartree + V_xc) to result_<El>.dat -- the
  // data used to build the SAP guess (see src/general/sap.cpp).
  parser.add<std::string>("pot", 0, "functional for the effective/SAP potential (empty = use --method)", false, "");
  parser.add<bool>("savepot", 0, "also save the xc screening potential to xcpot.dat?", false, false);
  parser.add<bool>("saveing", 0, "also save the density ingredients to xcing.dat?", false, false);
  parser.add<bool>("saveorb", 0, "save radial orbitals to orbs_<El>_<spin>_l<l>.dat?", false, false);

  parser.add<bool>("completeness", 0, "write GTO/STO completeness + importance profiles?", false, false);

  // Atomic-size diagnostics (parity with bespoke gensap).
  parser.add<double>("vdwthr", 0, "density threshold for the van der Waals radius", false, 0.001);
  parser.add<double>("eps_el", 0, "density threshold for the electron-count atomic radius", false, 0.073416683704840394115); // H analytic radius matches the vdW routine at 1e-3 (Rahm 2016)

  parser.parse_check(argc, argv);

  const int igrid   = parser.get<int>("grid");
  const double zexp = parser.get<double>("zexp");
  const int Nelem   = parser.get<int>("nelem");
  const int Z       = get_Z(parser.get<std::string>("Z"));
        int Q       = parser.get<int>("Q");
        int M       = parser.get<int>("M");
        int nela    = parser.get<int>("nela");
        int nelb    = parser.get<int>("nelb");
        int restr   = parser.get<int>("restricted");
  const int Nnodes  = parser.get<int>("nnodes");
        int Nquad   = parser.get<int>("nquad");
  const std::string method = parser.get<std::string>("method");
  const double dftthr = parser.get<double>("dftthr");
  const int primbas   = parser.get<int>("primbas");
  const std::string xparf = parser.get<std::string>("x_pars");
  const std::string cparf = parser.get<std::string>("c_pars");
  const int lmax      = parser.get<int>("lmax");
  const double Rmax   = parser.get<double>("Rmax");

  const int    iconf        = parser.get<int>("iconf");
  const int    conf_N       = parser.get<int>("conf_N");
  const double conf_R       = parser.get<double>("conf_R");
  const double conf_barrier = parser.get<double>("conf_barrier");
  const double shift_conf   = parser.get<double>("shift_conf");
  const bool   add_conf     = parser.get<bool>("add_conf");

  bool restricted;
  int Ntot;
  helfem::scf_driver::derive_nela_nelb_restricted(
      nela, nelb, restr, Q, M, Z, restricted, Ntot);
  (void) Ntot;  // sadatom SCF driver does its own per-l particle counting.

  helfem::Vector x_pars, c_pars;
  if (xparf.size()) {
    x_pars = scf::parse_xc_params(xparf);
    std::cout << "Exchange functional parameters\n" << x_pars.transpose() << std::endl;
  }
  if (cparf.size()) {
    c_pars = scf::parse_xc_params(cparf);
    std::cout << "Correlation functional parameters\n" << c_pars.transpose() << std::endl;
  }

  printf("Running %s %s calculation with Rmax=%e and %i elements.\n",
          restricted ? "restricted" : "unrestricted",
          method.c_str(), Rmax, Nelem);
  printf("nela=%d nelb=%d\n", nela, nelb);

  auto poly = std::shared_ptr<const polynomial_basis::PolynomialBasis>(
      polynomial_basis::get_basis(primbas, Nnodes));

  if (Nquad == 0)
    Nquad = 5 * poly->get_nbf();
  else if (Nquad < 2 * poly->get_nbf())
    throw std::logic_error("Insufficient radial quadrature.\n");
  printf("Using %i point quadrature rule.\n", Nquad);

  int x_func, c_func;
  ::parse_xc_func(x_func, c_func, method);
  ::print_info(x_func, c_func);
  if (!is_supported(x_func))
    throw std::logic_error("The specified exchange functional is not currently supported in HelFEM.\n");
  if (!is_supported(c_func))
    throw std::logic_error("The specified correlation functional is not currently supported in HelFEM.\n");

  const helfem::Vector bval = atomic::basis::form_grid(
      modelpotential::POINT_NUCLEUS, 0.0, Nelem, Rmax, igrid, zexp,
      0, 0, 0.0, Z, 0, 0, 0.0,
      iconf ? add_conf : false, shift_conf);

  sadatom::scf::AtomicSCFOptions opts;
  opts.Z            = Z;
  opts.lmax         = lmax;
  opts.poly         = poly;
  opts.Nquad        = Nquad;
  opts.bval         = bval;
  opts.nela         = nela;
  opts.nelb         = nelb;
  opts.restricted   = restricted;
  opts.x_func       = x_func;
  opts.c_func       = c_func;
  opts.x_pars       = x_pars;
  opts.c_pars       = c_pars;
  opts.dftthr       = dftthr;
  opts.iconf        = iconf;
  opts.conf_N       = conf_N;
  opts.conf_R       = conf_R;
  opts.conf_barrier = conf_barrier;
  opts.shift_conf   = shift_conf;
  opts.verbosity    = 5;
  opts.iguess       = parser.get<int>("iguess");
  opts.load_file    = parser.get<std::string>("load");
  opts.save_file    = parser.get<std::string>("save");

  // Explicit occupations. "auto" leaves OOO to fill by Aufbau. Otherwise
  // parse space-separated per-l integer counts and pin them via the
  // frozen-occupation API (opts.fixed_per_l_*). Restricted expects
  // lmax+1 per-l totals; unrestricted accepts lmax+1 totals (Hund
  // high-spin split into alpha/beta) or 2*(lmax+1) explicit alpha-then-
  // beta counts.
  {
    const std::string occstr = parser.get<std::string>("occs");
    if (occstr != "auto" && occstr != "AUTO" && occstr != "Auto") {
      std::istringstream iss(occstr);
      std::vector<int> vals; int v;
      while (iss >> v) vals.push_back(v);
      const int nl = lmax + 1;
      auto to_ivec = [](const std::vector<int> & x) {
        Eigen::VectorXi o(x.size());
        for (size_t i = 0; i < x.size(); ++i) o(i) = x[i];
        return o;
      };
      if (restricted) {
        if ((int) vals.size() != nl)
          throw std::logic_error("Restricted --occs needs lmax+1 per-l totals.\n");
        opts.fixed_per_l_a = to_ivec(vals);
      } else if ((int) vals.size() == nl) {
        // Hund high-spin split of each per-l total across n-shells: fill
        // shells of capacity 2*(2l+1), putting up to (2l+1) per shell in
        // alpha before beta (matches the bespoke driver's hund_rule).
        Eigen::VectorXi a = Eigen::VectorXi::Zero(nl), b = Eigen::VectorXi::Zero(nl);
        for (int l = 0; l < nl; ++l) {
          int numel = vals[l];
          while (numel > 0) {
            const int numsh = std::min(numel, 2 * (2 * l + 1));
            const int na    = std::min(numsh, 2 * l + 1);
            a(l) += na;
            b(l) += numsh - na;
            numel -= numsh;
          }
        }
        opts.fixed_per_l_a = a;
        opts.fixed_per_l_b = b;
      } else if ((int) vals.size() == 2 * nl) {
        opts.fixed_per_l_a = to_ivec(std::vector<int>(vals.begin(), vals.begin() + nl));
        opts.fixed_per_l_b = to_ivec(std::vector<int>(vals.begin() + nl, vals.end()));
      } else {
        throw std::logic_error("Unrestricted --occs needs lmax+1 totals or 2*(lmax+1) alpha/beta counts.\n");
      }
      printf("Using explicit occupations from --occs.\n");
    }
  }

  sadatom::scf::AtomicSCFResult result = sadatom::scf::run_atomic_scf(opts);

  // GTO/STO completeness + importance profiles for basis-set design.
  if (parser.get<bool>("completeness"))
    write_profiles(result, Z, lmax);

  // Density / atomic-size diagnostics (parity with bespoke gensap).
  {
    const helfem::Matrix & Pdiag = result.Prad;
    const double nucd  = result.basis.nuclear_density(Pdiag);
    const double gnucd = result.basis.nuclear_density_gradient(Pdiag);
    printf("\nElectron density          at the nucleus is % e\n", nucd);
    printf("Electron density gradient at the nucleus is % e\n", gnucd);
    if (nucd != 0.0)
      printf("Cusp condition is %.10f\n", -1.0 / (2 * Z) * gnucd / nucd);

    const double vdwthr = parser.get<double>("vdwthr");
    const double eps_el = parser.get<double>("eps_el");
    const double rvdw = result.basis.vdw_radius(Pdiag, vdwthr);
    printf("\nEstimated vdW radius with density threshold %e is %.6f bohr = %.6f A\n",
           vdwthr, rvdw, rvdw * BOHRINANGSTROM);
    printf("Note that this criterion is sensitive to numerical noise.\n");
    const double rincl = result.basis.electron_count_radius(Pdiag, eps_el);
    printf("Estimated radius with electron count threshold %e is %.6f bohr = %.6f A\n",
           eps_el, rincl, rincl * BOHRINANGSTROM);
  }

  // Save radial orbitals: one file per spin channel and l, columns
  // [r, phi_0(r), phi_1(r), ...] on the quadrature grid.
  if (parser.get<bool>("saveorb")) {
    const helfem::Vector r = result.basis.radii();
    auto dump = [&](const helfem::Cube & orbs, const char * spin) {
      for (size_t l = 0; l < orbs.size(); ++l) {
        const helfem::Matrix phi = result.basis.orbitals(orbs[l]);
        helfem::Matrix out(r.size(), phi.cols() + 1);
        out.col(0) = r;
        out.middleCols(1, phi.cols()) = phi;
        std::ostringstream oss;
        oss << "orbs_" << element_symbols[Z] << "_" << spin << "_l" << l << ".dat";
        io::write_raw_ascii(oss.str(), out);
      }
    };
    dump(result.orbs_a, restricted ? "r" : "a");
    if (!restricted)
      dump(result.orbs_b, "b");
    printf("Saved radial orbitals to orbs_%s_*.dat\n", element_symbols[Z].c_str());
  }

  // Effective-potential / SAP-table output. The potential functional is
  // --pot if given, otherwise the SCF --method. As in the bespoke
  // driver, the table is written whenever a functional is active.
  const std::string potmethod = parser.get<std::string>("pot");
  const bool savepot = parser.get<bool>("savepot");
  const bool saveing = parser.get<bool>("saveing");

  int xp_func = x_func, cp_func = c_func;
  if (potmethod.size()) {
    ::parse_xc_func(xp_func, cp_func, potmethod);
    ::print_info(xp_func, cp_func);
  }
  // Meta-GGA / range-separated potentials are not implemented in the
  // spherically symmetric screening path (matches the old driver).
  {
    bool gga, mgga_t, mgga_l;
    if (xp_func > 0) {
      ::is_gga_mgga(xp_func, gga, mgga_t, mgga_l);
      if (mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA potentials are not supported in the spherically symmetric program.\n");
    }
    if (cp_func > 0) {
      ::is_gga_mgga(cp_func, gga, mgga_t, mgga_l);
      if (mgga_t || mgga_l)
        throw std::logic_error("Meta-GGA potentials are not supported in the spherically symmetric program.\n");
    }
    if (xp_func > 0) {
      double o, a, b;
      ::range_separation(xp_func, o, a, b);
      if (o != 0.0 || a != 0.0 || b != 0.0)
        throw std::logic_error("Optimized effective potential is not implemented in the spherically symmetric program.\n");
    }
  }

  if (xp_func > 0 || cp_func > 0) {
    const helfem::Matrix pot = effective_potential_table(
        result.basis, result, restricted, xp_func, cp_func);
    std::ostringstream oss;
    oss << "result_" << element_symbols[Z] << ".dat";
    io::write_raw_ascii(oss.str(), pot);
    printf("Saved effective potential (SAP table) to %s\n", oss.str().c_str());

    if (savepot) {
      // xc screening potential: [r, v_xc]
      helfem::Matrix xcpot(pot.rows(), 2);
      xcpot.col(0) = pot.col(0);
      xcpot.col(1) = pot.col(6);
      io::write_raw_ascii("xcpot.dat", xcpot);
      printf("Saved xc screening potential to xcpot.dat\n");
    }
    if (saveing) {
      // density ingredients: [r, rho, grad, lapl, tau]
      helfem::Matrix ing(pot.rows(), 5);
      ing.col(0) = pot.col(0);
      ing.col(1) = pot.col(1);
      ing.col(2) = pot.col(2);
      ing.col(3) = pot.col(3);
      ing.col(4) = pot.col(4);
      io::write_raw_ascii("xcing.dat", ing);
      printf("Saved density ingredients to xcing.dat\n");
    }
  } else if (savepot || saveing) {
    printf("No functional active (HF) -- no xc potential/ingredients to save.\n");
  }

  return 0;
}
