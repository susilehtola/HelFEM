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

#ifndef SAD_SOLVER_H
#define SAD_SOLVER_H

#include <armadillo>
#include "dftgrid.h"
#include "../atomic/basis.h"

namespace helfem {
  namespace sadatom {
    namespace solver {

      /// Helper for printing out configurations
      typedef struct {
        int n;
        int l;
        double E;
        int nocc;
      } shell_occupation_t;
      /// Sort in energy
      bool operator<(const shell_occupation_t & lh, const shell_occupation_t & rh);

      /// Helper for defining orbital channels
      class OrbitalChannel {
      protected:
        /// Orbital coefficients: (nbf,nmo,lmax+1)
        arma::cube C;
        /// Orbitals energies: (nmo,lmax+1)
        arma::mat E;
        /// Orbital occupations per l channel
        arma::ivec occs;
        /// Restricted occupations?
        bool restr;
        /// Maximum angular channel
        int lmax;

        /// Get capacity of shell l
        arma::sword ShellCapacity(arma::sword l) const;
        /// Count number of occupied orbitals
        size_t CountOccupied(int l) const;
        /// Get occupied orbitals
        std::vector<shell_occupation_t> GetOccupied() const;

      public:
        /// Dummy constructor
        OrbitalChannel();
        /// Constructor
        OrbitalChannel(bool restr_);
        /// Destructor
        ~OrbitalChannel();

        /// Is this a doubly occupied configuration?
        bool Restricted() const;
        /// Changes restricted setting
        void SetRestricted(bool restr_);
        /// Checks if orbitals have been initialized
        bool OrbitalsInitialized() const;
        /// Checks if occupations have been initialized
        bool OccupationsInitialized() const;

        /// Get lmax
        int Lmax() const;
        /// Set lmax
        void SetLmax(int lmax);

        /// Counts the number of electrons
        arma::sword Nel() const;
        /// Gives the occupations per l channel
        arma::ivec Occs() const;
        /// Sets the occupations
        void SetOccs(const arma::ivec & occs_);

        /// Characterizes the configuration
        std::string Characterize() const;
        /// Print out orbital info
        void Print() const;
        /// Save radial part to disk
        void Save(const sadatom::basis::TwoDBasis & basis, const std::string & symbol) const;

        /// Checks if the occupations are the same
        bool operator==(const OrbitalChannel & rh) const;

        /// Updates the orbitals by diagonalization
        void UpdateOrbitals(const arma::cube & Fl, const arma::mat & Sinvh);
        /// Updates the orbitals by a damped diagonalization (ov and vo blocks scaled)
        void UpdateOrbitalsDamped(const arma::cube & Fl, const arma::mat & Sinvh, const arma::mat & S, double dampov);
        /// Updates the orbitals by diagonalization with a level shift
        void UpdateOrbitalsShifted(const arma::cube & Fl, const arma::mat & Sinvh, const arma::mat & S, double shift);
        /// Computes a new density matrix
        void UpdateDensity(arma::cube & Pl) const;
        /// Computes a full atomic density matrix
        arma::mat FullDensity() const;

        /// Determines new occupations
        void AufbauOccupations(arma::sword numel);
        /// Gives new trial configurations by moving electrons around
        std::vector<OrbitalChannel> MoveElectrons() const;
      };

      /// Restricted configuration
      typedef struct {
        /// Orbitals
        OrbitalChannel orbs;
        /// Densities
        arma::cube Pl;
        /// Fock matrix
        arma::cube Fl;
        /// Total energy of configuration
        double Econf;
        /// Kinetic energy
        double Ekin;
        /// Potential energy
        double Epot;
        /// Coulomb repulsion energy
        double Ecoul;
        /// Exchange-correlation energy
        double Exc;
        /// Converged?
        bool converged;
      } rconf_t;
      /// Checks if orbital occupations match
      bool operator==(const rconf_t & lh, const rconf_t & rh);
      /// Orders configurations in energy
      bool operator<(const rconf_t & lh, const rconf_t & rh);

      typedef struct {
        /// Orbitals
        OrbitalChannel orbsa, orbsb;
        /// Densities
        arma::cube Pal, Pbl;
        /// Fock matrices
        arma::cube Fal, Fbl;
        /// Total energy of configuration
        double Econf;
        /// Kinetic energy
        double Ekin;
        /// Potential energy
        double Epot;
        /// Coulomb repulsion energy
        double Ecoul;
        /// Exchange-correlation energy
        double Exc;
        /// Converged?
        bool converged;
      } uconf_t;
      /// Checks if orbital occupations match
      bool operator==(const uconf_t & lh, const uconf_t & rh);
      /// Orders configurations in energy
      bool operator<(const uconf_t & lh, const uconf_t & rh);

      /// SCF Solver
      class SCFSolver {
      protected:
        /// Maximum l value
        int lmax;

        /// Basis set to use
        sadatom::basis::TwoDBasis basis;
        /// Basis set to use for computing exact exchange
        atomic::basis::TwoDBasis atbasis;

        /// Overlap matrix
        arma::mat S;
        /// Half-inverse overlap
        arma::mat Sinvh;

        /// Kinetic energy, l-independent part
        arma::mat T;
        /// Kinetic energy, l-dependent part
        arma::mat Tl;
        /// Nuclear attraction
        arma::mat Vnuc;
        /// Core Hamiltonian
        arma::mat H0;

        /// Integration grid
        dftgrid::DFTGrid grid;
        /// Exchange functional
        int x_func;
        /// Correlation functional
        int c_func;

        /// Maximum number of SCF iterations
        int maxit;
        /// Level shift
        double shift;

        /// SCF convergence threshold (DIIS error)
        double convthr;
        /// DFT density threshold
        double dftthr;
        /// Start mixing in DIIS when error smaller than
        double diiseps;
        /// Use DIIS exclusively when error smaller than
        double diisthr;
        /// Number of matrices to keep in memory
        int diisorder;

        /// Verbose operation?
        bool verbose;

        /// Form supermatrix
        arma::mat SuperMat(const arma::mat & M) const;
        /// Form supermatrix from cube
        arma::mat SuperCube(const arma::cube & M) const;
        /// Form cube from supermatrix
        arma::cube MiniMat(const arma::mat & Msuper) const;
        /// Form l(l+1)/r^2 cube
        arma::cube KineticCube() const;
        /// Replicate matrix into a cube
        arma::cube ReplicateCube(const arma::mat & M) const;
        /// Exact exchange
        arma::cube Fxx(const OrbitalChannel & orbs) const;

      public:
        /// Constructor
        SCFSolver(int Z, int lmax, polynomial_basis::PolynomialBasis * poly, int Nquad, int Nelem, double Rmax, int igrid, double zexp, int x_func_, int c_func_, int maxit_, double shift_, double convthr_, double dftthr_, double diiseps_, double diisthr_, int diisorder_);
        /// Destructor
        ~SCFSolver();

        /// Set functional
        void set_func(int x_func_, int c_func_);

        /// Build total density
        arma::mat TotalDensity(const arma::cube & Pl) const;

        /// Initialize orbitals
        void Initialize(OrbitalChannel & orbs) const;
        /// Build the Fock operator, return the energy
        double FockBuild(rconf_t & conf);
        /// Build the Fock operator, return the energy
        double FockBuild(uconf_t & conf);

        /// Solve the SCF problem, return the energy
        double Solve(rconf_t & conf);
        /// Solve the SCF problem, return the energy
        double Solve(uconf_t & conf);

        /// Compute the spin-restricted effective potential
        arma::mat RestrictedPotential(rconf_t & conf);
        /// Compute the effective potential as the mean of spin-unrestricted potentials
        arma::mat UnrestrictedPotential(uconf_t & conf);
        /// Compute the effective potential as the spin-restricted potential of the average density
        arma::mat AveragePotential(uconf_t & conf);
        /// Compute the effective potential as the density weighted average of spin-unrestricted potentials
        arma::mat WeightedPotential(uconf_t & conf);

        /// Get the basis
        const sadatom::basis::TwoDBasis & Basis() const;
      };
    }
  }
}

#endif
