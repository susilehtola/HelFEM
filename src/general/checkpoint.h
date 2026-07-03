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

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <armadillo>
#include "../atomic/basis.h"
#include "../diatomic/basis.h"
#include "Matrix.h"

// Use C routines, since C++ routines don't seem to add any ease of use.
extern "C" {
#include <hdf5.h>
}

/// Length of symbol
#define SYMLEN 10

/// Checkpointing class.
class Checkpoint {
  /// Name of the file
  std::string filename;
  /// Is file open for writing?
  bool writemode;

  /// Is the file open
  bool opend;
  /// The checkpoint file
  hid_t file;

  // *** Helper functions ***

  /// Save value
  void write_hbool(const std::string & name, hbool_t val);
  /// Read value
  void read_hbool(const std::string & name, hbool_t & val);

 public:
  /// Create checkpoint file
  Checkpoint(const std::string & filename, bool write, bool trunc=true);
  /// Destructor
  ~Checkpoint();

  /// Open the file
  void open();
  /// Close the file
  void close();
  /// Flush the data
  void flush();

  /// Is the file open?
  bool is_open() const;
  /// Does the entry exist in the file?
  bool exist(const std::string & name);

  /**
   * Remove entry if exists. File needs to be opened beforehand. HDF5
   * doesn't reclaim any used space after the file has been closed, so
   * this is only useful when you want to replace an existing entry
   * with something new.
   */
  void remove(const std::string & name);

  /**
   * Access routines.
   *
   * An open file will be left open, a closed file will be left closed.
   */

  /// Save matrix
  void write(const std::string & name, const arma::mat & mat);
  /// Read matrix
  void read(const std::string & name, arma::mat & mat);

  /// Eigen overload: writes/reads a helfem::Matrix by bridging through
  /// arma::mat at the HDF5 boundary. Layout on disk is unchanged so
  /// checkpoints stay round-trippable between arma-typed and
  /// Eigen-typed callers.
  void write(const std::string & name, const helfem::Matrix & mat);
  void read(const std::string & name, helfem::Matrix & mat);

  /// Eigen overload for helfem::Vector: writes as a single-column
  /// arma::mat and reads back the first column.
  void write(const std::string & name, const helfem::Vector & v);
  void read(const std::string & name, helfem::Vector & v);

  /// Save complex matrix
  void cwrite(const std::string & name, const arma::cx_mat & mat);
  /// Read complex matrix
  void cread(const std::string & name, arma::cx_mat & mat);

  /// Save matrix
  void write(const std::string & name, const arma::imat & mat);
  /// Read matrix
  void read(const std::string & name, arma::imat & mat);

  /// Save array
  void write(const std::string & name, const std::vector<double> & v);
  /// Load array
  void read(const std::string & name, std::vector<double> & v);

  /// Save array
  void write(const std::string & name, const std::vector<hsize_t> & v);
  /// Load array
  void read(const std::string & name, std::vector<hsize_t> & v);

  /// Save basis set
  void write(const helfem::atomic::basis::TwoDBasis & basis);
  /// Save basis set
  void write(const helfem::diatomic::basis::TwoDBasis & basis);
  /// Save basis set
  void read(helfem::atomic::basis::TwoDBasis & basis);
  /// Save basis set
  void read(helfem::diatomic::basis::TwoDBasis & basis);

  /// Save value
  void write(const std::string & name, double val);
  /// Read value
  void read(const std::string & name, double & val);

  /// Save value
  void write(const std::string & name, int val);
  /// Read value
  void read(const std::string & name, int & val);

  /// Save value
  void write(const std::string & name, hsize_t val);
  /// Read value
  void read(const std::string & name, hsize_t & val);

  /// Save value
  void write(const std::string & name, bool val);
  /// Read value
  void read(const std::string & name, bool & val);

  /// Save value
  void write(const std::string & name, const std::string & val);
  /// Read value
  void read(const std::string & name, std::string & val);
};

/// Check for existence of file
bool file_exists(const std::string & name);

/// Get current working directory
std::string get_cwd();
/// Change to directory, create it first if wanted
void change_dir(std::string dir, bool create=false);

/// Get a temporary file name
std::string tempname();

#endif
