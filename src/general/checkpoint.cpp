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

#include "checkpoint.h"
#include "../general/polynomial_basis.h"
#include <istream>

// Helper macros
#define CHECK_OPEN() {if(!opend) {throw std::runtime_error("Cannot access checkpoint file that has not been opened!\n");}}
#define CHECK_WRITE() {if(!writemode) {throw std::runtime_error("Cannot write to checkpoint file that was opened for reading only!\n");}}
#define CHECK_EXIST() {if(!exist(name)) { std::ostringstream oss; oss << "The entry " << name << " does not exist in the checkpoint file!\n"; throw std::runtime_error(oss.str()); } }

Checkpoint::Checkpoint(const std::string & fname, bool writem, bool trunc) {
  writemode=writem;
  filename=fname;
  opend=false;

  if(writemode && (trunc || !file_exists(fname))) {
    // Truncate existing file, using default creation and access properties.
    file=H5Fcreate(fname.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    opend=true;

    // Close the file
    close();
  } else {
    // Open file
    open();
  }
}

Checkpoint::~Checkpoint() {
  if(opend)
    close();
}

void Checkpoint::open() {
  // Check that file exists
  if(!file_exists(filename)) {
    throw std::runtime_error("Trying to open nonexistent checkpoint file \"" + filename + "\"!\n");
  }

  if(!opend) {
    if(writemode)
      // Open in read-write mode
      file=H5Fopen(filename.c_str(),H5F_ACC_RDWR  ,H5P_DEFAULT);
    else
      // Open in read-only mode
      file=H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);

    // File has been opened
    opend=true;
  } else
    throw std::runtime_error("Trying to open checkpoint file that has already been opened!\n");
}

void Checkpoint::close() {
  if(opend) {
    H5Fclose(file);
    opend=false;
  } else
    throw std::runtime_error("Trying to close file that has already been closed!\n");
}

void Checkpoint::flush() {
  if(opend && writemode)
      H5Fflush(file,H5F_SCOPE_GLOBAL);
}

bool Checkpoint::is_open() const {
  return opend;
}

bool Checkpoint::exist(const std::string & name) {
  bool cl=false;
  if(!opend) {
    cl=true;
    open();
  }

  bool ret=H5Lexists(file, name.c_str(), H5P_DEFAULT);

  if(cl) close();

  return ret;
}

void Checkpoint::remove(const std::string & name) {
  CHECK_WRITE();

  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  if(exist(name)) {
    // Remove the entry from the file.
    H5Ldelete(file, name.c_str(), H5P_DEFAULT);
  }

  if(cl) close();
}

void Checkpoint::write(const std::string & name, const arma::mat & m) {
  CHECK_WRITE();

  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the matrix
  hsize_t dims[2];
  dims[0]=m.n_rows;
  dims[1]=m.n_cols;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(2,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, arma::mat & m) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=2) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 2, instead dimension is " << ndim << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  m.zeros(dims[0],dims[1]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);

  if(cl) close();
}

void Checkpoint::cwrite(const std::string & name, const arma::cx_mat & m) {
  arma::mat mreal=arma::real(m);
  arma::mat mim=arma::imag(m);

  write(name+".re",mreal);
  write(name+".im",mim);
}

void Checkpoint::cread(const std::string & name, arma::cx_mat & m) {
  arma::mat mreal, mim;
  read(name+".re",mreal);
  read(name+".im",mim);
  m=mreal*std::complex<double>(1.0,0.0)+mim*std::complex<double>(0.0,1.0);
}

void Checkpoint::write(const std::string & name, const arma::imat & m0) {
  CHECK_WRITE();

  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Convert to native int
  arma::Mat<int> m(arma::conv_to< arma::Mat<int> >::from(m0));
  
  // Remove possible existing entry
  remove(name);

  // Dimensions of the matrix
  hsize_t dims[2];
  dims[0]=m.n_rows;
  dims[1]=m.n_cols;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(2,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_INT);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, arma::imat & m0) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=2) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 2, instead dimension is " << ndim << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  arma::Mat<int> m;
  m.zeros(dims[0],dims[1]);
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);

  if(cl) close();

  // Convert to imat
  m0=arma::conv_to<arma::imat>::from(m);
}


void Checkpoint::write(const std::string & name, const std::vector<double> & v) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=v.size();

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::vector<double> & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  v.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, const std::vector<hsize_t> & v) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=v.size();

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HSIZE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::vector<hsize_t> & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_INTEGER) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not an integer value!\n";
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  v.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const helfem::atomic::basis::TwoDBasis & basis) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Atomic calculation
  write("HelFEM_ID",1);

  // Write nuclear charges
  write("Z",basis.get_Z());
  write("Zl",basis.get_Zl());
  write("Zr",basis.get_Zr());
  write("Rhalf",basis.get_Rhalf());
  write("bval",basis.get_bval());

  write("finitenuc",basis.get_nuclear_model());
  write("Rrms",basis.get_nuclear_size());

  write("n_quad",basis.get_nquad());
  write("poly_id",basis.get_poly_id());
  write("poly_order",basis.get_poly_order());

  write("lval",basis.get_lval());
  write("mval",basis.get_mval());

  if(cl) close();
}


void Checkpoint::read(helfem::atomic::basis::TwoDBasis & basis) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Atomic calculation
  int id;
  read("HelFEM_ID",id);
  if(id!=1)
    throw std::logic_error("Checkpoint does not correspond to an atomic calculation!\n");

  // Write nuclear charges
  int Z, Zl, Zr;
  read("Z",Z);
  read("Zl",Zl);
  read("Zr",Zr);

  int finitenuc;
  read("finitenuc",finitenuc);
  double Rrms;
  read("Rrms",Rrms);

  double Rhalf;
  read("Rhalf",Rhalf);

  arma::vec bval;
  read("bval",bval);

  int n_quad, poly_id, poly_order;
  read("n_quad",n_quad);
  read("poly_id",poly_id);
  read("poly_order",poly_order);

  arma::ivec lval, mval;
  read("lval", lval);
  read("mval", mval);

  helfem::polynomial_basis::PolynomialBasis *poly(helfem::polynomial_basis::get_basis(poly_id,poly_order));
  basis=helfem::atomic::basis::TwoDBasis(Z, (helfem::modelpotential::nuclear_model_t) finitenuc, Rrms, poly, n_quad, bval, lval, mval, Zl, Zr, Rhalf);
  delete poly;
  
  if(cl) close();
}

void Checkpoint::write(const helfem::diatomic::basis::TwoDBasis & basis) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Atomic calculation
  write("HelFEM_ID",2);

  // Write nuclear charges
  write("Z1",basis.get_Z1());
  write("Z2",basis.get_Z2());
  write("Rhalf",basis.get_Rhalf());
  write("bval",basis.get_bval());

  write("n_quad",basis.get_nquad());
  write("poly_id",basis.get_poly_id());
  write("poly_order",basis.get_poly_order());

  write("lval",basis.get_lval());
  write("mval",basis.get_mval());

  if(cl) close();
}


void Checkpoint::read(helfem::diatomic::basis::TwoDBasis & basis) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Atomic calculation
  int id;
  read("HelFEM_ID",id);
  if(id!=2)
    throw std::logic_error("Checkpoint does not correspond to a diatomic calculation!\n");

  // Write nuclear charges
  int Z1, Z2;
  read("Z1",Z1);
  read("Z2",Z2);

  double Rhalf;
  read("Rhalf",Rhalf);

  arma::vec bval;
  read("bval",bval);

  int n_quad, poly_id, poly_order;
  read("n_quad",n_quad);
  read("poly_id",poly_id);
  read("poly_order",poly_order);

  arma::ivec lval, mval;
  read("lval", lval);
  read("mval", mval);

  helfem::polynomial_basis::PolynomialBasis *poly(helfem::polynomial_basis::get_basis(poly_id,poly_order));
  basis=helfem::diatomic::basis::TwoDBasis(Z1, Z2, Rhalf, poly, n_quad, bval, lval, mval);
  delete poly;
  
  if(cl) close();
}

void Checkpoint::write(const std::string & name, double val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, double & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}


void Checkpoint::write(const std::string & name, int val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_INT);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, int & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);
  if(hclass!=H5T_INTEGER)
    throw std::runtime_error("Error - datatype is not integer!\n");

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, hsize_t val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HSIZE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, hsize_t & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);
  if(hclass!=H5T_INTEGER)
    throw std::runtime_error("Error - datatype is not integer!\n");

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, bool val) {
  hbool_t tmp;
  tmp=val;
  write_hbool(name,tmp);
}

void Checkpoint::read(const std::string & name, bool & v) {
  hbool_t tmp;
  read_hbool(name,tmp);
  v=tmp;
}

void Checkpoint::write_hbool(const std::string & name, hbool_t val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possibly existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HBOOL);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read_hbool(const std::string & name, hbool_t & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);

  if(cl) close();
}

void Checkpoint::write(const std::string & name, const std::string & val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possibly existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=val.size()+1;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_CHAR);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, val.c_str());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::string & val) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_INTEGER) {
    std::ostringstream oss;
    oss << "Error - " << name << " does not consist of characters!\n";
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate work memory
  char *wrk=(char *)malloc(dims[0]*sizeof(char));
  H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, wrk);
  val=std::string(wrk);
  free(wrk);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

bool file_exists(const std::string & name) {
  std::ifstream file(name.c_str());
  return file.good();
}

std::string get_cwd() {
  // Initial array size
  size_t m=1024;
  char *p=(char *) malloc(m);
  char *r=NULL;

  while(true) {
    // Get cwd in array p
    r=getcwd(p,m);
    // Success?
    if(r==p)
      break;

    // Failed, increase m
    m*=2;
    p=(char *) realloc(p,m);
  }

  std::string cwd(p);
  free(p);
  return cwd;
}

void change_dir(std::string dir, bool create) {
  if(create) {
    std::string cmd="mkdir -p "+dir;
    int err=system(cmd.c_str());
    if(err) {
      std::ostringstream oss;
      oss << "Could not create directory \"" << dir << "\".\n";
      throw std::runtime_error(oss.str());
    }
  }

  // Go to directory
  int direrr=chdir(dir.c_str());
  if(direrr) {
    std::ostringstream oss;
    oss << "Could not change to directory \"" << dir << "\".\n";
    throw std::runtime_error(oss.str());
  }
}

std::string tempname() {
  // Get random file name
  char *tmpname=tempnam("./",".chk");
  std::string name(tmpname);
  free(tmpname);

  return name;
}
