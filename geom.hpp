#ifndef W_GEOM_HPP
#define W_GEOM_HPP 1

#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <armadillo>

#include "constant.hpp"
#include "elements.hpp"
#include "message.hpp"


namespace willow {


inline arma::mat read_geom (const std::string& fname,
			    std::vector<std::string>& at_name)
{
    
  std::string str_geom;

  const int sys_me = mpi::rank ();

  if (sys_me == 0) {
    std::ifstream ifs_geom (fname);
    assert (ifs_geom.good());
    
    // Read the entire file into a string that can be broadcasted
    // 
    std::ostringstream oss;
    //is_geom >> oss.rdbuf();
    oss << ifs_geom.rdbuf ();
    str_geom = oss.str();
  }

  mpi::barrier ();
  mpi::broadcast_string (str_geom);
  
  std::istringstream is(str_geom);
  // xyz file
  
  size_t natom;
  
  is >> natom;
  
  arma::mat pos(3,natom);
  
  std::string rest_of_line;
  std::getline (is, rest_of_line);
  
  // second line = comment;
  std::string comment;
  std::getline (is, comment);
  
  
  for (size_t ia = 0; ia < natom; ++ia) {
    std::string line;
    std::getline (is, line);
    
    std::istringstream iss (line);
    std::string name;
    double x, y, z;
    
    iss >> name >> x >> y >> z;
    
    at_name.push_back (name);
    double *xyz = pos.colptr(ia);
    // length unit is A
    /*xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;*/

    // Bohr
    xyz[0] = x*ang2bohr;
    xyz[1] = y*ang2bohr;
    xyz[2] = z*ang2bohr;
    
  }
  
  return pos;
  
}




inline arma::ivec atom_to_Z (std::vector<std::string>& at_name)
{

  const size_t natom = at_name.size();
  arma::ivec Z_list(natom);

  for (auto ia = 0; ia < natom; ++ia) {
    const auto symbol = at_name[ia];
    for (const auto& e : willow::element_info) {
      if (symbol == e.symbol) {
	Z_list(ia) = e.Z;
	break;
      }
    }
  }

  return Z_list;
  
}


} // namespace willow

#endif

