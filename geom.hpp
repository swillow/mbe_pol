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


using namespace std;

namespace willow {


inline arma::mat read_geom (const string& fname, vector<string>& at_name, vector<double>& at_Q)
{
    
  std::string str_geom;
  
  std::ifstream ifs_geom (fname);
  assert (ifs_geom.good());
    
  // Read the entire file into a string that can be broadcasted
  // 
  std::ostringstream oss;
  //is_geom >> oss.rdbuf();
  oss << ifs_geom.rdbuf ();
  str_geom = oss.str();
  
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
    double x, y, z, chg;
    
    iss >> name >> x >> y >> z >> chg;
    
    at_name.push_back (name);
    at_Q.push_back (chg);
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


inline vector<double> atom_to_phys_mass (const vector<string>& at_name)
{

  const size_t natom = at_name.size();
  vector<double> at_mass;

  for (const auto symbol : at_name) {
    for (const auto& e : willow::element_info) {
      if (symbol == e.symbol) {
	at_mass.push_back(e.mass);
	break;
      }
    }
  }

  return at_mass;
  
}


inline vector<unsigned short> atom_to_Z (vector<string>& at_name)
{

  const size_t natom = at_name.size();
  vector<unsigned short> Z_list;

  for (const auto symbol : at_name) {
    for (const auto& e : willow::element_info) {
      if (symbol == e.symbol) {
	Z_list.push_back(e.Z);
	break;
      }
    }
  }

  return Z_list;
  
}


} // namespace willow

#endif

