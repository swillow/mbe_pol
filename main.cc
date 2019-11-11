#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <armadillo>

#include "geom.hpp"
#include "pes.hpp"
#include "message.hpp"

using namespace willow;
// 
// Unit : AU
// 
int main (int argc, char *argv[])
{

  // mpi start
  mpi::init (argc, argv);
  const auto sys_me = mpi::rank ();
  
  std::cout << std::setprecision (8);
  std::cout << std::fixed;

  // xyz file name
  const std::string fname_geom  = (argc > 1) ? argv[1] : "cage.xyz";
  // RHF vs MP2
  const std::string theory      = (argc > 2) ? argv[2] : "RHF";
  // BasisSet: aug-cc-pvdz, 6-311g**
  const std::string basis       = (argc > 3) ? argv[3] : "aug-cc-pvdz";
  // 

  
  std::vector<std::string>  at_names;
  arma::mat  pos = read_geom (fname_geom, at_names);
  arma::ivec Zs  = atom_to_Z (at_names);

  // -----

  if (sys_me == 0) {
    std::cout << "START Embedded Many Body Expansion Method " << std::endl;
  }
  const std::string method_embe = "EMBE";
  // Epol (Q): what is Q for Epol?
  // ESP, RESP, ESP_SCF, RESP_SCF
  // Here, ESP_SCF and RESP_SCF provide the atomic point charges using the self-consistent field scheme.
  
  const std::string epol_resp_scf = "RESP_SCF";
  willow::PES embe_pot_grad (pos.n_cols, theory, basis, method_embe, epol_resp_scf);
  embe_pot_grad.compute (pos, Zs);


  if (sys_me == 0) {
    std::cout << "START Many Body Expansion Method " << std::endl;
  }
  
  
  const std::string method_mbe = "MBE";
  const std::string epol_resp  = "RESP";
  willow::PES mbe_pot_grad (pos.n_cols, theory, basis, method_mbe, epol_resp);
  mbe_pot_grad.compute (pos, Zs);

  mpi::finalize ();
  
  return 0;
  
}
