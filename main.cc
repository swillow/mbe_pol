#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <armadillo>
#include <libint2.hpp>
//#include "input.hpp"
#include "geom.hpp"
#include "pot_embem.hpp"
#include "pot_mbem.hpp"

using namespace std;
using namespace willow;
using namespace willow::embem;
// 
// Unit : AU
// 
int main (int argc, char *argv[])
{
  
  libint2::initialize ();
  libint2::Shell::do_enforce_unit_normalization (false);
  
  const string fname_geom  = (argc > 1) ? argv[1] : "cage.xyz";
  
  vector<string>  at_names;
  vector<double>  at_Qs;
  arma::mat pos_cent            = read_geom (fname_geom, at_names, at_Qs);
  vector<unsigned short> Z_cent = atom_to_Z (at_names);

  // -----
  const double en_ref = -456.297066396184;
  cout << "Reference Energy (RHF/aug-cc-pVDZ)  " <<
    setprecision(8) << en_ref << endl << endl;
  
  cout << "START Embedded Many Body Expansion Method " << endl;
  
  double en_pot = embem::pot_embem (pos_cent, Z_cent, at_Qs);

  cout << "embem pot " << setprecision(8) << en_pot << endl << endl;
  


  cout << "START Many Body Expansion Method " << endl;
  
  double en_pot_mbem = embem::pot_mbem (pos_cent, Z_cent);
  
  cout << " mbem pot " << setprecision(8) << en_pot_mbem << endl << endl;

  libint2::finalize ();
  
  return 0;
  
}
