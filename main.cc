#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <armadillo>

#include "geom.hpp"
#include "pot.hpp"
#include "message.hpp"

using namespace std;
using namespace willow;
// 
// Unit : AU
// 
int main (int argc, char *argv[])
{

  // mpi start
  mpi::init (argc, argv);
  const auto sys_me = mpi::rank ();
  
  pot::libint2_init ();
  cout << std::setprecision (6);
  cout << std::fixed;
  
  const string fname_geom  = (argc > 1) ? argv[1] : "cage.xyz";
  
  vector<string>  at_names;
  arma::vec  at_Qs;
  arma::mat  pos_cent = read_geom (fname_geom, at_names, at_Qs);
  arma::ivec Z_cent   = atom_to_Z (at_names);

  /*
  pot::esp_update (pos_cent, Z_cent, at_Qs);
  at_Qs = 0.9*at_Qs;
  */
  
  // -----
  if (sys_me == 0) {
    const double en_ref = -456.297066396184;
    cout << "Reference Energy (RHF/aug-cc-pVDZ)  " 
	 << en_ref << endl << endl;
  
    cout << "START Embedded Many Body Expansion Method " << endl;
  }
  
  double en_pot = pot::embem (pos_cent, Z_cent, at_Qs);

  if (sys_me == 0) {
    cout << "embem pot " << en_pot << endl << endl;

    cout << "START Many Body Expansion Method " << endl;
  }
  
  double en_pot_mbem = pot::mbem (pos_cent, Z_cent);

  if (sys_me == 0) {
    cout << " mbem pot " << en_pot_mbem << endl << endl;
  }
  
  pot::libint2_end ();

  mpi::finalize ();
  
  return 0;
  
}
