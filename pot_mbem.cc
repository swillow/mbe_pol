#include <iostream>
#include <vector>
#include <libint2.hpp>
#include <armadillo>
#include "w-qcmol/Molecule.hpp"
#include "w-qcmol/Integrals.hpp"
#include "w-qcmol/RHF.hpp"
#include "pot_embem.hpp"

//
// Many-Body Expansion Method (MBEM)
// 

using namespace std;
using namespace libint2;
using namespace willow::qcmol;

namespace willow { namespace mbem {


static arma::vec pot_mbem_monomer (const arma::mat& pos,
				   const vector<unsigned short>& Zs);

static double pot_mbem_dimer   (const arma::mat& pos,
				const vector<unsigned short>& Zs,
				const arma::vec& en_mons);


static double qcmol_rhf   (const vector<Atom>& atoms);

			   
double pot (const arma::mat& pos,
	    const vector<unsigned short>& Zs)
{

  arma::vec en_mons = pot_mbem_monomer (pos, Zs);
  auto en_dim = pot_mbem_dimer   (pos, Zs, en_mons);

  return arma::sum(en_mons) + en_dim;

}




arma::vec pot_mbem_monomer (const arma::mat& pos,
			    const vector<unsigned short>& Zs)
{

  const auto nwat = pos.n_cols/3;
  
  // energy of embedded monomer
  vector<Atom> iwat(3);
  arma::vec    en_mons(nwat);
  en_mons.zeros();
  
  //double en_mon = 0;
  
  for (auto iw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      iwat[isite].atomic_number = Zs[iO+isite];
      iwat[isite].x = pos(0,iO+isite);
      iwat[isite].y = pos(1,iO+isite);
      iwat[isite].z = pos(2,iO+isite);
    }
    
    en_mons(iw) = qcmol_rhf (iwat);

    cout << "IW ener " << iw << "  " << en_mons(iw) << endl;
  }


  return en_mons;
  
}




double pot_mbem_dimer (const arma::mat& pos,
		       const vector<unsigned short>& Zs,
		       const arma::vec& en_mons)
{

  const auto nwat = pos.n_cols/3;
  
  // energy of embedded monomer
  vector<Atom> ijwat(6);
  double en_dimer = 0;
  
  for (auto iw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      ijwat[isite].atomic_number = Zs[iO+isite];
      ijwat[isite].x = pos(0,iO+isite);
      ijwat[isite].y = pos(1,iO+isite);
      ijwat[isite].z = pos(2,iO+isite);
    }

    for (auto jw = iw+1; jw < nwat; ++jw) {
      const auto jO = 3*jw;

      // O H1 H2
      for (auto jsite = 0; jsite < 3; ++jsite) {
	ijwat[3+jsite].atomic_number = Zs[jO+jsite];
	ijwat[3+jsite].x = pos(0,jO+jsite);
	ijwat[3+jsite].y = pos(1,jO+jsite);
	ijwat[3+jsite].z = pos(2,jO+jsite);
      }

      const auto ener = qcmol_rhf (ijwat);
      en_dimer += (ener - en_mons(iw) - en_mons(jw));

      cout << "IJ DIMER " << iw << "  "  << jw << " " << ener << "  " <<
	(ener - en_mons(iw) - en_mons(jw)) << endl;
    } // jw

  } // iw

  return en_dimer;

}



double qcmol_rhf (const vector<Atom>& atoms)
{

  BasisSet bs ("aug-cc-pVDZ", atoms);

  Integrals ints (atoms, bs);

  RHF rhf (atoms, bs, ints, false);

  return rhf.energy ();
  
}



}  }  // namespace willow::embem
