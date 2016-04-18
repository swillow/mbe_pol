#include <iostream>
#include <vector>
#include <libint2.hpp>
#include <armadillo>
#include "w-qcmol/Integrals.hpp"
#include "w-qcmol/RHF.hpp"
#include "pot_embem.hpp"

//
// Embedded Many-Body Expansion Method (EMBEM)
// 

using namespace std;
using namespace libint2;
using namespace willow::qcmol;

namespace willow { namespace embem {


static arma::vec pot_embem_monomer (const arma::mat& pos,
				 const vector<unsigned short>& Zs,
				 const vector<double>& Qs);

static double pot_embem_dimer   (const arma::mat& pos,
				 const vector<unsigned short>& Zs,
				 const vector<double>& Qs,
				 const arma::vec& en_mons);


static double qcmol_rhf   (const vector<Atom>& atoms,
			   const vector<QAtom>& atoms_Q);



void   pot_init ()
{
  
  libint2::initialize ();
  libint2::Shell::do_enforce_unit_normalization (false);
  
}



void   pot_end ()
{
  libint2::finalize();
}



double pot (const arma::mat& pos,
	    const vector<unsigned short>& Zs,
	    const vector<double>& Qs)
{

  arma::vec en_mons = pot_embem_monomer (pos, Zs, Qs);
  auto en_dim = pot_embem_dimer   (pos, Zs, Qs, en_mons);

  return arma::sum(en_mons) + en_dim;

}




arma::vec pot_embem_monomer (const arma::mat& pos,
			     const vector<unsigned short>& Zs,
			     const vector<double>& Qs)
{

  const auto nwat = pos.n_cols/3;
  
  // energy of embedded monomer
  vector<Atom> atoms_W(3);
  arma::vec    en_mons(nwat);
  en_mons.zeros();
  
  //double en_mon = 0;
  
  for (auto iw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      atoms_W[isite].atomic_number = Zs[iO+isite];
      atoms_W[isite].x = pos(0,iO+isite);
      atoms_W[isite].y = pos(1,iO+isite);
      atoms_W[isite].z = pos(2,iO+isite);
    }
    
    vector<QAtom> atoms_Q;
    for (auto qw = 0; qw < nwat; ++qw) {
      if (qw == iw) continue;

      for (auto qsite = 0; qsite < 3; ++qsite) {
	QAtom qatom;
	const auto qat = 3*qw + qsite;
	qatom.x = pos(0,qat);
	qatom.y = pos(1,qat);
	qatom.z = pos(2,qat);
	qatom.charge = Qs[qat];
	atoms_Q.push_back (qatom);
      }
      
    }

    en_mons(iw) = qcmol_rhf (atoms_W, atoms_Q);

    cout << "   IW ener " << iw << "  " << en_mons(iw) << endl;

  }


  return en_mons;
  
}




double pot_embem_dimer (const arma::mat& pos,
			const vector<unsigned short>& Zs,
			const vector<double>& Qs,
			const arma::vec& en_mons)
{

  const auto nwat = pos.n_cols/3;
  
  // energy of the embedded dimer
  vector<Atom> atoms_W(6);
  double en_dimer = 0;
  
  for (auto iw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      atoms_W[isite].atomic_number = Zs[iO+isite];
      atoms_W[isite].x = pos(0,iO+isite);
      atoms_W[isite].y = pos(1,iO+isite);
      atoms_W[isite].z = pos(2,iO+isite);
    }

    for (auto jw = iw+1; jw < nwat; ++jw) {
      const auto jO = 3*jw;

      // O H1 H2
      for (auto jsite = 0; jsite < 3; ++jsite) {
	atoms_W[3+jsite].atomic_number = Zs[jO+jsite];
	atoms_W[3+jsite].x = pos(0,jO+jsite);
	atoms_W[3+jsite].y = pos(1,jO+jsite);
	atoms_W[3+jsite].z = pos(2,jO+jsite);
      }

      
      
      vector<QAtom> atoms_Q;
      for (auto qw = 0; qw < nwat; ++qw) {
	if (qw == iw || qw == jw) continue;

	for (auto qsite = 0; qsite < 3; ++qsite) {
	  QAtom qatom;
	  const auto qat = 3*qw + qsite;
	  qatom.charge = Qs[qat];
	  qatom.x = pos(0,qat);
	  qatom.y = pos(1,qat);
	  qatom.z = pos(2,qat);
	  
	  atoms_Q.push_back (qatom);
	}
      
      } // qw
      
      const auto ener = qcmol_rhf (atoms_W, atoms_Q);
      en_dimer += (ener - en_mons(iw) - en_mons(jw));

      cout << "   IJ DIMER " << iw << "  "  << jw << " " << ener << "  " <<
	(ener - en_mons(iw) - en_mons(jw)) << endl;
    } // jw

  } // iw

  return en_dimer;

}



double qcmol_rhf (const vector<Atom>& atoms,
		  const vector<QAtom>& atoms_Q)
{

  BasisSet bs ("aug-cc-pVDZ", atoms);

  Integrals ints (atoms, bs, atoms_Q);

  RHF rhf (atoms, bs, ints, false, atoms_Q);


  return rhf.energy ();
  
}



}  }  // namespace willow::embem
