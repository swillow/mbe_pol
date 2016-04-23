#include <iostream>
#include <vector>
#include <libint2.hpp>
#include <armadillo>
#include "w-qcmol/Integrals.hpp"
#include "w-qcmol/RHF.hpp"
#include "w-qcmol/MP2.hpp"
#include "w-qcmol/esp.hpp"
#include "message.hpp"
#include "pot.hpp"

//
// Embedded Many-Body Expansion Method (EMBEM)
// 

using namespace std;
using namespace willow;



namespace willow { namespace pot {


static arma::vec pot_embem_monomer (const arma::mat& pos,
				    const arma::ivec& Zs,
				    const arma::vec& Qs);

static double pot_embem_dimer   (const arma::mat& pos,
				 const arma::ivec& Zs,
				 const arma::vec& Qs,
				 const arma::vec& en_mons);

static arma::vec pot_mbem_monomer (const arma::mat& pos,
				   const arma::ivec& Zs);

static double pot_mbem_dimer   (const arma::mat& pos,
				const arma::ivec& Zs,
				const arma::vec& en_mons);

static double qcmol_rhf (const std::vector<libint2::Atom>& atoms,
			 const double qm_chg = 0.0,
			 const std::vector<willow::qcmol::QAtom>& atoms_Q =
			 std::vector<willow::qcmol::QAtom> ());


static double qcmol_mp2 (const std::vector<libint2::Atom>& atoms,
			 const double qm_chg = 0.0,
			 const std::vector<willow::qcmol::QAtom>& atoms_Q =
			 std::vector<willow::qcmol::QAtom> ());


static arma::vec qcmol_esp (const std::vector<libint2::Atom>& atoms,
			    const double qm_chg = 0.0,
			    const std::vector<willow::qcmol::QAtom>& atoms_Q =
			    std::vector<willow::qcmol::QAtom> ());




void   libint2_init ()
{
  
  libint2::initialize ();
  libint2::Shell::do_enforce_unit_normalization (false);
  
}



void   libint2_end ()
{
  libint2::finalize();
}



void esp_update (const arma::mat& pos,
		 const arma::ivec& Zs,
		 arma::vec& Qs)
{

  const auto natom = pos.n_cols;
  const auto nwat  = natom/3;
  const auto sys_me = mpi::rank ();
  const auto sys_nproc = mpi::size ();
  
  vector<libint2::Atom> atoms_W(3);
  
  for (auto icycl = 0; icycl < 10; ++icycl) {

    arma::vec Qs_new(natom);
    Qs_new.zeros();
    
    for (auto iw = sys_me; iw < nwat; iw = iw + sys_nproc) {

      const auto iO = 3*iw;
      double qm_chg = 0.0;
      
      for (auto isite = 0; isite < 3; ++isite) {
	atoms_W[isite].atomic_number = Zs[iO+isite];
	atoms_W[isite].x = pos(0,iO+isite);
	atoms_W[isite].y = pos(1,iO+isite);
	atoms_W[isite].z = pos(2,iO+isite);
	qm_chg += Qs(iO+isite);
      }
      
      vector<qcmol::QAtom> atoms_Q;
      for (auto qw = 0; qw < nwat; ++qw) {
	if (qw == iw) continue;
	
	for (auto qsite = 0; qsite < 3; ++qsite) {
	  qcmol::QAtom qatom;
	  const auto qat = 3*qw + qsite;
	  qatom.x = pos(0,qat);
	  qatom.y = pos(1,qat);
	  qatom.z = pos(2,qat);
	  qatom.charge = Qs(qat);
	  atoms_Q.push_back (qatom);
	}
	
      }
      
      arma::vec chg_iw = qcmol_esp (atoms_W, qm_chg, atoms_Q);
      
      for (auto isite = 0; isite < 3; ++isite) {
	Qs_new(iO+isite) = chg_iw(isite);
      }

    } // iw

    mpi::barrier ();
    // allreduce = reduce + broadcast
    mpi::reduce_dbl_sum (Qs_new);
    mpi::broadcast_dbl (Qs_new.memptr(), natom);

    double diff = 0.0;
    for (auto ia = 0; ia < natom; ++ia) {
      diff += (Qs_new(ia) - Qs(ia))*(Qs_new(ia) - Qs(ia));
      Qs(ia) = Qs_new(ia);
    }

    diff = sqrt(diff/(double)natom);

    if (mpi::rank() == 0) {
      cout << "ESP " << icycl << "  "  << diff << endl;
    }
    if (diff < 0.0001) break;
  }
  
}




//---
// Embedded Many-Body Expansion Method
//---
double embem (const arma::mat& pos,
	      const arma::ivec& Zs,
	      const arma::vec& Qs)
{

  arma::vec en_mons = pot_embem_monomer (pos, Zs, Qs);

  if (mpi::rank() == 0)
    cout << "En MON " << sum (en_mons) << endl;
  
  auto en_dim = pot_embem_dimer   (pos, Zs, Qs, en_mons);

  return arma::sum(en_mons) + en_dim;

}


//----
// Many-Body Expansion Method
//---

double mbem (const arma::mat& pos,
	     const arma::ivec& Zs)
{

  arma::vec en_mons = pot_mbem_monomer (pos, Zs);
  auto en_dim = pot_mbem_dimer   (pos, Zs, en_mons);

  return arma::sum(en_mons) + en_dim;

}





arma::vec pot_embem_monomer (const arma::mat& pos,
			     const arma::ivec& Zs,
			     const arma::vec& Qs)
{

  const auto sys_me    = mpi::rank ();
  const auto sys_nproc = mpi::size ();

  const auto nwat = pos.n_cols/3;
  
  // energy of embedded monomer
  vector<libint2::Atom> atoms_W(3);
  arma::vec    en_mons(nwat);
  en_mons.zeros();
  
  for (auto iw = sys_me; iw < nwat; iw = iw + sys_nproc) {
    const auto iO = 3*iw;

    // O H1 H2
    double qm_chg = 0.0;
    for (auto isite = 0; isite < 3; ++isite) {
      atoms_W[isite].atomic_number = Zs(iO+isite);
      atoms_W[isite].x = pos(0,iO+isite);
      atoms_W[isite].y = pos(1,iO+isite);
      atoms_W[isite].z = pos(2,iO+isite);
      qm_chg += Qs(iO+isite);
    }
    
    vector<qcmol::QAtom> atoms_Q;
    for (auto qw = 0; qw < nwat; ++qw) {
      if (qw == iw) continue;

      for (auto qsite = 0; qsite < 3; ++qsite) {
	qcmol::QAtom qatom;
	const auto qat = 3*qw + qsite;
	qatom.x = pos(0,qat);
	qatom.y = pos(1,qat);
	qatom.z = pos(2,qat);
	qatom.charge = Qs(qat);
	atoms_Q.push_back (qatom);
      }
      
    }

    en_mons(iw) = qcmol_rhf (atoms_W, qm_chg, atoms_Q);
    //en_mons(iw) = qcmol_mp2 (atoms_W, qm_chg, atoms_Q);

    cout << "   IW ener " << iw << "  " << en_mons(iw) << endl;

  }
  
  mpi::barrier ();
  
  // all_reduce = reduce + broadcast
  mpi::reduce_dbl_sum (en_mons); 
  mpi::broadcast_dbl  (en_mons.memptr(), nwat);
  
  return en_mons;
  
}




double pot_embem_dimer (const arma::mat& pos,
			const arma::ivec& Zs,
			const arma::vec& Qs,
			const arma::vec& en_mons)
{

  const auto sys_me    = mpi::rank ();
  const auto sys_nproc = mpi::size ();
  
  const auto nwat = pos.n_cols/3;
  
  // energy of the embedded dimer
  vector<libint2::Atom> atoms_W(6);
  double en_dimer = 0;
  
  for (auto iw = 0, ijw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;
    
    for (auto jw = iw+1; jw < nwat; ++jw, ++ijw) {
      
      if (ijw % sys_nproc != sys_me) continue;
      
      const auto jO = 3*jw;
      
      // O H1 H2
      double qm_chg = 0.0;
      for (auto isite = 0; isite < 3; ++isite) {
	atoms_W[isite].atomic_number = Zs(iO+isite);
	atoms_W[isite].x = pos(0,iO+isite);
	atoms_W[isite].y = pos(1,iO+isite);
	atoms_W[isite].z = pos(2,iO+isite);
	qm_chg += Qs(iO+isite);
      }
      // O H1 H2
      for (auto jsite = 0; jsite < 3; ++jsite) {
	atoms_W[3+jsite].atomic_number = Zs(jO+jsite);
	atoms_W[3+jsite].x = pos(0,jO+jsite);
	atoms_W[3+jsite].y = pos(1,jO+jsite);
	atoms_W[3+jsite].z = pos(2,jO+jsite);
	qm_chg += Qs(jO+jsite);
      }

      qm_chg = 0.0;
      
      vector<qcmol::QAtom> atoms_Q;
      for (auto qw = 0; qw < nwat; ++qw) {
	if (qw == iw || qw == jw) continue;

	for (auto qsite = 0; qsite < 3; ++qsite) {
	  qcmol::QAtom qatom;
	  const auto qat = 3*qw + qsite;
	  qatom.charge = Qs(qat);
	  qatom.x = pos(0,qat);
	  qatom.y = pos(1,qat);
	  qatom.z = pos(2,qat);
	  
	  atoms_Q.push_back (qatom);
	}
      
      } // qw
      
      const auto ener = qcmol_rhf (atoms_W, qm_chg, atoms_Q);
      //const auto ener = qcmol_mp2 (atoms_W, qm_chg, atoms_Q);
      en_dimer += (ener - en_mons(iw) - en_mons(jw));

      cout << "   IJ DIMER " << iw << "  "  << jw << " " << ener << "  " <<
	(ener - en_mons(iw) - en_mons(jw)) << endl;
    } // jw

  } // iw

  mpi::barrier ();
  mpi::reduce_dbl_sum_1 (en_dimer);

  return en_dimer;

}




arma::vec pot_mbem_monomer (const arma::mat& pos,
			    const arma::ivec& Zs)
{
  
  const auto sys_me    = mpi::rank ();
  const auto sys_nproc = mpi::size ();

  const auto nwat = pos.n_cols/3;
  
  // energy of monomer
  vector<libint2::Atom> wat_W(3);
  arma::vec    en_mons(nwat);
  en_mons.zeros();
  
  
  for (auto iw = sys_me; iw < nwat; iw = iw + sys_nproc) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      wat_W[isite].atomic_number = Zs(iO+isite);
      wat_W[isite].x = pos(0,iO+isite);
      wat_W[isite].y = pos(1,iO+isite);
      wat_W[isite].z = pos(2,iO+isite);
    }
    
    en_mons(iw) = qcmol_rhf (wat_W);

    cout << "IW ener " << iw << "  " << en_mons(iw) << endl;
  }
  
  mpi::barrier ();
  mpi::reduce_dbl_sum (en_mons);

  return en_mons;
  
}




double pot_mbem_dimer (const arma::mat& pos,
		       const arma::ivec& Zs,
		       const arma::vec& en_mons)
{
  
  const auto sys_me    = mpi::rank ();
  const auto sys_nproc = mpi::size ();
  
  const auto nwat = pos.n_cols/3;
  
  // energy of embedded monomer
  vector<libint2::Atom> wat_W(6);
  double en_dimer = 0;
  
  for (auto iw = 0, ijw = 0; iw < nwat; ++iw) {
    const auto iO = 3*iw;

    // O H1 H2
    for (auto isite = 0; isite < 3; ++isite) {
      wat_W[isite].atomic_number = Zs(iO+isite);
      wat_W[isite].x = pos(0,iO+isite);
      wat_W[isite].y = pos(1,iO+isite);
      wat_W[isite].z = pos(2,iO+isite);
    }

    for (auto jw = iw+1; jw < nwat; ++jw, ++ijw) {

      if (ijw % sys_nproc != sys_me) continue;
      
      const auto jO = 3*jw;

      // O H1 H2
      for (auto jsite = 0; jsite < 3; ++jsite) {
	wat_W[3+jsite].atomic_number = Zs(jO+jsite);
	wat_W[3+jsite].x = pos(0,jO+jsite);
	wat_W[3+jsite].y = pos(1,jO+jsite);
	wat_W[3+jsite].z = pos(2,jO+jsite);
      }

      const auto ener = qcmol_rhf (wat_W);
      en_dimer += (ener - en_mons(iw) - en_mons(jw));

      cout << "IJ DIMER " << iw << "  "  << jw << " " << ener << "  " <<
	(ener - en_mons(iw) - en_mons(jw)) << endl;
    } // jw

  } // iw
  
  mpi::barrier ();
  
  mpi::reduce_dbl_sum_1 (en_dimer);

  
  return en_dimer;

}




double qcmol_rhf (const vector<libint2::Atom>& atoms,
		  const double qm_chg,
		  const vector<qcmol::QAtom>& atoms_Q)
{

  libint2::BasisSet bs ("aug-cc-pVDZ", atoms);

  qcmol::Integrals ints (atoms, bs, atoms_Q);

  qcmol::RHF rhf (atoms, bs, ints, round(qm_chg), false, atoms_Q);


  return rhf.energy ();
  
}


double qcmol_mp2 (const vector<libint2::Atom>& atoms,
		  const double qm_chg,
		  const vector<qcmol::QAtom>& atoms_Q)
{

  libint2::BasisSet bs ("aug-cc-pVDZ", atoms);

  qcmol::Integrals ints (atoms, bs, atoms_Q);

  qcmol::MP2 mp2 (atoms, bs, ints, round(qm_chg), false, atoms_Q);

  return (mp2.mp2_energy () + mp2.rhf_energy());
  
}



arma::vec qcmol_esp (const vector<libint2::Atom>& atoms,
		     const double qm_chg,
		     const vector<qcmol::QAtom>& atoms_Q)
{

  libint2::BasisSet bs ("aug-cc-pVDZ", atoms);

  qcmol::Integrals ints (atoms, bs, atoms_Q);

  qcmol::ESP esp (atoms, bs, ints, round(qm_chg), false, atoms_Q);


  return esp.get_atomic_charges ();
  
}




}  }  // namespace willow::embem
