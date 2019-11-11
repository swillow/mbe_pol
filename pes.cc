#include <iostream>

#include "w-qcmol/RHF.hpp"
#include "w-qcmol/MP2.hpp"
#include "w-qcmol/esp.hpp"
#include "message.hpp"
#include "pes.hpp"

// Methods
// (1) Many-Body Expansion (MBE)
// (2) Embedded Many-Body Expansion (EMBE) or Binary Interaction Method (BIM)
// (3) + Polarization Energy (Epol)


namespace willow { 


static std::vector<libint2::Atom> get_Atom_list (const std::vector<int>& qm_wat,
						 const arma::mat& pos,
						 const arma::ivec& Zs);

static std::vector<qcmol::QAtom> get_QAtom_list (const std::vector<int>& qm_wat,
						 const arma::mat& pos,
						 const arma::vec& Qs);

PES::PES (const int& natom,
	  const std::string& theory,
	  const std::string& basis,
	  const std::string& method,
	  const std::string& epol_esp)
{
  sys_me    = mpi::rank();
  sys_nproc = mpi::size();

  //---
  libint2::initialize ();
  libint2::Shell::do_enforce_unit_normalization (false);
  //---
  //-----
  l_RHF = true;
  l_MP2 = false;
  
  if (theory == "MP2") {
    l_MP2 = true;
    l_RHF = false;
  }

  //-----
  m_basis = basis;

  //----
  l_mbe = true;
  l_esp_scf = false;
  
  if (method == "EMBE") {
    l_mbe = false;
    l_esp_scf = true;
  }
  
  //-----
  l_resp = true;
  if (epol_esp == "ESP" || epol_esp == "ESP_SCF")
    l_resp = false;

  if (epol_esp == "ESP_SCF" || epol_esp == "RESP_SCF")
    l_esp_scf = true;

  //----
  m_natom = natom;
  m_nwat  = m_natom/3;

  m_enr_mon = arma::vec (m_nwat, arma::fill::zeros);
  m_grd_mon = arma::mat (3, m_natom, arma::fill::zeros);
  m_grd_dim = arma::mat (3, m_natom, arma::fill::zeros);
  m_grd_pol = arma::mat (3, m_natom, arma::fill::zeros);
  m_esp_mol = arma::vec (m_natom, arma::fill::zeros);
  m_esp_scf = arma::vec (m_natom, arma::fill::zeros);
  
}

PES::~PES()
{
  libint2::finalize();
}


void PES::compute (const arma::mat& pos,
		   const arma::ivec& Zs)
{
  double en = 0.0;

  // obtain the 
  esp_mol_update (pos, Zs, m_esp_mol);
  
  if (l_esp_scf) {
    m_esp_scf = m_esp_mol;
    esp_scf_update (pos, Zs, m_esp_scf);
  }
/*
  if (l_mbe) {
    en = compute_mbe (pos, Zs);
    if (sys_me == 0) {
      std::cout << "Ener_MBE(2) " << en << std::endl;
      //std::cout << "Grd_MBE(2) \n" << m_grd_mon + m_grd_dim;
    }
  }
  else {
    en = compute_embe (pos, Zs);
    if (sys_me == 0) {
      std::cout << "Ener_EMBE(2) " << en << std::endl;
      //std::cout << "Grd_EMBE(2) \n" << m_grd_mon + m_grd_dim;
    }
  }
*/
  double epol = compute_epol  (pos, Zs);

  if (sys_me == 0) {
    std::cout << "Ener_pol " << epol << std::endl;
    //std::cout << "Grd_pol \n" << m_grd_pol;
  }
  
}


void PES::esp_mol_update (const arma::mat& pos,
			  const arma::ivec& Zs,
			  arma::vec& Qs)
{
  
  arma::vec Qs_new  (m_natom, arma::fill::zeros);
  arma::mat qm_grd  (3, m_natom, arma::fill::zeros);
  int qm_chg = 0;
  arma::vec qm_esp (3, arma::fill::zeros);
  
  for (auto iw = sys_me; iw < m_nwat; iw += sys_nproc) {

    const auto iO = 3*iw;
    std::vector<int> qm_wat = {iw};
    std::vector<libint2::Atom> atoms   = get_Atom_list (qm_wat, pos, Zs);
    std::vector<qcmol::QAtom>  atoms_Q = get_QAtom_list(qm_wat, pos, Qs);
    
    double en = run_qcmol_esp (atoms, qm_chg, atoms_Q, qm_grd,
			       qm_esp);

    Qs_new.subvec (iO, iO+2) = qm_esp;
  } // iw

  mpi::barrier ();
  // allreduce = reduce + broadcast
  mpi::allreduce_dbl_vec_sum (Qs_new);

  Qs = Qs_new;
  
}




void PES::esp_scf_update (const arma::mat& pos,
			  const arma::ivec& Zs,
			  arma::vec& Qs)
{
  arma::vec Qs_new = Qs;
  double diff_conv = 0.001;
  
  for (auto icycl = 0; icycl < 10; ++icycl) {

    arma::vec Qs_old = Qs_new;
    esp_mol_update (pos, Zs, Qs_new);

    arma::vec dQ = Qs_new - Qs_old;
    double diff = std::sqrt (arma::dot (dQ, dQ)/(double)dQ.n_elem);

    if (diff < 0.0001) break;
  }

  Qs = Qs_new;
  
}




//---
// Embedded Many-Body Expansion Method
// or Binary Interaction Method
//---
double PES::compute_embe (const arma::mat& pos,
			  const arma::ivec& Zs)
{

  double en_mon = embe_monomer (pos, Zs);
  double en_dim = embe_dimer   (pos, Zs);

  return en_mon + en_dim;

}


//----
// Many-Body Expansion 
//---

double PES::compute_mbe (const arma::mat& pos,
			 const arma::ivec& Zs)
{

  double en_mon = mbe_monomer (pos, Zs);
  double en_dim = mbe_dimer   (pos, Zs);

  return en_mon + en_dim;

}





double PES::embe_monomer (const arma::mat& pos,
			  const arma::ivec& Zs)
{

  m_enr_mon.zeros();
  int chg_im = 0;
  int nsite  = 3;
  
  // energy of embedded monomer
  for (auto iw = sys_me; iw < m_nwat; iw += sys_nproc) {
    const auto iO = 3*iw;
    std::vector<int> qm_wat = {iw};
    // O H1 H2
    std::vector<libint2::Atom> atoms  = get_Atom_list (qm_wat, pos, Zs);
    std::vector<qcmol::QAtom> atoms_Q = get_QAtom_list(qm_wat, pos, m_esp_scf);
    arma::mat wat_grd (3, m_natom, arma::fill::zeros);
    m_enr_mon(iw) = run_qcmol (atoms, chg_im, atoms_Q, wat_grd); 

    for (auto isite = 0; isite < nsite; ++isite) {
      auto iat = iO + isite;
      m_grd_mon.col(iat) += wat_grd.col(isite);
    }

    for (auto ksite = 0; ksite < m_natom-nsite; ++ksite) {
      auto kat = atoms_Q[ksite].iatom;
      m_grd_mon.col(kat) += wat_grd.col(nsite+ksite);
    }
    
  }
  
  mpi::barrier ();
  
  // all_reduce = reduce + broadcast
  mpi::allreduce_dbl_vec_sum (m_enr_mon); 
  mpi::allreduce_dbl_mat_sum (m_grd_mon); 
  
  return arma::sum(m_enr_mon);
  
}




double PES::embe_dimer (const arma::mat& pos,
			const arma::ivec& Zs)
{
  const int nsite = 3;
  
  double en_dimer = 0.0;
  int chg_iw = 0;
  int chg_jw = 0;
  m_grd_dim.zeros();
  
  for (auto iw = 0, ijw = 0; iw < m_nwat; ++iw) {
    const auto iO = 3*iw;
    
    for (auto jw = iw+1; jw < m_nwat; ++jw, ++ijw) {
      
      if (ijw % sys_nproc != sys_me) continue;
      
      const auto jO = 3*jw;

      // Dimer (Iw, Jw)
      std::vector<int> qm_wat = {iw, jw};
      std::vector<libint2::Atom> atoms_ij
	= get_Atom_list (qm_wat, pos, Zs);
      std::vector<qcmol::QAtom> atoms_Q_ij
	= get_QAtom_list (qm_wat, pos, m_esp_scf);
      arma::mat grd_ij (3, m_natom, arma::fill::zeros);
      double en_ij = run_qcmol (atoms_ij, chg_iw+chg_jw, atoms_Q_ij, grd_ij);

      // Monomer (Iw)
      std::vector<int> qm_iwat = {iw};
      std::vector<libint2::Atom> atoms_i =
	get_Atom_list (qm_iwat, pos, Zs);
      std::vector<qcmol::QAtom> atoms_Q_i =
	get_QAtom_list (qm_iwat, pos, m_esp_scf);
      arma::mat grd_i (3, m_natom, arma::fill::zeros);
      double en_i = run_qcmol (atoms_i, chg_iw, atoms_Q_i, grd_i);

      // Monomer (Jw)
      std::vector<int> qm_jwat = {jw};
      std::vector<libint2::Atom> atoms_j =
	get_Atom_list (qm_jwat, pos, Zs);
      std::vector<qcmol::QAtom> atoms_Q_j =
	get_QAtom_list (qm_jwat, pos, m_esp_scf);
      arma::mat grd_j (3, m_natom, arma::fill::zeros);
      double en_j = run_qcmol (atoms_j, chg_jw, atoms_Q_j, grd_j);

      
      en_dimer += (en_ij - en_i - en_j);

      // gradient
      // QM region
      for (auto ksite = 0; ksite < nsite; ++ksite) {
	// QM(IW)
	int iat = iO+ksite;
	m_grd_dim.col(iat) += (grd_ij.col(ksite) - grd_i.col(ksite));
	// QM(JW)
	int jat = jO+ksite;
	m_grd_dim.col(jat) += (grd_ij.col(nsite+ksite) - grd_j.col(ksite));
      }

      // MM region
      // from grd_ij.col(6:m_natom)
      for (auto ksite = 0; ksite < m_natom-6; ++ksite) {
	int kat = atoms_Q_ij[ksite].iatom;
	m_grd_dim.col(kat) += grd_ij.col(6+ksite);
      }
      
      // from grd_i.col(3:m_natom) and grd_j (3:m_natom)
      for (auto ksite = 0; ksite < m_natom-3; ++ksite) {
	int kat = atoms_Q_i[ksite].iatom;
	m_grd_dim.col(kat) -= grd_i.col(3+ksite);
	kat = atoms_Q_j[ksite].iatom;
	m_grd_dim.col(kat) -= grd_j.col(3+ksite);
      }
      
      
    } // jw

  } // iw

  mpi::barrier ();
  mpi::allreduce_dbl_sum (en_dimer);
  mpi::reduce_dbl_mat_sum (m_grd_dim);
  
  return en_dimer;

}




double PES::mbe_monomer (const arma::mat& pos,
			 const arma::ivec& Zs)
{
  
  // energy of monomer
  m_enr_mon.zeros();
  int chg_im = 0;
  int nsite  = 3;
  std::vector<qcmol::QAtom> atoms_Q;

  for (auto iw = sys_me; iw < m_nwat; iw += sys_nproc) {
    const auto iO = 3*iw;
    std::vector<int> qm_wat = {iw};
    // O H1 H2
    std::vector<libint2::Atom> atoms = get_Atom_list(qm_wat, pos, Zs);
    arma::mat grd_im (3, nsite, arma::fill::zeros);
    m_enr_mon(iw) = run_qcmol (atoms, chg_im, atoms_Q, grd_im);
    m_grd_mon.submat (0, iO, 2, iO+2) = grd_im;
  }
  
  mpi::barrier ();
  mpi::allreduce_dbl_vec_sum (m_enr_mon);
  mpi::allreduce_dbl_mat_sum (m_grd_mon);
  
  return arma::sum (m_enr_mon);
  
}




double PES::mbe_dimer (const arma::mat& pos,
		       const arma::ivec& Zs)
{
  
  // energy of embedded monomer
  double en_dimer = 0;
  int chg_iw = 0;
  int chg_jw = 0;
  std::vector<qcmol::QAtom> atoms_Q_void;
  
  for (auto iw = 0, ijw = 0; iw < m_nwat; ++iw) {
    const auto iO = 3*iw;

    for (auto jw = iw+1; jw < m_nwat; ++jw, ++ijw) {

      if (ijw % sys_nproc != sys_me) continue;
      
      const auto jO = 3*jw;
      std::vector<int> qm_wat = {iw, jw};
      std::vector<libint2::Atom> atoms_ij = get_Atom_list (qm_wat, pos, Zs);
      arma::mat grd_ij (3, 6, arma::fill::zeros);
      const auto ener = run_qcmol (atoms_ij, (chg_iw+chg_jw),
				   atoms_Q_void, grd_ij);

      en_dimer += (ener - m_enr_mon(iw) - m_enr_mon(jw));

      // gradient
      arma::mat grd_i = m_grd_mon.submat (0, iO, 2, iO+2);
      arma::mat grd_j = m_grd_mon.submat (0, jO, 2, jO+2);

      for (auto ksite = 0; ksite < 3; ++ksite) {
	int iat = iO + ksite;
	m_grd_dim.col(iat) += (grd_ij.col(ksite) - grd_i.col(ksite));
	int jat = jO + ksite;
	m_grd_dim.col(jat) += (grd_ij.col(3+ksite) - grd_j.col(ksite));
      }
      
    } // jw

  } // iw
  
  mpi::barrier ();
  mpi::allreduce_dbl_sum (en_dimer);
  mpi::reduce_dbl_mat_sum (m_grd_dim);

  return en_dimer;

}


double PES::compute_epol (const arma::mat& pos,
			  const arma::ivec& Zs)
{
  // E_pol= <psi_I:QI| H_{I/QI} | psi_I:QI> - <psi_I | H_{I/QI}| psi_I>
  //      = <psi_I:QI| H_{I/QI} | psi_I:QI> - <psi_I | H_I | psi_I >
  //        - (q_I^QM - q_I)*q_B/R_AB
  //
  const int chg_iw = 0;
  const int nsite  = 3;
  
  arma::vec esp;
  if (l_esp_scf) {
    esp = m_esp_scf;
  }
  else {
    esp = m_esp_mol;
  }

  arma::vec3 pos_i;
  arma::vec3 pos_k;
  std::vector<qcmol::QAtom> atoms_Q_void;

  m_grd_pol.zeros();
  double en_pol = 0.0;
  for (auto iw = sys_me; iw < m_nwat; iw += sys_nproc) {
    const auto iO = 3*iw;
    std::vector<int> qm_wat = {iw};
    std::vector<libint2::Atom> atoms   = get_Atom_list  (qm_wat, pos, Zs);
    std::vector<qcmol::QAtom>  atoms_Q = get_QAtom_list (qm_wat, pos, esp);
  
    const int natoms_Q = atoms_Q.size();

    arma::mat grd_qmmm (3, nsite+natoms_Q, arma::fill::zeros);
    arma::mat grd_qm   (3, nsite, arma::fill::zeros);
    double en_qmmm   = run_qcmol (atoms, chg_iw, atoms_Q, grd_qmmm);
    arma::vec esp_qm (nsite, arma::fill::zeros);
    double en_qm     = run_qcmol_esp (atoms, chg_iw, atoms_Q_void,
				      grd_qm, esp_qm);
    
    double en_mm  = 0.0;
    arma::mat grd_mm (3, nsite+natoms_Q, arma::fill::zeros);
    
    for (auto isite = 0; isite < nsite; ++isite) {
      double q_i = esp_qm(isite); //m_esp_scf(iO+isite);
      pos_i = pos.col(iO + isite);
      
      for (auto ksite = 0; ksite < natoms_Q; ++ksite) {
	pos_k(0) = atoms_Q[ksite].x;
	pos_k(1) = atoms_Q[ksite].y;
	pos_k(2) = atoms_Q[ksite].z;
	double q_k = atoms_Q[ksite].charge;

	arma::vec3 pos_ik = pos_i - pos_k;
	double rik2 = arma::dot (pos_ik, pos_ik);
	double rik  = std::sqrt (rik2);
	double u_qq = q_i*q_k/rik;
	en_mm += u_qq;
	arma::vec3 grd_ik = -u_qq/rik2*pos_ik;

	grd_mm.col(isite)       += grd_ik;
	grd_mm.col(nsite+ksite) -= grd_ik;
      }
    }

    en_pol += (en_qmmm - en_qm - en_mm);
    
    // Gradient
    // QM region
    for (auto isite = 0; isite < nsite; ++isite) {
      auto iat = iO + isite;

      arma::vec3 grd_i =
	grd_qmmm.col(isite) - grd_qm.col(isite) - grd_mm.col(isite);
      m_grd_pol.col(iat) += grd_i;
    }

    // MM region
    for (auto ksite = 0; ksite < natoms_Q; ++ksite) {
      auto kat = atoms_Q[ksite].iatom;

      arma::vec3 grd_k = grd_qmmm.col(nsite+ksite) - grd_mm.col(nsite+ksite);
      m_grd_pol.col(kat) += grd_k;
    }
    
  } // iw

  mpi::barrier ();
  mpi::allreduce_dbl_sum (en_pol);
  mpi::reduce_dbl_mat_sum (m_grd_pol);

  return en_pol;
}



double PES::run_qcmol (const std::vector<libint2::Atom>& atoms,
		       const int& qm_chg,
		       const std::vector<qcmol::QAtom>& atoms_Q,
		       arma::mat& grads)
{

  libint2::BasisSet bs (m_basis, atoms);
  qcmol::Integrals ints (atoms, bs, atoms_Q);

  bool l_print = false;
  bool l_grd   = true;
  double ener  = 0.0;
  arma::mat Dm;
  
  grads.zeros();
  
  if (l_RHF) {
    qcmol::RHF rhf (atoms, bs, ints, l_print, l_grd, qm_chg, atoms_Q);
    ener = rhf.get_energy();
    grads = rhf.get_gradient();
  }
  else {
    qcmol::MP2 mp2 (atoms, bs, ints, l_print, l_grd, qm_chg, atoms_Q);
    ener = mp2.get_energy();
    grads = mp2.get_gradient();
  }

  return ener;
  
}




double PES::run_qcmol_esp (const std::vector<libint2::Atom>& atoms,
			   const int& qm_chg,
			   const std::vector<qcmol::QAtom>& atoms_Q,
			   arma::mat& grads,
			   arma::vec& qm_esp)
{

  libint2::BasisSet bs (m_basis, atoms);
  qcmol::Integrals ints (atoms, bs, atoms_Q);

  bool l_print = false;
  bool l_grd = true;
  double ener = 0.0;
  arma::mat Dm;
  
  if (l_RHF) {
    qcmol::RHF rhf (atoms, bs, ints, l_print, l_grd, qm_chg, atoms_Q);
    ener  = rhf.get_energy();
    grads = rhf.get_gradient();
    Dm    = rhf.densityMatrix();
  }
  else {
    qcmol::MP2 mp2 (atoms, bs, ints, l_print, l_grd, qm_chg, atoms_Q);
    ener  = mp2.get_energy();
    grads = mp2.get_gradient();
    Dm    = mp2.densityMatrixMP2();
  }
  
  qm_esp = qcmol::esp_atomic_charges (atoms, bs, Dm, l_resp);
  
  return ener;
  
}



std::vector<libint2::Atom> get_Atom_list (const std::vector<int>& qm_wat,
					  const arma::mat& pos,
					  const arma::ivec& Zs)
{
  const int nsite = 3;

  std::vector<libint2::Atom> atoms;
  for (auto ii = 0; ii < qm_wat.size(); ++ii) {
    const int iw = qm_wat[ii];
    const int iO = 3*iw;
    
    for (auto isite = 0; isite < nsite; ++isite) {
      // iw
      libint2::Atom atom;
      atom.atomic_number = Zs(iO+isite);
      atom.x = pos(0,iO+isite);
      atom.y = pos(1,iO+isite);
      atom.z = pos(2,iO+isite);
      atoms.emplace_back (atom);
    }
    
  }
  
  return atoms;
  
}



std::vector<qcmol::QAtom> get_QAtom_list (const std::vector<int>& qm_wat,
					  const arma::mat& pos,
					  const arma::vec& Qs)
{
  const int nsite = 3;
  int nwat = pos.n_cols/nsite;
  std::vector<qcmol::QAtom> atoms_Q;

  for (int qw = 0; qw < nwat; ++qw) {

    bool l_qm_wat = false;
    for (int ii = 0; ii < qm_wat.size(); ++ii) {
      
      if (qw == qm_wat[ii]) {
	l_qm_wat = true;
	break;
      }
    }
    
    if (l_qm_wat) continue;
    
    for (auto qsite = 0; qsite < nsite; ++qsite) {
      qcmol::QAtom qatom;
      const auto qat = 3*qw + qsite;
      qatom.iatom = qat;
      qatom.x = pos(0,qat);
      qatom.y = pos(1,qat);
      qatom.z = pos(2,qat);
      qatom.charge = Qs(qat);
      atoms_Q.emplace_back (qatom);
    }
  }
  
  return atoms_Q;
  
}



}  // namespace willow
