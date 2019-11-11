#ifndef W_POT_EMBEM_H
#define W_POT_EMBEM_H

#include <armadillo>
#include <vector>
#include <string>
#include <libint2.hpp>
#include "w-qcmol/Integrals.hpp"

namespace willow { 

class PES {
public:

  PES (const int& natom,
       const std::string& theory,
       const std::string& basis,
       const std::string& method,
       const std::string& epol_esp);
  
  ~PES ();

  void compute (const arma::mat& pos,
		const arma::ivec& Zs);

  void esp_mol_update (const arma::mat& pos,
		       const arma::ivec& Zs,
		       arma::vec& Qs);
  
  void esp_scf_update (const arma::mat& pos,
		       const arma::ivec& Zs,
		       arma::vec& Qs);
  
  // Embedded Many-Body Expansion or Binary Interaction Method 
  double compute_embe (const arma::mat& pos,
		       const arma::ivec& Zs);
  
  // Many-Body Expansion 
  double  compute_mbe (const arma::mat& pos,
		       const arma::ivec& Zs);

  // Polarization Energy
  double  compute_epol (const arma::mat& pos,
			const arma::ivec& Zs);

protected:
  double embe_monomer (const arma::mat& pos,
		       const arma::ivec& Zs);

  double embe_dimer (const arma::mat& pos,
		     const arma::ivec& Zs);

  double mbe_monomer (const arma::mat& pos,
		      const arma::ivec& Zs);

  double mbe_dimer (const arma::mat& pos,
		    const arma::ivec& Zs);

  double run_qcmol_esp (const std::vector<libint2::Atom>& atoms,
			const int& qm_chg,
			const std::vector<qcmol::QAtom>& atoms_Q,
			arma::mat& grads,
			arma::vec& qm_esp);
  
  double run_qcmol (const std::vector<libint2::Atom>& atoms,
		    const int& qm_chg,
		    const std::vector<qcmol::QAtom>& atoms_Q,
		    arma::mat& grads);
  
private:

  arma::vec m_enr_mon;
  arma::mat m_grd_mon;
  arma::mat m_grd_dim;
  arma::mat m_grd_pol;

  arma::vec m_esp_mol; // (r)esp charges of the molecule
  arma::vec m_esp_scf; // (r)esp charges in scf
  
  bool l_RHF;
  bool l_MP2;
  bool l_mbe;
  bool l_esp_scf;
  bool l_resp;

  int sys_me;
  int sys_nproc;
  int m_natom;
  int m_nwat;

  std::string m_basis;
};

} // namespace willow::pot


#endif
