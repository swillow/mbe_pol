#ifndef MP2_HPP
#define MP2_HPP

#include "Integrals.hpp"
#include "Molecule.hpp"
#include "RHF.hpp"
#include <armadillo>
#include <libint2.h>
#include <vector>

namespace willow {
namespace qcmol {

class MP2 : public RHF {

public:
  MP2(const std::vector<libint2::Atom> &atoms, const libint2::BasisSet &bs,
      const Integrals &ints, const bool l_print = true,
      const bool l_grad = false, const int qm_chg = 0,
      const std::vector<QAtom> &atoms_Q = std::vector<QAtom>());

  double mp2_energy() { return emp2; };
  double get_energy() { return (mp2_energy() + rhf_energy()); };
  arma::mat densityMatrixMP2() { return m_pmp2; };

protected:
  int nocc;
  int nvir;
  int ncore;
  int nmo;
  int nbf;

  int iocc1;
  int iocc2;
  int ivir1;
  int ivir2;

  int naocc;
  int navir;

  // Eigen Values and Vectors
  arma::vec eval;
  arma::mat Cmat;
  arma::mat mo_tiajb;
  arma::mat m_pmp2; // densityMatrixMP2

  // arma::vec   mo_ints (const double* ao_tei);
  arma::mat mo_half_ints(const double *ao_tei);
  arma::mat mo_iajb_ints(const arma::mat &mo_hf_int);
  arma::mat cphf(const arma::mat &l_ai, const arma::vec &ao_tei,
                 const bool l_print = true);

  double emp2;
  // double erhf;
};

} // namespace qcmol
} // namespace willow

#endif
