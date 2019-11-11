#ifndef RHF_HPP
#define RHF_HPP

#include "EigenSolver.hpp"
#include "Integrals.hpp"
#include <armadillo>

namespace willow {
namespace qcmol {

class RHF {
public:
  RHF(const std::vector<libint2::Atom> &atoms, const libint2::BasisSet &bs,
      const Integrals &ints, const bool l_print = true,
      const bool l_grad = false, const int qm_chg = 0,
      const std::vector<QAtom> &atoms_Q = std::vector<QAtom>());

  arma::mat densityMatrix();
  arma::mat energyWeightDensityMatrix();
  arma::mat g_matrix(const arma::vec &TEI, arma::mat &Dm);

  template <libint2::Operator obtype>
  arma::mat compute_1body_ints_deriv(const libint2::BasisSet &bs,
                                     const std::vector<libint2::Atom> &atoms,
                                     const arma::mat &Dm);

  arma::mat compute_1body_nuclear_deriv(const libint2::BasisSet &bs,
                                        const std::vector<libint2::Atom> &atoms,
                                        const arma::mat &Dm,
                                        const std::vector<QAtom> &atoms_Q);

  arma::mat compute_nuclear_gradient(const std::vector<libint2::Atom> &atoms,
                                     const std::vector<QAtom> &atoms_Q);

  arma::mat compute_2body_ints_deriv(const libint2::BasisSet &bs,
                                     const std::vector<libint2::Atom> &atoms,
                                     const arma::mat &Dm,
                                     const arma::mat &Schwartz);

  double rhf_energy() { return (E_nuc + E_hf); };
  double get_nocc() { return m_nocc; };

  arma::mat get_gradient() { return m_grad; };
  virtual double get_energy() { return rhf_energy(); };

  EigenSolver eig_solver;

protected:
  arma::mat m_grad;

private:
  size_t m_nocc;
  double E_nuc;
  double E_hf;
};

} // namespace qcmol
} // namespace willow

#endif
