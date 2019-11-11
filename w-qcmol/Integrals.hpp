#ifndef W_INTEGRALS_HPP
#define W_INTEGRALS_HPP

#include <armadillo>
#include <libint2.hpp>
#include <vector>

namespace willow {
namespace qcmol {

struct QAtom {
  int iatom;
  double x, y, z;
  double charge;
};

#define INDEX(i, j)                                                            \
  ((i >= j) ? (((i) * ((i) + 1) / 2) + (j)) : (((j) * ((j) + 1) / 2) + (i)))

class Integrals {
public:
  // Constructor
  Integrals(const std::vector<libint2::Atom> &atoms,
            const libint2::BasisSet &bs,
            const std::vector<QAtom> &atoms_Q = std::vector<QAtom>());

  template <libint2::Operator obtype>
  arma::mat compute_1body_ints(const libint2::BasisSet &bs,
                               const std::vector<libint2::Atom> &atoms,
                               const std::vector<QAtom> &Q_atoms);

  // Data members
  arma::mat Sm;      // Overlap
  arma::mat Tm;      // Kinetic-energy
  arma::mat Vm;      // Nuclear-Attraction
  arma::mat SmInvh;  // Orthogonalize Basis
  arma::mat Km;      // Schwartz screening
  arma::vec TEI;     // Store 2-e integrals
  bool l_eri_direct; //
};

extern arma::mat compute_2body_fock_memory(const arma::vec &TEI,
                                           const arma::mat &Dm);

extern arma::mat compute_shellblock_norm(const libint2::BasisSet &obs,
                                         const arma::mat &A);

extern arma::mat compute_2body_fock(const libint2::BasisSet &bs,
                                    const arma::mat &D,
                                    const arma::mat &Schwartz);

extern arma::mat compute_2body_fock_general(const libint2::BasisSet &bs,
                                            const arma::mat &Dm,
                                            const libint2::BasisSet &Dm_bs,
                                            bool Dm_is_diagonal = false);

} // namespace qcmol
} // namespace willow

#endif
