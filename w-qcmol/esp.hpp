#ifndef ESP_HPP
#define ESP_HPP

#include <armadillo>
#include <libint2.hpp>
#include <vector>

namespace willow {
namespace qcmol {

arma::vec esp_atomic_charges(const std::vector<libint2::Atom> &atoms,
                             const libint2::BasisSet &bs, const arma::mat &Dm,
                             const bool &l_resp);

}
} // namespace willow

#endif
