#include "Integrals.hpp"
#include "Molecule.hpp"
#include "RHF.hpp"
#include "diis.hpp"

using namespace std;
using namespace libint2;

namespace willow {
namespace qcmol {

arma::mat RHF::compute_2body_ints_deriv(const libint2::BasisSet &bs,
                                        const std::vector<libint2::Atom> &atoms,
                                        const arma::mat &Dm,
                                        const arma::mat &Schwartz) {
  //  libint2::Timers<1> timer;
  //  bool l_debug = false;
  //  if (l_debug) {
  //    timer.set_now_overhead (25);
  //    timer.start (0);
  //  }

  const auto nbf = bs.nbf();
  const auto nshells = bs.size();
  const auto natom = atoms.size();
  const auto shell2atom = bs.shell2atom(atoms);

  arma::vec grd(3 * natom, arma::fill::zeros);

  libint2::Engine engine(libint2::Operator::coulomb, bs.max_nprim(), bs.max_l(),
                         1);

  const auto precision = numeric_limits<double>::epsilon();
  engine.set_precision(precision);
  const auto &buf = engine.results();

  auto shell2bf = bs.shell2bf();

  // loop over permutationally-unique set of shells
  for (auto s1 = 0; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];
    auto nbf1 = bs[s1].size();
    auto iat = shell2atom[s1];

    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto bf2_first = shell2bf[s2];
      auto nbf2 = bs[s2].size();
      auto jat = shell2atom[s2];
      auto s12_cut = Schwartz(s1, s2);

      for (auto s3 = 0; s3 <= s1; ++s3) {
        auto bf3_first = shell2bf[s3];
        auto nbf3 = bs[s3].size();
        auto kat = shell2atom[s3];

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for (auto s4 = 0; s4 <= s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto nbf4 = bs[s4].size();
          auto lat = shell2atom[s4];
          auto s34_cut = Schwartz(s3, s4);

          if (s12_cut * s34_cut < precision) {
            continue;
          }

          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;

          auto s13_s24_deg = 2.0;
          if (s1 == s3) {
            if (s2 == s4)
              s13_s24_deg = 1.0;
          }
          // (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s13_s24_deg;

          engine.compute2<Operator::coulomb, BraKet::xx_xx, 1>(bs[s1], bs[s2],
                                                               bs[s3], bs[s4]);

          arma::vec tmp(9, arma::fill::zeros);

          for (auto di = 0; di != 9; di++) {
            const auto shset = buf[di];

            double sum = 0.0;
            for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
              const auto bf1 = f1 + bf1_first;

              for (auto f2 = 0; f2 != nbf2; ++f2) {
                const auto bf2 = f2 + bf2_first;

                for (auto f3 = 0; f3 != nbf3; ++f3) {
                  const auto bf3 = f3 + bf3_first;

                  for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;

                    const auto eri4 = shset[f1234];
                    const auto value = eri4 * s1234_deg;

                    sum += 2.0 * Dm(bf1, bf2) * Dm(bf3, bf4) * value;
                    sum -= 0.5 * Dm(bf1, bf3) * Dm(bf2, bf4) * value;
                    sum -= 0.5 * Dm(bf1, bf4) * Dm(bf2, bf3) * value;
                  }
                }
              }
            }

            tmp(di) = 0.25 * sum;
          } // end di

          // store the gradient

          grd(3 * iat) += tmp(0);
          grd(3 * iat + 1) += tmp(1);
          grd(3 * iat + 2) += tmp(2);
          grd(3 * jat) += tmp(3);
          grd(3 * jat + 1) += tmp(4);
          grd(3 * jat + 2) += tmp(5);
          grd(3 * kat) += tmp(6);
          grd(3 * kat + 1) += tmp(7);
          grd(3 * kat + 2) += tmp(8);
          grd(3 * lat) -= (tmp(0) + tmp(3) + tmp(6));     // tmp(9);
          grd(3 * lat + 1) -= (tmp(1) + tmp(4) + tmp(7)); // tmp(10);
          grd(3 * lat + 2) -= (tmp(2) + tmp(5) + tmp(8)); // tmp(11);
        }
      }
    }
  }

  //  if (l_debug) {
  //    timer.stop(0);
  //    std::cout << "done (" << timer.read(0) << " s)" << std::endl;
  //  }

  arma::mat m_grd(grd.memptr(), 3, natom, true);

  return m_grd;
}

} // namespace qcmol
} // namespace willow
