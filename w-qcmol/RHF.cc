#include "RHF.hpp"
#include "Integrals.hpp"
#include "Molecule.hpp"
#include "diis.hpp"

using namespace std;
using namespace libint2;

namespace willow {
namespace qcmol {

static double compute_nuclear_repulsion_energy(const vector<Atom> &atoms,
                                               const vector<QAtom> &atoms_Q);

static arma::mat compute_soad(const vector<Atom> &atoms);

static double electronic_HF_energy(const arma::mat &Dm, const arma::mat &Hm,
                                   const arma::mat &Fm);

RHF::RHF(const vector<Atom> &atoms, const BasisSet &bs, const Integrals &ints,
         const bool l_print, const bool l_grad, const int qm_chg,
         const vector<QAtom> &atoms_Q) {
  E_nuc = 0.0;
  if (atoms.size() == 0)
    return;
  //
  E_nuc = compute_nuclear_repulsion_energy(atoms, atoms_Q);
  //
  const double t_elec = num_electrons(atoms) - qm_chg;
  m_nocc = round(t_elec / 2);

  // -- Core Guess
  arma::mat Hm = ints.Tm + ints.Vm;
  const auto nbf = ints.SmInvh.n_rows;
  const auto nmo = ints.SmInvh.n_cols;

  arma::mat Dm(nbf, nbf, arma::fill::zeros);
  arma::mat Fm(nbf, nbf, arma::fill::zeros);

  { // use SOAD to guess density matrix
    arma::mat Dm_min = compute_soad(atoms);
    BasisSet bs_min("STO-3G", atoms);

    Fm = Hm + compute_2body_fock_general(bs, Dm_min, bs_min, true);
    eig_solver.compute(ints.SmInvh, Fm);
    Dm = densityMatrix();
  }

  // ********************
  // SCF Loop
  // ********************

  DIIS diis(ints.Sm, ints.SmInvh, Fm);

  // compute HF energy
  E_hf = electronic_HF_energy(Dm, Hm, Fm);

  const auto maxiter = 100;
  const auto conv = 1.e-7;

  auto iter = 0;
  auto rmsd = 1.0;
  auto E_diff = 0.0;

  // Prepare for incremental Fock build

  do {
    iter++;
    auto E_hf_last = E_hf;
    auto Dm_last = Dm;

    // Build a New Fock Matrix

    if (ints.l_eri_direct) {
      Fm = Hm + compute_2body_fock(bs, Dm, ints.Km);
    } else {
      Fm = Hm + g_matrix(ints.TEI, Dm);
    }

    Fm = diis.getF(Fm, Dm);

    eig_solver.compute(ints.SmInvh, Fm);
    Dm = densityMatrix();

    E_hf = electronic_HF_energy(Dm, Hm, Fm);

    E_diff = E_hf - E_hf_last;
    rmsd = norm(Dm - Dm_last);
    /*
    auto sum = 0.0;
    for (auto ib = 0; ib < nbf; ib++)
      for (auto jb = 0; jb < nbf; jb++) {
        sum += D(ib,jb)*ints.Sm_(jb,ib); // 2 electons per MO
      }
    */

    if (l_print) {
      printf(" %02d %20.12f %20.12f %20.12f %20.12f\n", iter, E_hf,
             E_hf + E_nuc, E_diff, rmsd);
    }

  } while ((fabs(E_diff) > conv) && (iter < maxiter));

  if (l_print)
    cout << "EHF " << E_hf << "  " << E_hf + E_nuc << endl;

  // Gradient Calculion
  if (l_grad) {

    arma::mat Wm = energyWeightDensityMatrix();

    const auto natom = atoms.size();

    // overlap
    arma::mat grd_s =
        compute_1body_ints_deriv<libint2::Operator::overlap>(bs, atoms, Wm);
    // kinetic
    arma::mat grd_t =
        compute_1body_ints_deriv<libint2::Operator::kinetic>(bs, atoms, Dm);

    // nuclear
    arma::mat grd_v = compute_1body_nuclear_deriv(bs, atoms, Dm, atoms_Q);

    arma::mat grd_rep = compute_nuclear_gradient(atoms, atoms_Q);

    arma::mat grd_eri = compute_2body_ints_deriv(bs, atoms, Dm, ints.Km);

    const int natom_q = atoms_Q.size();
    m_grad = arma::mat(3, natom + natom_q, arma::fill::zeros);

    for (auto ia = 0; ia < natom; ia++) {
      m_grad.col(ia) += grd_t.col(ia) + grd_v.col(ia) - grd_s.col(ia) +
                        grd_rep.col(ia) + grd_eri.col(ia);
    }

    for (auto ia = natom; ia < natom + natom_q; ia++) {
      m_grad.col(ia) += grd_v.col(ia) + grd_rep.col(ia);
    }

    if (l_print) {
      std::cout << "Total Gradient :\n";
      std::cout << m_grad.t();
    }
  }
}

arma::mat RHF::g_matrix(const arma::vec &TEI, arma::mat &Dm) {

  const auto nbf = Dm.n_rows;

  // return G = J - 0.5K
  arma::mat Gm(nbf, nbf, arma::fill::zeros);

  for (auto ic = 0; ic < nbf; ++ic) {

    for (auto jc = 0; jc <= ic; ++jc) {
      auto ij = INDEX(ic, jc);

      for (auto kc = 0; kc < nbf; ++kc) {
        for (auto lc = 0; lc < kc; ++lc) {

          auto kl = INDEX(kc, lc);

          auto ijklc = INDEX(ij, kl);
          auto eri4 = TEI(ijklc);
          // (ij|kl)
          Gm(ic, jc) += Dm(kc, lc) * eri4;
          Gm(ic, lc) -= 0.5 * Dm(kc, jc) * eri4;

          // (ij|lk)
          Gm(ic, jc) += Dm(lc, kc) * eri4;
          Gm(ic, kc) -= 0.5 * Dm(lc, jc) * eri4;

          if (ic > jc) {
            // (ji|kl)
            Gm(jc, ic) += Dm(kc, lc) * eri4;
            Gm(jc, lc) -= 0.5 * Dm(kc, ic) * eri4;

            // (ji|lk)
            Gm(jc, ic) += Dm(lc, kc) * eri4;
            Gm(jc, kc) -= 0.5 * Dm(lc, ic) * eri4;
          }

        } // exit lc

        auto kl = INDEX(kc, kc);
        auto ijklc = INDEX(ij, kl);
        auto eri4 = TEI(ijklc);

        Gm(ic, jc) += Dm(kc, kc) * eri4;
        Gm(ic, kc) -= 0.5 * Dm(kc, jc) * eri4;

        if (ic > jc) {
          Gm(jc, ic) += Dm(kc, kc) * eri4;
          Gm(jc, kc) -= 0.5 * Dm(kc, ic) * eri4;
        }

      } // exit kc
    }
  }

  return 0.5 * (Gm + Gm.t());
}

double compute_nuclear_repulsion_energy(const vector<Atom> &atoms,
                                        const vector<QAtom> &atoms_Q) {
  auto enuc = 0.0;

  for (auto i = 0; i < atoms.size(); ++i) {
    double qi = static_cast<double>(atoms[i].atomic_number);
    for (auto j = i + 1; j < atoms.size(); ++j) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      double qj = static_cast<double>(atoms[j].atomic_number);

      enuc += qi * qj / r;
    }

    for (auto j = 0; j < atoms_Q.size(); ++j) {
      auto xij = atoms[i].x - atoms_Q[j].x;
      auto yij = atoms[i].y - atoms_Q[j].y;
      auto zij = atoms[i].z - atoms_Q[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      if (r2 > 0.01) { // to avoid the overlapped Qs
        auto r = sqrt(r2);
        double qj = atoms_Q[j].charge;
        enuc += qi * qj / r;
      }
    }
  }

  return enuc;
}

arma::mat compute_soad(const vector<Atom> &atoms) {
  // compute number of atomic orbitals
  size_t nao = 0;
  for (const auto &atom : atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2)
      nao += 1;
    else if (Z <= 10)
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported ";
  }

  arma::mat Dm(nao, nao);
  Dm.zeros();

  size_t ao_offset = 0; // first AO of this atom
  for (const auto &atom : atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) {         // H, He
      Dm(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    } else if (Z <= 10) {
      Dm(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      Dm(ao_offset + 1, ao_offset + 1) =
          (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4) / 3 : 0;
      for (auto xyz = 0; xyz != 3; ++xyz)
        Dm(ao_offset + 2 + xyz, ao_offset + 2 + xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return Dm; //
}

double electronic_HF_energy(
    const arma::mat &Dm, const arma::mat &Hm,
    const arma::mat &Fm) { // See Eq. (3.184) from Szabo and Ostlund

  const auto nbf = Dm.n_rows;

  double Ehf = 0.0;

  for (auto i = 0; i < nbf; i++)
    for (auto j = 0; j < nbf; j++)
      Ehf += 0.5 * Dm(i, j) * (Hm(i, j) + Fm(i, j));

  return Ehf;
}

arma::mat RHF::energyWeightDensityMatrix() {

  const arma::vec E = eig_solver.eigenvalues();
  const arma::mat C = eig_solver.eigenvectors();
  arma::mat W(C.n_rows, C.n_rows);
  W.zeros();

  for (auto n = 0; n < m_nocc; n++)
    W += 2.0 * E(n) * C.col(n) * arma::trans(C.col(n));

  return W;
}

arma::mat RHF::densityMatrix() {
  // Form Density Matrix
  // 2.0 : number of occupied electron on each MO.

  const arma::mat Cm = eig_solver.eigenvectors();
  const auto nbf = Cm.n_rows;
  const arma::mat Cm_occ = Cm.submat(0, 0, nbf - 1, m_nocc - 1);
  arma::mat Dm = 2.0 * Cm_occ * Cm_occ.t();

  return Dm;
}

template <libint2::Operator obtype>
arma::mat RHF::compute_1body_ints_deriv(const BasisSet &bs,
                                        const vector<Atom> &atoms,
                                        const arma::mat &Dm) {
  const auto nshl = bs.size();
  const auto nbf = bs.nbf();
  const auto natom = atoms.size();

  const unsigned deriv_order = 1;

  constexpr auto nopers = libint2::operator_traits<obtype>::nopers;

  const auto nresults =
      nopers * libint2::num_geometrical_derivatives(natom, deriv_order);

  arma::vec grd(nresults, arma::fill::zeros);

  libint2::Engine engine(obtype, bs.max_nprim(), bs.max_l(), deriv_order);

  const auto &buf = engine.results();

  auto shell2bf = bs.shell2bf();
  auto shell2atom = bs.shell2atom(atoms);

  for (auto s1 = 0, s12 = 0; s1 != nshl; ++s1) {
    auto bf1 = shell2bf[s1];
    auto nbf1 = bs[s1].size();
    auto at1 = shell2atom[s1];

    assert(at1 != -1);

    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {

      auto bf2 = shell2bf[s2];
      auto nbf2 = bs[s2].size();
      auto at2 = shell2atom[s2];

      auto nbf12 = nbf1 * nbf2;

      engine.compute(bs[s1], bs[s2]);

      assert(deriv_order == 1);

      // 1. Process derivatives with respect to the Gaussian origins first
      //
      for (unsigned int d = 0; d != 6; ++d) { // 2 centers x 3 axes
        auto iat = d < 3 ? at1 : at2;
        auto op_start = (3 * iat + d % 3) * nopers;
        auto op_fence = op_start + nopers;
        const auto *buf_idx = buf[d];

        if (buf_idx == nullptr)
          continue;

        for (unsigned int op = op_start; op != op_fence; ++op) {

          for (auto ib = 0, ij = 0; ib < nbf1; ib++)
            for (auto jb = 0; jb < nbf2; jb++, ij++) {
              const auto val = buf_idx[ij];

              grd(op) += Dm(bf1 + ib, bf2 + jb) * val;

              if (s1 != s2) {
                grd(op) += Dm(bf2 + jb, bf1 + ib) * val;
              }
            }
        } // exit op
      }   // exit d
    }
  }

  arma::mat m_grd(grd.memptr(), 3, natom, true);

  return m_grd;
}

arma::mat RHF::compute_1body_nuclear_deriv(const BasisSet &bs,
                                           const vector<Atom> &atoms,
                                           const arma::mat &Dm,
                                           const vector<QAtom> &atoms_Q) {
  const auto nshl = bs.size();
  const auto nbf = bs.nbf();
  const auto natom = atoms.size();
  const auto natom_Q = atoms_Q.size();

  const unsigned deriv_order = 1;
  constexpr auto nopers = libint2::operator_traits<Operator::nuclear>::nopers;

  const auto nresults = nopers * libint2::num_geometrical_derivatives(
                                     natom + natom_Q, deriv_order);

  arma::vec grd(nresults, arma::fill::zeros);
  // grd_Q = arma::vec(atoms_Q.size()*3, arma::fill::zeros);

  libint2::Engine engine(Operator::nuclear, bs.max_nprim(), bs.max_l(),
                         deriv_order);

  const auto &buf = engine.results();

  // nuclear attraction ints engine
  vector<pair<double, array<double, 3>>> q;
  for (const auto &atom : atoms) {
    q.push_back(
        {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}});
  }

  for (auto iq = 0; iq < natom_Q; ++iq) {
    q.push_back(
        {atoms_Q[iq].charge, {{atoms_Q[iq].x, atoms_Q[iq].y, atoms_Q[iq].z}}});
  }

  engine.set_params(q);

  auto shell2bf = bs.shell2bf();
  auto shell2atom = bs.shell2atom(atoms);

  for (auto s1 = 0, s12 = 0; s1 != nshl; ++s1) {
    auto bf1 = shell2bf[s1];
    auto nbf1 = bs[s1].size();
    auto at1 = shell2atom[s1];

    assert(at1 != -1);

    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {

      auto bf2 = shell2bf[s2];
      auto nbf2 = bs[s2].size();
      auto at2 = shell2atom[s2];

      auto nbf12 = nbf1 * nbf2;

      engine.compute(bs[s1], bs[s2]);

      assert(deriv_order == 1);

      // 1. Process derivatives with respect to the Gaussian origins first
      //
      for (unsigned int d = 0; d != 6;
           ++d) { // 2 centers x 3 axes = 6 cartesian geometric derivatives

        auto iat = d < 3 ? at1 : at2;
        auto op_start = (3 * iat + d % 3) * nopers;
        auto op_fence = op_start + nopers;

        const auto *buf_idx = buf[d];

        if (buf_idx == nullptr)
          continue;

        for (unsigned int op = op_start; op != op_fence; ++op) {

          for (auto ib = 0, ij = 0; ib < nbf1; ib++)
            for (auto jb = 0; jb < nbf2; jb++, ij++) {
              const auto val = buf_idx[ij];

              grd(op) += Dm(bf1 + ib, bf2 + jb) * val;

              if (s1 != s2) {
                grd(op) += Dm(bf2 + jb, bf1 + ib) * val;
              }
            }
        }

      } // d

      // 2. Process derivatives of nuclear Coulomb operators,
      for (unsigned int iat = 0; iat != (natom + natom_Q); ++iat) {
        for (unsigned int ixyz = 0; ixyz != 3; ++ixyz) {

          const auto *buf_idx = buf[6 + iat * 3 + ixyz];

          auto op_start = (3 * iat + ixyz) * nopers;
          auto op_fence = op_start + nopers;

          for (unsigned int op = op_start; op != op_fence; ++op) {

            for (auto ib = 0, ij = 0; ib < nbf1; ib++)
              for (auto jb = 0; jb < nbf2; jb++, ij++) {
                const double val = buf_idx[ij];

                grd(op) += Dm(bf1 + ib, bf2 + jb) * val;
                if (s1 != s2)
                  grd(op) += Dm(bf2 + jb, bf1 + ib) * val;
              }
          }
        }
      } // iat =
    }
  }

  arma::mat m_grd(grd.memptr(), 3, natom + natom_Q, true);

  return m_grd;
}

arma::mat RHF::compute_nuclear_gradient(const vector<Atom> &atoms,
                                        const vector<QAtom> &atoms_Q) {

  const auto natom = atoms.size();
  const auto natom_q = atoms_Q.size();

  arma::vec grd(3 * (natom + natom_q), arma::fill::zeros);

  for (auto i = 0; i < natom; ++i) {
    double chg_i = (double)atoms[i].atomic_number;
    double xi = atoms[i].x;
    double yi = atoms[i].y;
    double zi = atoms[i].z;

    for (auto j = 0; j < i; ++j) {
      double chg_j = (double)atoms[j].atomic_number;
      // calculate distance

      double dx = xi - atoms[j].x;
      double dy = yi - atoms[j].y;
      double dz = zi - atoms[j].z;

      double rij2 = dx * dx + dy * dy + dz * dz;
      double rij3 = sqrt(rij2) * rij2;

      arma::vec tmp(3);

      tmp(0) = chg_i * chg_j / rij3 * dx;
      tmp(1) = chg_i * chg_j / rij3 * dy;
      tmp(2) = chg_i * chg_j / rij3 * dz;

      grd(3 * i) -= tmp(0);
      grd(3 * i + 1) -= tmp(1);
      grd(3 * i + 2) -= tmp(2);
      grd(3 * j) += tmp(0);
      grd(3 * j + 1) += tmp(1);
      grd(3 * j + 2) += tmp(2);
    }

    for (auto j = 0; j < natom_q; ++j) {
      const auto k = natom + j;
      double chg_j = atoms_Q[j].charge;
      // calculate distance

      double dx = xi - atoms_Q[j].x;
      double dy = yi - atoms_Q[j].y;
      double dz = zi - atoms_Q[j].z;

      double rij2 = dx * dx + dy * dy + dz * dz;
      double rij3 = sqrt(rij2) * rij2;

      arma::vec tmp(3);

      tmp(0) = chg_i * chg_j / rij3 * dx;
      tmp(1) = chg_i * chg_j / rij3 * dy;
      tmp(2) = chg_i * chg_j / rij3 * dz;

      grd(3 * i) -= tmp(0);
      grd(3 * i + 1) -= tmp(1);
      grd(3 * i + 2) -= tmp(2);

      grd(3 * k) += tmp(0);
      grd(3 * k + 1) += tmp(1);
      grd(3 * k + 2) += tmp(2);
    }
  }

  arma::mat m_grd(grd.memptr(), 3, natom + natom_q, true);

  return m_grd;
}

arma::mat RHF::compute_2body_ints_deriv(const BasisSet &bs,
                                        const vector<Atom> &atoms,
                                        const arma::mat &Dm,
                                        const arma::mat &Schwartz) {
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
          auto s13_s24_deg = (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s13_s24_deg;

          engine.compute2<Operator::coulomb, BraKet::xx_xx, 1>(bs[s1], bs[s2],
                                                               bs[s3], bs[s4]);

          arma::vec tmp(9, arma::fill::zeros);

          for (auto di = 0; di != 9; di++) {
            const auto shset = buf[di];

            if (shset == nullptr)
              continue;

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

  arma::mat m_grd(grd.memptr(), 3, natom, true);

  return m_grd;
}

} // namespace qcmol
} // namespace willow
