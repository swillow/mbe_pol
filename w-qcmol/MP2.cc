// Ishimura, K. Pulay, P. and Nagase, S.
// J. Comput. Chem. Vol. 28 page 2035--2042 (2007)
//----------
//#include <libint2.h>
#include "MP2.hpp"
#include "RHF.hpp"

namespace willow {
namespace qcmol {

MP2::MP2(const std::vector<libint2::Atom> &atoms, const libint2::BasisSet &bs,
         const Integrals &ints, const bool l_print, const bool l_grad,
         const int qm_chg, const std::vector<QAtom> &atoms_Q)
    : RHF(atoms, bs, ints, l_print, false, qm_chg, atoms_Q) {

  emp2 = 0.0;
  if (atoms.size() == 0)
    return;

  // erhf = rhf_energy ();

  nocc = get_nocc();
  ncore = compute_ncore(atoms);
  nbf = ints.SmInvh.n_rows;
  nmo = ints.SmInvh.n_cols;

  iocc1 = ncore;
  iocc2 = nocc - 1;
  ivir1 = nocc;
  ivir2 = nmo - 1;

  // active occupied MO
  // active virtual  MO
  naocc = nocc - ncore;
  navir = nmo - nocc;
  nvir = navir;

  eval = eig_solver.eigenvalues();
  Cmat = eig_solver.eigenvectors();

  arma::mat mo_hf_int = mo_half_ints(ints.TEI.memptr());
  // (bj|ai) = (navir, nocc, navir, naocc)
  arma::mat mo_tei = mo_iajb_ints(mo_hf_int);

  arma::mat mo_tiajb(nocc * navir, naocc * navir, arma::fill::zeros);

  { // calc. ecorr and store mo_tiajb

    for (auto im = 0; im < naocc; im++) {
      auto moi = iocc1 + im;

      for (auto am = 0; am < navir; am++) {
        auto moa = ivir1 + am;

        for (auto moj = 0; moj < nocc; moj++) {
          // auto moj = iocc1 + jm;

          for (auto bm = 0; bm < navir; bm++) {
            auto mob = ivir1 + bm;

            double den = eval(moi) + eval(moj) - eval(moa) - eval(mob);

            auto mia = im * navir + am;
            auto mib = im * navir + bm;

            auto mjb = moj * navir + bm;
            auto mja = moj * navir + am;

            auto o_iajb = mo_tei(mjb, mia);
            auto o_ibja = mo_tei(mja, mib);
            auto t_iajb = (2.0 * o_iajb - o_ibja) / den;

            mo_tiajb(mjb, mia) = t_iajb;

            if (moj < ncore)
              continue;

            emp2 += t_iajb * o_iajb;

          } // bm

        } // jm
      }   // am
    }     // im

    if (l_print)
      std::cout << "EMP2     " << emp2 << std::endl;
  }

  if (!l_grad)
    return;

  const arma::mat Caocc = Cmat.submat(0, iocc1, nbf - 1, iocc2);
  const arma::mat Cocc = Cmat.submat(0, 0, nbf - 1, iocc2);
  const arma::mat Cvir = Cmat.submat(0, ivir1, nbf - 1, ivir2);

  arma::mat p_ij(nocc, nocc, arma::fill::zeros);   // Eq. (13), (14), (15)
  arma::mat p_ab(navir, navir, arma::fill::zeros); // Eq. (18)
  arma::mat w_ij(nocc, nocc, arma::fill::zeros);
  arma::mat w_ab(navir, navir, arma::fill::zeros); // Eq. (7)

  {
    arma::vec v_t_ov(navir * nocc, arma::fill::zeros);
    arma::mat m_t_ov(v_t_ov.memptr(), navir, nocc, false);

    arma::vec v_o_ov(navir * nocc, arma::fill::zeros);
    arma::mat m_o_ov(v_o_ov.memptr(), navir, nocc, false);

    //
    // p_jk = -2 sum(iab) t(bj|ai) (bk|ai)/(d_iabk)  Eq. (15)
    // w_jk = -2 sum(iab) t(bj|ai) (bk|ai)           Eq. (6)
    //
    for (auto im = 0, iam = 0; im < naocc; im++) {
      auto moi = iocc1 + im;
      for (auto am = 0; am < navir; am++, iam++) {
        auto moa = ivir1 + am;
        v_t_ov = mo_tiajb.col(iam); // (navir, nocc)
        v_o_ov = mo_tei.col(iam);

        // t^T (navir, naocc) o(navir, naocc)
        // w_ij.submat(iocc1,iocc1,iocc2,iocc2) -= 2.0*m_t_ov.t()*m_o_ov;

        w_ij.submat(iocc1, iocc1, iocc2, iocc2) -=
            2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2).t() *
            m_o_ov.submat(0, iocc1, navir - 1, iocc2);

        // t (navir, naocc) o(navir, naocc)
        w_ab -= 2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2) *
                m_o_ov.submat(0, iocc1, navir - 1, iocc2).t();

        // t^T (navir, naocc) o(navir, naocc)
        // p_ij.submat(iocc1,iocc1,iocc2,iocc2) -= 2.0*m_t_ov.t()*m_o_ov;
        for (auto jm = 0; jm < naocc; jm++) {
          auto moj = iocc1 + jm;
          for (auto bm = 0; bm < navir; bm++) {
            auto mob = ivir1 + bm;
            double den = eval(moi) + eval(moj) - eval(moa) - eval(mob);
            m_o_ov(bm, moj) /= den;
          }
        }

        p_ij.submat(iocc1, iocc1, iocc2, iocc2) -=
            2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2).t() *
            m_o_ov.submat(0, iocc1, navir - 1, iocc2);

        p_ab += 2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2) *
                m_o_ov.submat(0, iocc1, navir - 1, iocc2).t();

        // p_jK =  2 sum(iab) t(bj|ai) (bK|ai)/(e_j - e_K)  Eq. (14)

        p_ij.submat(iocc1, 0, iocc2, ncore - 1) -=
            2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2).t() *
            m_o_ov.submat(0, 0, navir - 1, ncore - 1);
      } // exit am
    }   // exit im

    for (auto moj = 0; moj < ncore; moj++) {
      for (auto moi = iocc1; moi < nocc; moi++) {
        p_ij(moi, moj) /= (eval(moj) - eval(moi));
      }
    }

    p_ij.submat(0, iocc1, ncore - 1, iocc2) =
        p_ij.submat(iocc1, 0, iocc2, ncore - 1).t();
  }

  arma::mat w_ai(navir, nocc, arma::fill::zeros);

  { //

    arma::vec v_t_ov(navir * nocc, arma::fill::zeros);
    arma::mat m_t_ov(v_t_ov.memptr(), navir, nocc, false);
    arma::mat m_ao_uv(nbf, nbf, arma::fill::zeros);

    // Eq. (8) W_bk = -4 sum(iaj) t(bj|ai) o(kj|ai)
    for (auto im = 0, iam = 0; im < naocc; im++) {
      for (auto am = 0; am < navir; am++, iam++) {

        arma::vec v_ao_uv = mo_hf_int.col(iam);

        for (auto ib = 0, ijb = 0; ib < nbf; ib++)
          for (auto jb = 0; jb <= ib; jb++, ijb++) {
            double tmp = v_ao_uv(ijb);
            m_ao_uv(ib, jb) = tmp;
            m_ao_uv(jb, ib) = tmp;
          }

        v_t_ov = mo_tiajb.col(iam);

        w_ai -= 2.0 * m_t_ov.submat(0, iocc1, navir - 1, iocc2) *
                (Caocc.t() * m_ao_uv.t() * Cocc);
        // l_ai.submat(0, iocc1, navir-1, iocc2) += tmpval;

      } // am
    }   // exit im
  }

  arma::mat l_ai = -w_ai;

  arma::mat l_ai4(navir, naocc, arma::fill::zeros);
  {

    arma::vec v_t_ov(navir * nocc, arma::fill::zeros);
    arma::mat m_t_ov(v_t_ov.memptr(), navir, nocc, false);
    arma::mat m_ao_uv(nbf, nbf, arma::fill::zeros);

    // l_ai4 = sum(iab) t(bj|ai) o(bc|ai)
    for (auto im = 0, iam = 0; im < naocc; im++) {
      for (auto am = 0; am < navir; am++, iam++) {

        arma::mat mo_ia_uv = mo_hf_int.col(iam);

        for (auto ib = 0, ijb = 0; ib < nbf; ib++)
          for (auto jb = 0; jb <= ib; jb++, ijb++) {
            double tmp = mo_ia_uv(ijb);
            m_ao_uv(ib, jb) = tmp;
            m_ao_uv(jb, ib) = tmp;
          }

        v_t_ov = mo_tiajb.col(iam);

        l_ai4 -= 2.0 * Cvir.t() * m_ao_uv * Cvir *
                 m_t_ov.submat(0, iocc1, navir - 1, iocc2);
      }
    }

    l_ai.submat(0, iocc1, navir - 1, iocc2) += l_ai4;
  }

  //             (nbf,navir)
  arma::mat Dm2 = Cvir * p_ab * Cvir.t() + Cocc * p_ij * Cocc.t();
  arma::mat Gmat = g_matrix(ints.TEI, Dm2);

  l_ai -= 2.0 * Cvir.t() * Gmat * Cocc;

  // computation of l_ai is now complete

  //------
  // Solve the CPHF equations
  //------
  arma::mat p_ai =
      cphf(l_ai, ints.TEI, l_print); //(navir, nocc, arma::fill::zeros);

  // Now we have the matrices p_ij, p_ai, p_ab, w_ij, w_ai, w_ab.
  // We can calculate the remaining contributions to the gradient.

  arma::mat m_p2ao(nbf, nbf, arma::fill::zeros);
  arma::mat m_phf = densityMatrix();

  {

    arma::mat m_p2mo(nmo, nmo, arma::fill::zeros);

    m_p2mo.submat(0, 0, iocc2, iocc2) = p_ij;
    m_p2mo.submat(ivir1, 0, ivir2, iocc2) = p_ai;
    m_p2mo.submat(0, ivir1, iocc2, ivir2) = p_ai.t();
    m_p2mo.submat(ivir1, ivir1, ivir2, ivir2) = p_ab;

    // (nbf,nbf) = (nbf,nmo)*(nmo,nmo)*(nbf,nmo)^T
    m_p2ao = Cmat * m_p2mo * Cmat.t();

    //
    // Compute the MP2 density matrix
    //

    m_pmp2 = m_phf + m_p2ao;
  }

  // Update Wab using Pab
  {
    // w_ab = - 0.5 p_ab(e_a+e_b)
    //---
    // if (a>b) {
    // w_ab = - e_a p_ab
    // w_ba = - e_b p_ab
    // }
    for (auto am = 0; am < navir; am++) {
      auto moa = ivir1 + am;
      for (auto bm = 0; bm < am; bm++) {
        auto mob = ivir1 + bm;
        w_ab(bm, am) -= eval(moa) * p_ab(bm, am);
        w_ab(am, bm) -= eval(mob) * p_ab(bm, am);
      }
      w_ab(am, am) -= eval(moa) * p_ab(am, am);
    }
  }

  // Update Wij using Pij
  {

    for (auto moi = 0; moi < nocc; moi++) {
      for (auto moj = 0; moj < moi; moj++) {
        if (moi < ncore && moj < ncore)
          continue;

        if (moj < ncore) {
          w_ij(moj, moi) -= eval(moi) * p_ij(moj, moi);
          w_ij(moi, moj) -= eval(moi) * p_ij(moi, moj);
        } else {
          w_ij(moj, moi) -= eval(moi) * p_ij(moj, moi);
          w_ij(moi, moj) -= eval(moj) * p_ij(moi, moj);
        }
      }

      w_ij(moi, moi) -= eval(moi) * p_ij(moi, moi);
    }
  }

  // Finish computation of Waj
  {
    for (auto moi = 0; moi < nocc; moi++) {
      for (auto am = 0; am < navir; am++) {
        w_ai(am, moi) -= eval(moi) * p_ai(am, moi);
      }
    }
  }

  // Finish computation of Wkj

  {
    //                                       (navir,nocc)
    arma::mat Dm = Cocc * p_ij.t() * Cocc.t() + Cvir * p_ai * Cocc.t() +
                   Cocc * p_ai.t() * Cvir.t() + Cvir * p_ab * Cvir.t();
    arma::mat Gm = g_matrix(ints.TEI, Dm);
    w_ij -= 2.0 * Cocc.t() * Gm * Cocc;
  }

  // 1.2 Back Transformation : t(bj|ai) --> t(vu|ai)
  arma::mat m_12_bt(naocc * navir, nbf * nbf, arma::fill::zeros);
  {
    // -- first --
    arma::vec v_t_ov(navir * nocc, arma::fill::zeros);
    arma::mat m_t_ov(v_t_ov.memptr(), navir, nocc, false);

    arma::vec v_uv(nbf * nbf, arma::fill::zeros);
    arma::mat m_uv(v_uv.memptr(), nbf, nbf, false);
    arma::mat mt_12_bt(nbf * nbf, naocc * navir, arma::fill::zeros);

    for (auto im = 0, iam = 0; im < naocc; im++) {
      // 1.2 Back Transformation : t(bj|ai) --> t(vu|ai)
      for (auto am = 0; am < navir; am++, iam++) {
        v_t_ov = mo_tiajb.col(iam);
        // (1) Back Transformation : t(bj|ai) --> t(vu|ai)
        // Cvir(nbf,navir) t_ov(navir, naocc) * Cocc(nbf, naocc)
        m_uv = Cvir * m_t_ov.submat(0, iocc1, navir - 1, iocc2) * Caocc.t();
        mt_12_bt.col(iam) = v_uv;
      } // exit am
    }

    m_12_bt = mt_12_bt.t();
  }

  const auto natom = atoms.size();

  // Now we have the matrices p_ij, p_ai, p_ab, w_ij, w_ai, w_ab.
  // We can calculate the remaining contributions to the gradient.
  {
    arma::mat m_w2ao(nbf, nbf, arma::fill::zeros);
    arma::mat m_w2mo(nmo, nmo, arma::fill::zeros);

    m_w2mo.submat(0, 0, iocc2, iocc2) = w_ij;
    m_w2mo.submat(ivir1, 0, ivir2, iocc2) = w_ai;
    m_w2mo.submat(0, ivir1, iocc2, ivir2) = w_ai.t();
    m_w2mo.submat(ivir1, ivir1, ivir2, ivir2) = w_ab;

    // (nbf,nbf) = (nbf,nmo)*(nmo,nmo)*(nbf,nmo)^T
    m_w2ao = Cmat * m_w2mo * Cmat.t();

    //
    // Compute the HF energy weighted density matrix
    //

    arma::mat m_whf = energyWeightDensityMatrix();

    const auto nshells = bs.size();
    const auto shell2atom = bs.shell2atom(atoms);

    arma::mat Schwartz = ints.Km;

    arma::vec mp_grd(3 * natom, arma::fill::zeros);

    libint2::Engine engine(libint2::Operator::coulomb, bs.max_nprim(),
                           bs.max_l(), 1);

    const auto precision = std::numeric_limits<double>::epsilon();
    engine.set_precision(precision);
    const auto &buf = engine.results();

    auto shell2bf = bs.shell2bf();

    arma::vec v_uv(nbf * nbf, arma::fill::zeros);

    // Two Electron Derivatives

    for (auto s1 = 0; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];
      auto nbf1 = bs[s1].size();
      auto iat = shell2atom[s1];

      for (auto s2 = 0; s2 <= s1; ++s2) {
        auto bf2_first = shell2bf[s2];
        auto nbf2 = bs[s2].size();
        auto jat = shell2atom[s2];
        auto s12_cut = Schwartz(s1, s2);

        if (s12_cut < precision)
          continue;

        //--- 3,4 backtransform
        arma::mat sh12_34_bt(nbf * nbf, nbf1 * nbf2, arma::fill::zeros);
        arma::mat sh21_34_bt(nbf * nbf, nbf1 * nbf2, arma::fill::zeros);
        {
          arma::vec v_t_ov(naocc * navir, arma::fill::zeros);
          arma::mat m_t_ov(v_t_ov.memptr(), navir, naocc, false);

          arma::vec v_uv(nbf * nbf, arma::fill::zeros);
          arma::mat m_t_uv(v_uv.memptr(), nbf, nbf, false);
          for (auto f1 = 0, f12 = 0; f1 != nbf1; f1++) {
            auto bf1 = f1 + bf1_first;

            for (auto f2 = 0; f2 != nbf2; f2++, f12++) {
              auto bf2 = f2 + bf2_first;
              v_t_ov = m_12_bt.col(bf1 * nbf + bf2);
              m_t_uv = Cvir * m_t_ov * Caocc.t();
              sh12_34_bt.col(f12) = v_uv;

              v_t_ov = m_12_bt.col(bf2 * nbf + bf1);
              m_t_uv = Cvir * m_t_ov * Caocc.t();
              sh21_34_bt.col(f12) = v_uv;
            }
          }

          sh12_34_bt *= 2.0;
          sh21_34_bt *= 2.0;
        }

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

            if (s12_cut * s34_cut < precision)
              continue;

            auto s12_deg = (s1 == s2) ? 0.5 : 1.0;
            auto s34_deg = (s3 == s4) ? 0.5 : 1.0;
            auto s13_s24_deg = (s1 == s3) ? ((s2 == s4) ? 0.5 : 1.0) : 1.0;
            auto s1234_deg = s12_deg * s34_deg * s13_s24_deg;

            engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx,
                            1>(bs[s1], bs[s2], bs[s3], bs[s4]);

            arma::vec mp_tmp(9, arma::fill::zeros);

            for (auto di = 0; di != 9; di++) {
              const auto shset = buf[di];

              if (shset == nullptr)
                continue;

              double mp_sum = 0.0;
              // double hf_sum = 0.0;
              for (auto f1 = 0, f12 = 0, f1234 = 0; f1 != nbf1; f1++) {
                auto bf1 = f1 + bf1_first;

                for (auto f2 = 0; f2 != nbf2; f2++, f12++) {
                  auto bf2 = f2 + bf2_first;
                  auto ijb = bf1 * (bf1 + 1) / 2 + bf2;

                  for (auto f3 = 0; f3 != nbf3; f3++) {
                    auto bf3 = f3 + bf3_first;

                    for (auto f4 = 0; f4 != nbf4; f4++, f1234++) {
                      auto bf4 = f4 + bf4_first;
                      auto klb = bf3 * (bf3 + 1) / 2 + bf4;

                      auto eri4 = shset[f1234];
                      auto value = eri4 * s1234_deg;
                      /*
                                auto mp_tval =
                                4.0*m_phf(bf1,bf2)*(m_phf(bf3,bf4) +
                         m_p2ao(bf3,bf4))
                                + 4.0*m_phf(bf3,bf4)*m_p2ao(bf1,bf2)
                                -     m_phf(bf2,bf4)*(m_phf(bf1,bf3) +
                         m_p2ao(bf1,bf3))
                                -     m_phf(bf2,bf3)*(m_phf(bf1,bf4) +
                         m_p2ao(bf1,bf4))
                                -     m_phf(bf1,bf4)*m_p2ao(bf2,bf3)
                                -     m_phf(bf1,bf3)*m_p2ao(bf2,bf4);
                      */
                      auto mp_tval = 4.0 * m_phf(bf1, bf2) * m_pmp2(bf3, bf4) +
                                     4.0 * m_phf(bf3, bf4) * m_p2ao(bf1, bf2) -
                                     m_phf(bf2, bf4) * m_pmp2(bf1, bf3) -
                                     m_phf(bf2, bf3) * m_pmp2(bf1, bf4) -
                                     m_phf(bf1, bf4) * m_p2ao(bf2, bf3) -
                                     m_phf(bf1, bf3) * m_p2ao(bf2, bf4);

                      auto tval = sh12_34_bt(bf3 * nbf + bf4, f12) +
                                  sh21_34_bt(bf3 * nbf + bf4, f12) +
                                  sh12_34_bt(bf4 * nbf + bf3, f12) +
                                  sh21_34_bt(bf4 * nbf + bf3, f12);

                      mp_sum += (mp_tval + 2.0 * tval) * value;
                    }
                  }
                }
              }
              mp_tmp(di) = mp_sum;
              // hf_tmp(di) = hf_sum;
            } // exit di

            mp_grd(3 * iat) += mp_tmp(0);
            mp_grd(3 * iat + 1) += mp_tmp(1);
            mp_grd(3 * iat + 2) += mp_tmp(2);
            mp_grd(3 * jat) += mp_tmp(3);
            mp_grd(3 * jat + 1) += mp_tmp(4);
            mp_grd(3 * jat + 2) += mp_tmp(5);
            mp_grd(3 * kat) += mp_tmp(6);
            mp_grd(3 * kat + 1) += mp_tmp(7);
            mp_grd(3 * kat + 2) += mp_tmp(8);
            mp_grd(3 * lat) -= (mp_tmp(0) + mp_tmp(3) + mp_tmp(6));
            mp_grd(3 * lat + 1) -= (mp_tmp(1) + mp_tmp(4) + mp_tmp(7));
            mp_grd(3 * lat + 2) -= (mp_tmp(2) + mp_tmp(5) + mp_tmp(8));
          } // s4
        }   // s3

      } // exit s2
    }   // exit s1

    arma::mat mp_grd_eri(mp_grd.memptr(), 3, natom, true);

    // hcore: kinetic & nuclear-electron energy contribution

    arma::mat mp_grd_t =
        compute_1body_ints_deriv<libint2::Operator::kinetic>(bs, atoms, m_pmp2);

    arma::mat mp_grd_v =
        compute_1body_nuclear_deriv(bs, atoms, m_pmp2, atoms_Q);

    // overlap
    arma::mat m_wmp2 = m_whf - m_w2ao; //??

    arma::mat mp_grd_s =
        compute_1body_ints_deriv<libint2::Operator::overlap>(bs, atoms, m_wmp2);

    arma::mat mp_grd_rep = compute_nuclear_gradient(atoms, atoms_Q);

    const int natom_q = atoms_Q.size();

    m_grad = arma::mat(3, natom + natom_q, arma::fill::zeros);

    for (auto ia = 0; ia < natom; ia++) {
      m_grad.col(ia) +=
          //(mp_grd_2pdm.col(ia)
          (mp_grd_t.col(ia) + mp_grd_v.col(ia) - mp_grd_s.col(ia) +
           mp_grd_rep.col(ia) + mp_grd_eri.col(ia));
    }

    for (auto ia = natom; ia < natom + natom_q; ia++) {
      m_grad.col(ia) += mp_grd_v.col(ia) + mp_grd_rep.col(ia);
    }

    if (l_print) {
      std::cout << "Total Gradient :\n";
      std::cout << m_grad.t();
    }
  }
}

} // namespace qcmol
} // namespace willow
