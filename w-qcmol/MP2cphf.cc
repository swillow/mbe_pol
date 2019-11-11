#include "MP2.hpp"
#include "RHF.hpp"
#include <armadillo>

namespace willow {
namespace qcmol {

arma::mat MP2::cphf(const arma::mat &l_ai, const arma::vec &ao_tei,
                    const bool l_print) {

  const double epsilon = 1.0e-8;
  // arma::vec v_p_ai (nvir*nocc, arma::fill::zeros);
  // arma::mat pm_ai  (v_p_aj.memptr(), nvir, nocc, false);

  arma::mat Cvir = Cmat.submat(0, ivir1, nbf - 1, ivir2);
  arma::mat Cocc = Cmat.submat(0, 0, nbf - 1, iocc2);

  arma::vec v_Pm(nvir * nocc, arma::fill::zeros);
  arma::mat Pm(v_Pm.memptr(), nvir, nocc, false);

  for (auto moi = 0; moi < nocc; moi++) {
    for (auto am = 0; am < nvir; am++) {
      auto moa = ivir1 + am;
      Pm(am, moi) = l_ai(am, moi) / (eval(moa) - eval(moi));
    }
  }

  std::vector<arma::vec> Pm_list;
  Pm_list.push_back(v_Pm);

  arma::vec v_APm(nvir * nocc, arma::fill::zeros);
  arma::mat APm(v_APm.memptr(), nvir, nocc, false);
  std::vector<arma::vec> APm_list;

  // arma::mat Dm  (nbf,  nbf,  arma::fill::zeros);
  // arma::mat Gm  (nbf,  nbf,  arma::fill::zeros);

  arma::vec P_sum_old(nvir * nocc, arma::fill::zeros);
  arma::vec P_sum_new(nvir * nocc, arma::fill::zeros);

  const int maxiter = 30;

  for (auto iter = 0; iter < maxiter; ++iter) {

    const int ndim = iter + 1;

    // Compute A*P[i-1] (call AP matrix)
    v_Pm = Pm_list[iter];

    arma::mat Dm = Cvir * Pm * Cocc.t();
    arma::mat Gm = g_matrix(ao_tei, Dm);
    APm = 4.0 * Cvir.t() * Gm * Cocc;

    for (auto moi = 0; moi < nocc; moi++) {
      for (auto am = 0; am < nvir; am++) {
        auto moa = ivir1 + am;
        double tmpval = APm(am, moi) / (eval(moi) - eval(moa));
        APm(am, moi) = tmpval;
      }
    }

    APm_list.push_back(v_APm);

    // compute alpha
    // Solve the linear system of equations C*X = B
    // where C is matrix, X and B are vector
    // put result (x) into array alpha
    arma::vec alpha(ndim, arma::fill::zeros);
    {
      arma::vec norm(ndim, arma::fill::zeros);

      for (auto i = 0; i < ndim; i++) {
        norm(i) = arma::norm(Pm_list[i]);
      }

      // Construct matrix Cm
      arma::mat Cm(ndim, ndim, arma::fill::zeros);

      for (auto j = 0; j < ndim; j++) {
        for (auto i = 0; i < ndim; i++) {

          double tmp1 = -arma::dot(Pm_list[i], APm_list[j]);

          if (i == j) {
            tmp1 += arma::dot(Pm_list[i], Pm_list[i]);
          }

          Cm(i, j) = tmp1 / (norm(i) * norm(j));
        }
      }

      arma::vec Bv(ndim, arma::fill::zeros);
      Bv(0) = norm(0);

      arma::vec Xv = Cm.i() * Bv;

      for (auto i = 0; i < ndim; i++) {
        alpha(i) = Xv(i) / norm(i);
      }
    }

    // P_sum_new = alpha[0]*P[0] +
    P_sum_new.zeros();
    for (auto j = 0; j < ndim; j++) {
      P_sum_new += alpha(j) * Pm_list[j];
    }

    // Test for convergence
    // (based on RMS (P2aj_new - P2aj_old)
    // and max abs. val. of element

    double tmpval = 0.0;
    double maxabs = 0.0;
    for (auto j = 0; j < nocc * nvir; j++) {
      double tmpval1 = P_sum_new(j) - P_sum_old(j);
      double tmpval2 = tmpval1 * tmpval1;
      tmpval += tmpval2;
      if (tmpval2 > maxabs)
        maxabs = tmpval2;
    }

    if (std::sqrt(tmpval / (nocc * nvir)) < epsilon &&
        std::sqrt(maxabs) < epsilon)
      break;

    // Put P_sum_new into P_sum_old

    P_sum_old = P_sum_new;

    // Compute projection of A*P[i-1] on P[0], ..., P[i-1]

    arma::vec projctn(nocc * nvir, arma::fill::zeros);

    for (auto j = 0; j < iter + 1; j++) {
      double dot_prod = arma::dot(Pm_list[j], Pm_list[j]);
      double coef = arma::dot(Pm_list[j], APm) / dot_prod;

      projctn += coef * Pm_list[j];
    }

    Pm_list.push_back(v_APm - projctn);

    // Test for convergence (based on norm (Pm[i]))

    double rmsd = arma::norm(Pm_list[iter + 1]) / sqrt(nocc * nvir);
    // double rmsd = arma::norm (p_ai - p_ai_old);

    if (l_print)
      std::cout << "Iter CPHF " << iter << "   " << rmsd << std::endl;
    if (rmsd < epsilon)
      break;
  }

  // Converged vector is in P_sum_new

  arma::mat p_ai(P_sum_new.memptr(), nvir, nocc, true);

  return p_ai;
}

/*
arma:mat MP2::cphf_old ()
{
  double epsilon = 1.0e-8;

  for (auto moi = 0; moi < nocc; moi++) {
    for (auto am = 0; am < navir; am++) {
      auto moa = ivir1 + am;
      p_ai(am,moi) = l_ai(am,moi)/(eval(moa) - eval(moi));
    }
  }

  // Solve the CPHF equations (iteratively, with DIIS like method)
  int ii = 0;
  int niter = 0;
  const int maxiter = 30;
  for (auto iter = 0; iter != maxiter; ++iter) {

    arma::mat p_ai_old = p_ai;
    // Compute A*P[ii-1]
    // Dm(nbf,nbf) = (nbf,navir)*(navir,nocc)*(nbf,nocc)
    arma::mat Dm = Cvir*p_ai_old*Cocc.t();
    arma::mat Gm = g_matrix (ints.TEI, Dm);

    //  APm(navir,nocc) =
    arma::mat APm = 2.0*Cvir.t()*Gm*Cocc;

    for (auto moi = 0; moi < nocc; moi++) {
      for (auto am = 0; am < navir; am++) {
        auto moa = ivir1 + am;
        p_ai (am,moi) = (l_ai(am,moi) - APm(am,moi))/(eval(moa) - eval(moi));
      }
    }
    p_ai = 0.6*p_ai + 0.4*p_ai_old;

    double rmsd = arma::norm (p_ai - p_ai_old);

    std::cout << "Iter CPHF " << iter << "   " << rmsd << std::endl;
    if (rmsd < 1.0e-7) break;
  }

  std::cout << "p_ai \n";

  return p_ai;
}
*/

} // namespace qcmol
} // namespace willow
