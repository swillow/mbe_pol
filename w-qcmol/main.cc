#if __cplusplus <= 199711L
#error "QCMOL requires C++11 support"
#endif

// standard C++ headers
#include "Integrals.hpp"
#include "MP2.hpp"
#include "Molecule.hpp"
#include "RHF.hpp"
#include "esp.hpp"
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace libint2;
using namespace willow;

int main(int argc, char *argv[]) {

  try {
    // Initialize molecule

    libint2::initialize();

    cout << std::setprecision(10);
    cout << std::fixed;

    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

    vector<Atom> atoms = qcmol::read_geometry(filename);
    //--
    if (filename == "cage.xyz") {
      vector<qcmol::QAtom> atoms_Q = {
          {1, 5.830098634e-01, 1.747649029e+00, 2.374118142e-01, -0.82369},
          {2, 6.763772708e-02, 1.388446541e+00, -5.225068626e-01, 0.48143},
          {3, 4.921571765e-01, 2.699373871e+00, 1.984531638e-01, 0.34226},
          {4, -6.913716823e-01, -4.866230560e-01, 1.662094357e+00, -0.83501},
          {5, -1.966956586e-01, -1.092424055e+00, 1.095426341e+00, 0.41222},
          {6, -2.750255599e-01, 3.667626592e-01, 1.487915248e+00, 0.42279},
          {7, 2.806577430e+00, 1.078807054e-01, 1.105409846e-01, -0.80161},
          {8, 2.167450080e+00, 8.356004598e-01, 1.980781347e-01, 0.41908},
          {9, 3.312598851e+00, 1.076955035e-01, 9.236267319e-01, 0.38253},
          {10, -2.902521997e+00, 4.487878460e-03, 7.438771560e-02, -0.82897},
          {11, -2.270920381e+00, -2.485516133e-01, 7.714953391e-01, 0.43249},
          {12, -3.659225202e+00, -5.711763469e-01, 1.767583601e-01, 0.39648}};
    }

    std::cout << atoms_Q.size() << std::endl;
    //----
    // Create Basis Set
    //    libint2::Shell::do_enforce_unit_normalization(true);
    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::BasisSet bs("aug-cc-pVDZ", atoms);
    //    libint2::BasisSet bs  ("6-311+g2d_p", atoms);
    //    libint2::BasisSet bs  ("6-311++g2d_2p", atoms);
    //    libint2::BasisSet bs  ("6-311-w", atoms);
    // libint2::BasisSet bs  ("6-31G", atoms);

    qcmol::Integrals ints(atoms, bs, atoms_Q);

    // (atoms, bs, ints, l_print, l_grad, qm_chg, atoms_Q)
    // qcmol::RHF rhf (atoms,bs,ints,true,true);
    const int qm_chg = 0;
    /*
        qcmol::ESP esp (atoms, bs, ints,false,false, qm_chg, atoms_Q);
        arma::vec esp_chg = esp.get_atomic_charges();
        std::cout << esp_chg;
    */

    qcmol::MP2 mp2(atoms, bs, ints, true, true, qm_chg, atoms_Q);

    std::cout << "MP2 Energy \n";
    std::cout << mp2.get_energy() << std::endl;

    arma::mat mp2_grd = mp2.get_gradient();
    arma::vec sum(3, arma::fill::zeros);
    for (auto i = 0; i < mp2_grd.n_cols; ++i) {
      sum += mp2_grd.col(i);
    }
    std::cout << "MP2 Gradient\n";
    std::cout << mp2_grd.t();
    std::cout << sum.t();

    libint2::finalize();

  } // end of try block;

  catch (const char *ex) {
    cerr << "Caught exception: " << ex << endl;
    return 1;
  } catch (string &ex) {
    cerr << "Caught exception: " << ex << endl;
    return 1;
  } catch (exception &ex) {
    cerr << ex.what() << endl;
    return 1;
  } catch (...) {
    cerr << "Caught Unknown Exception: " << endl;
    return 1;
  }

  return 0;
}
