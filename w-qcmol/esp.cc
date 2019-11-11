#include "esp.hpp"

namespace willow {
namespace qcmol {

const double ang2bohr = 1.0 / 0.52917721067; // 2014 CODATA value

static std::vector<arma::vec3>
esp_grid(const std::vector<libint2::Atom> &atoms);
static arma::vec esp_esp(const std::vector<libint2::Atom> &atoms,
                         const libint2::BasisSet &bs, const arma::mat &Dm,
                         const std::vector<arma::vec3> &grids);

static arma::vec esp_fit(const std::vector<libint2::Atom> &atoms,
                         const std::vector<arma::vec3> &grids,
                         const arma::vec &grids_val, const bool &l_resp);

arma::vec esp_atomic_charges(const std::vector<libint2::Atom> &atoms,
                             const libint2::BasisSet &bs, const arma::mat &Dm,
                             const bool &l_resp) {
  std::vector<arma::vec3> grids = esp_grid(atoms);

  arma::vec grids_val = esp_esp(atoms, bs, Dm, grids);

  arma::vec qf = esp_fit(atoms, grids, grids_val, l_resp);

  return qf;
}

std::vector<arma::vec3> esp_grid(const std::vector<libint2::Atom> &atoms) {

  arma::vec radius(108, arma::fill::zeros);

  // A
  radius(0) = 0.0;   // Ghost
  radius(1) = 0.3;   // H
  radius(2) = 1.22;  // He
  radius(3) = 1.23;  // Li
  radius(4) = 0.89;  // Be
  radius(5) = 0.88;  // B
  radius(6) = 0.77;  // C
  radius(7) = 0.70;  // N
  radius(8) = 0.66;  // O
  radius(9) = 0.58;  // F
  radius(10) = 1.60; // Ne
  radius(11) = 1.40; // Na
  radius(12) = 1.36; // Mg
  radius(13) = 1.25; // Al
  radius(14) = 1.17; // Si
  radius(15) = 1.10; // P
  radius(16) = 1.04; // S
  radius(17) = 0.99; // Cl
  radius(18) = 1.91; // Ar
  radius(20) = 1.74;
  radius(21) = 1.44;
  radius(22) = 1.32;
  radius(23) = 1.22;
  radius(24) = 1.19;
  radius(25) = 1.17;
  radius(26) = 1.17;
  radius(27) = 1.16;
  radius(28) = 1.15;
  radius(29) = 1.17;
  radius(30) = 1.25;
  radius(31) = 1.25;
  radius(32) = 1.22;
  radius(33) = 1.21;
  radius(34) = 1.17;
  radius(35) = 1.14;
  radius(36) = 1.98;
  radius(37) = 2.22;
  radius(38) = 1.92;
  radius(39) = 1.62;
  radius(40) = 1.45;
  radius(41) = 1.34;
  radius(42) = 1.29;
  radius(43) = 1.27;
  radius(44) = 1.24;
  radius(45) = 1.25;
  radius(46) = 1.28;
  radius(47) = 1.34;
  radius(48) = 1.41;
  radius(49) = 1.50;
  radius(50) = 1.40;
  radius(51) = 1.41;
  radius(52) = 1.37;
  radius(53) = 1.33;
  radius(54) = 2.09;
  radius(55) = 2.35;
  radius(56) = 1.98;
  radius(57) = 1.69;
  radius(58) = 1.65;
  radius(59) = 1.65;
  radius(60) = 1.64;
  radius(61) = 1.65;
  radius(62) = 1.66;
  radius(63) = 1.65;
  radius(64) = 1.61;
  radius(65) = 1.59;
  radius(66) = 1.59;
  radius(67) = 1.58;
  radius(68) = 1.57;
  radius(69) = 1.56;
  radius(70) = 1.56;
  radius(71) = 1.56;
  radius(72) = 1.44;
  radius(73) = 1.34;
  radius(74) = 1.30;
  radius(75) = 1.28;
  radius(76) = 1.26;
  radius(77) = 1.26;
  radius(78) = 1.29;
  radius(79) = 1.34;
  radius(80) = 1.44;
  radius(81) = 1.55;
  radius(82) = 1.54;
  radius(83) = 1.52;
  radius(84) = 1.53;
  radius(85) = 1.50;
  radius(86) = 2.20;
  radius(87) = 3.24;
  radius(88) = 2.68;
  radius(89) = 2.25;
  radius(90) = 2.16;
  radius(91) = 1.93;
  radius(92) = 1.66;
  radius(93) = 1.57;
  radius(94) = 1.81;
  radius(95) = 2.21;
  radius(96) = 1.43;
  radius(97) = 1.42;
  radius(98) = 1.40;
  radius(99) = 1.39;
  radius(100) = 1.38;
  radius(101) = 1.37;
  radius(102) = 1.36;
  radius(103) = 1.34;
  radius(104) = 1.30;
  radius(105) = 1.30;
  radius(106) = 1.30;
  radius(107) = 1.30;

  for (auto i = 1; i <= 107; i++) {
    radius(i) = (radius(i) + 0.7) * ang2bohr;
  }
  double grd_min_x = atoms[0].x;
  double grd_min_y = atoms[0].y;
  double grd_min_z = atoms[0].z;

  double grd_max_x = atoms[0].x;
  double grd_max_y = atoms[0].y;
  double grd_max_z = atoms[0].z;

  for (auto ia = 1; ia < atoms.size(); ++ia) {

    grd_min_x = std::min(grd_min_x, atoms[ia].x);
    grd_min_y = std::min(grd_min_y, atoms[ia].y);
    grd_min_z = std::min(grd_min_z, atoms[ia].z);

    grd_max_x = std::max(grd_max_x, atoms[ia].x);
    grd_max_y = std::max(grd_max_y, atoms[ia].y);
    grd_max_z = std::max(grd_max_z, atoms[ia].z);
  }
  //
  // calculate the grid size
  //
  // A --> au
  const double rcut = 3.0 * ang2bohr;
  const double rcut2 = rcut * rcut;
  const double spac = 0.5 * ang2bohr;

  const int ngrid_x = int((grd_max_x - grd_min_x + 2.0 * rcut) / spac) + 1;
  const int ngrid_y = int((grd_max_y - grd_min_y + 2.0 * rcut) / spac) + 1;
  const int ngrid_z = int((grd_max_z - grd_min_z + 2.0 * rcut) / spac) + 1;

  const auto small = 1.0e-8;
  arma::vec3 vg;

  std::vector<arma::vec3> grids;

  for (auto iz = 0; iz < ngrid_z; ++iz)
    for (auto iy = 0; iy < ngrid_y; ++iy)
      for (auto ix = 0; ix < ngrid_x; ++ix) {

        vg.zeros();

        vg(0) = grd_min_x - rcut + ix * spac;
        vg(1) = grd_min_y - rcut + iy * spac;
        vg(2) = grd_min_z - rcut + iz * spac;

        double dmin = rcut2;

        bool lupdate = true;

        for (auto ia = 0; ia < atoms.size(); ++ia) {
          int iz = atoms[ia].atomic_number;
          auto rad2 = radius(iz) * radius(iz);

          auto dx = vg(0) - atoms[ia].x;
          auto dy = vg(1) - atoms[ia].y;
          auto dz = vg(2) - atoms[ia].z;

          auto d2 = dx * dx + dy * dy + dz * dz;

          if ((rad2 - d2) > small) {
            lupdate = false;
            break;
          }

          if ((dmin - d2) > small)
            dmin = d2;
        }

        if (lupdate) {
          if ((rcut2 - dmin) > small) {
            grids.push_back(vg);
          }
        }
      }

  return grids;
}

arma::vec esp_esp(const std::vector<libint2::Atom> &atoms,
                  const libint2::BasisSet &bs, const arma::mat &Dm,
                  const std::vector<arma::vec3> &grids) {

  const auto nsize = grids.size();

  arma::vec result(nsize, arma::fill::zeros);

  const auto nshells = bs.size();
  const auto nbf = bs.nbf();

  libint2::Engine engine(libint2::Operator::nuclear, bs.max_nprim(), bs.max_l(),
                         0);

  const auto &buf = engine.results();

  const auto shell2bf = bs.shell2bf();

  arma::mat Vm(nbf, nbf);

  for (auto ig = 0; ig < grids.size(); ++ig) {

    const auto xg = grids[ig](0);
    const auto yg = grids[ig](1);
    const auto zg = grids[ig](2);

    // Interaction with electron density
    std::vector<std::pair<double, std::array<double, 3>>> q;
    q.push_back({1.0, {{xg, yg, zg}}});

    engine.set_params(q);

    // calc Vm
    Vm.zeros();

    for (auto s1 = 0; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1];
      auto nbf1 = bs[s1].size();

      for (auto s2 = 0; s2 != nshells; ++s2) {
        auto bf2 = shell2bf[s2];
        auto nbf2 = bs[s2].size();

        engine.compute(bs[s1], bs[s2]);
        const auto *buf0 = buf[0];

        for (auto ib = 0, ij = 0; ib < nbf1; ib++) {
          for (auto jb = 0; jb < nbf2; jb++, ij++) {
            const double val = buf0[ij];
            Vm(bf1 + ib, bf2 + jb) = val;
            Vm(bf2 + jb, bf1 + ib) = val;
          }
        } // ib

      } // s2
    }   // s1

    //
    // get electrostatic potential on the grid points
    // -- from electron density
    double gval = 0.0;

    for (auto i = 0; i < nbf; i++)
      for (auto j = 0; j < nbf; j++)
        gval += Dm(i, j) * Vm(i, j);

    // get electrostatic potential on the grid points
    // -- from nuclei
    auto zval = 0.0;

    for (auto ia = 0; ia < atoms.size(); ++ia) {
      auto xij = atoms[ia].x - xg;
      auto yij = atoms[ia].y - yg;
      auto zij = atoms[ia].z - zg;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      zval += static_cast<double>(atoms[ia].atomic_number) / r;
    }

    result(ig) = gval + zval;
  }

  return result;
}

arma::vec esp_fit(const std::vector<libint2::Atom> &atoms,
                  const std::vector<arma::vec3> &grids,
                  const arma::vec &grids_val, const bool &l_resp) {

  // set up matrix of linear coefficients
  auto natoms = atoms.size();
  auto ndim = natoms + 1;

  arma::mat am(ndim, ndim, arma::fill::zeros);
  arma::vec bv(ndim, arma::fill::zeros);

  for (auto i = 0; i < atoms.size(); ++i) {
    auto xi = atoms[i].x;
    auto yi = atoms[i].y;
    auto zi = atoms[i].z;

    for (auto j = i; j < atoms.size(); ++j) {
      auto xj = atoms[j].x;
      auto yj = atoms[j].y;
      auto zj = atoms[j].z;

      auto sum = 0.0;

      for (auto k = 0; k < grids.size(); ++k) {
        auto xg = grids[k](0);
        auto yg = grids[k](1);
        auto zg = grids[k](2);

        auto rig2 = (xi - xg) * (xi - xg) + (yi - yg) * (yi - yg) +
                    (zi - zg) * (zi - zg);
        auto rjg2 = (xj - xg) * (xj - xg) + (yj - yg) * (yj - yg) +
                    (zj - zg) * (zj - zg);

        sum += 1.0 / sqrt(rig2 * rjg2);
      }

      am(i, j) = sum;
      am(j, i) = sum;
    }

    am(i, natoms) = 1.0;
    am(natoms, i) = 1.0;
  }

  // construct column vector b

  for (auto i = 0; i < atoms.size(); ++i) {
    auto xi = atoms[i].x;
    auto yi = atoms[i].y;
    auto zi = atoms[i].z;

    auto sum = 0.0;
    for (auto k = 0; k < grids.size(); ++k) {
      auto xg = grids[k](0);
      auto yg = grids[k](1);
      auto zg = grids[k](2);
      auto val = grids_val(k);

      auto rig2 =
          (xi - xg) * (xi - xg) + (yi - yg) * (yi - yg) + (zi - zg) * (zi - zg);

      sum += val / sqrt(rig2);
    }
    bv(i) = sum;
  }
  bv(natoms) = 0.0; // b(natoms) = charge;

  arma::mat am_inv = am.i();

  arma::vec qf(natoms, arma::fill::zeros);
  // No Reconstraint

  for (auto i = 0; i < natoms; ++i) {
    auto sum = 0.0;

    for (auto j = 0; j < ndim; ++j) {
      sum = sum + am_inv(i, j) * bv(j);
    }
    qf(i) = sum;
  }

  if (l_resp) {
    // start RESP
    arma::vec qf_keep(natoms, arma::fill::zeros);

    arma::mat am_keep = am;
    int niter = 0;
    while (niter < 25) {
      niter++;

      am = am_keep;

      for (auto i = 0; i < natoms; ++i) {
        if (atoms[i].atomic_number != 1) {
          am(i, i) = am_keep(i, i) + 0.001 / sqrt(qf(i) * qf(i) + 0.1 * 0.1);
        }
      }

      am_inv = am.i();

      for (auto i = 0; i < natoms; ++i) {
        auto sum = 0.0;

        for (auto j = 0; j < ndim; ++j) {
          sum = sum + am_inv(i, j) * bv(j);
        }
        qf(i) = sum;
      }

      auto difm = 0.0;
      for (auto i = 0; i < natoms; ++i) {
        auto dif = (qf(i) - qf_keep(i)) * (qf(i) - qf_keep(i));
        if (difm < dif)
          difm = dif;
      }

      difm = sqrt(difm);

      qf_keep = qf;

      if (difm < 1.0e-4)
        break;
    }
  } // if (l_resp)

  return qf;
}

} // namespace qcmol
} // namespace willow
