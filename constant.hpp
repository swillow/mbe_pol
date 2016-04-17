#ifndef W_CONSTANTS_HPP
#define W_CONSTANTS_HPP

namespace willow {

// --- (2010 CODATA)---


const double kB        =  1.3806488e-23; // J/K
const double au_len    = 0.52917721092;   // A
const double au_energy = 4.35974434e-18; //    ! J
const double au_time   = 2.418884326502e-17; // ! s
const double au_charge = 1.602176565e-19; //  ! C
const double au_freq   = 2.19475e5;    //  ! au--> cm-1 
const double au_kcal   = 627.509469; // ! kcal

const double ang2bohr  = 1.0/au_len;  //   ! A    --> bohr
const double bohr2ang  = au_len; //        ! bohr --> A

const double boltz     = kB/au_energy; //    ! K in atomic unit

/*   atomic mass unit   */
const double  amu_mass  = 1.660538921e-27; // ! kg 1.6605402d-27 
//  ! plack constant
const double hbar = 1.0;



}



#endif
