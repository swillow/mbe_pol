# mbe_pol (Many-Body Exansion with Polarization Energy)

MBE-pol is developed based on fragment-based method for quantum mechanics (QM) calculations on larger molecular systems and condensed phases as well. 

The potential energy of the system is defined as 

$E = E_\mathrm{MBE(2)} + \kappa \sum_I E_I^\mathrm{pol},$

with

$E_\mathrm{MBE(2)}  =  \sum_I ( E_{I} - E_I^\mathrm{min}) + \sum_{I > J}  ( E_{IJ} -
E_{I} - E_{J})$

$E_I^\mathrm{pol} 
 =  \langle \Psi_{I:Q_I} | \hat{H}_{I:Q_I} | \Psi_{I:Q_I} \rangle -
\langle \Psi_I | \hat{H}_{I:Q_I} | \Psi_I \rangle$

Here, 
$\hat{H}_{I:Q_I}  =  \hat{H}_I  + \sum_{i \in I} \sum_{B \in Q_I} \frac{q_B}{ | \pmb{r}_i - \pmb{R}_B |}  + \sum_{A \in I} \sum_{B \in Q_I} \frac{Z_A q_B}{R_{AB}}$


