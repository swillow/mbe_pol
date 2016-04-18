# w-embem
w-embem is a simple program of the embedded many-body expansion method.


I used 'w-qcmol' to calculate the energy of the water hexamer (cage.xyz) at RHF/aug-cc-pVDZ.

** Method  (CPU time) : Energy  
* RHF (7 min) : -456.29707
* EMBEM (34 sec) : -456.29728
* MBEM (33 sec) : -456.28309

Here, MBEM (Many-Body Expansion Method) calculates the energies of monomers and dimers without the embedding field.


EMBEM (Embedded Many-Body Expansion Method) calculate the energies of monomers and dimers with the embedding field.

