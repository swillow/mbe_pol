# w-embem
w-embem is a simple program of the embedded many-body expansion method.

I used 'w-qcmol' to calculate the energy of RHF/aug-cc-pVDZ.

-- example: water hexamer (cage.xyz) ---

Method                    : CPU times :  Energy
RHF/aug-cc-pVDZ (w-qcmol) : 7min      : -456.29707
EMBEM (RHF/aug-cc-pVDZ)   : 34 sec    : -456.29728
 MBEM (RHF/aug-cc-pVDZ)   : 33 sec    : -456.28309
---
Here, MBEM (Many-Body Exapansion Method) calculates the energies of monomers and dimers without the embedding field.
EMBEM (Embedded Many-Body Expansion Method) calculates the energies of monomers and dimers with the embedding field.

