#!/usr/bin/env python

'''All-electron HF with stored smoothed truncated Coulomb exchange.

The smoothing parameter eta=3 is usually sufficient for gapped systems.
The conservative default eta=4 is recommended for metallic systems.
'''

import numpy as np

from pyscf.pbc import df, gto, scf
from pyscf.pbc.df import rsdf_stc


cell = gto.M(
    a=np.eye(3) * 5.0,
    atom='He 0 0 0',
    basis='cc-pvdz',
    precision=1e-8,
)
kpts = cell.make_kpts([2, 1, 1])

mf = scf.KRHF(cell, kpts)
mf = rsdf_stc.density_fit(mf, auxbasis='weigend', eta=3.0)
mf.kernel()

# FFTDF_STC is a numerical reference (mainly useful with pseudopotentials).
fft_reference = df.FFTDF_STC(cell, kpts, eta=3.0)
