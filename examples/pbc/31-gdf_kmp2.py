#!/usr/bin/env python

import numpy as np

from pyscf.pbc import gto
from pyscf.pbc import scf


cell = gto.Cell()
cell.atom = '''
H 0.0 0.0 0.0
H 1.4 0.0 0.0
'''
cell.a = np.eye(3) * 5.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.precision = 1e-6
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([1, 1, 1])
kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
kmf.conv_tol = 1e-8
kmf.kernel()

assert kmf.converged

mymp = kmf.MP2()
emp2, _ = mymp.kernel(with_t2=False)

print(f'GDF_TEST_CLASS = {mymp.__class__.__name__}')
print(f'GDF_TEST_E_HF = {kmf.e_tot:.15f}')
print(f'GDF_TEST_E_MP2 = {emp2:.15f}')
print(f'GDF_TEST_E_TOTAL = {mymp.e_tot:.15f}')
print('GDF_TEST_STATUS = PASS')
