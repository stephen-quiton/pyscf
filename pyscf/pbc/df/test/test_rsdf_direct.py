# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import rsdf_direct_helper as helper
from pyscf.pbc.df.incore import make_auxcell


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.M(
            atom='He 0 0 0; He 1.1 0.3 0.2', a=np.eye(3) * 5,
            unit='Bohr', basis={'He': [[0, [1.2, 1.]], [1, [0.8, 1.]]]},
            precision=1e-9, verbose=0)
        cls.auxcell = make_auxcell(cls.cell, {
            'He': [[0, [1.4, 1.]], [1, [1.0, 1.]], [2, [0.9, 1.]]]})
        cls.kpts = cls.cell.get_abs_kpts(np.array([
            [0.13, 0.07, 0.11], [-0.09, 0.05, 0.04], [0.03, -0.11, 0.08]]))

    def check_order(self, pairs, bvk_kmesh=None):
        cell, auxcell = self.cell, self.auxcell
        nao, naux = cell.nao_nr(), auxcell.nao_nr()
        # Rectangular AO blocks and nonzero offsets in all three shell ranges.
        shls_slice = (1, cell.nbas, 1, cell.nbas - 1, 1, auxcell.nbas)
        i0, i1, j0, j1 = cell.ao_loc_nr()[list(shls_slice[:4])]
        l0, l1 = auxcell.ao_loc_nr()[list(shls_slice[4:])]
        intors = [('int3c2e', 1)]
        # Exercise components on the complex kernels restored here.  The
        # existing Gamma derivative accumulator has overlapping scratch space.
        if np.any(pairs):
            intors.append(('int3c2e_ip1', 3))
        for intor, comp in intors:
            with self.subTest(intor=intor):
                kwargs = dict(intor=intor, aosym='s1', kptij_lst=pairs,
                              bvk_kmesh=bvk_kmesh)
                reference = helper.aux_e2_nospltbas(
                    cell, auxcell, 0.5, j3c_order='ijL', **kwargs)
                actual = helper.aux_e2_nospltbas(
                    cell, auxcell, 0.5, j3c_order='Lij', **kwargs)
                self.assertEqual(actual.shape, (len(pairs), comp, naux, nao**2))
                np.testing.assert_allclose(actual, reference.transpose(0, 1, 3, 2),
                                           atol=1e-12, rtol=0)
                if np.any(pairs):
                    self.assertTrue(np.iscomplexobj(actual))
                    self.assertGreater(np.max(abs(reference.imag)), 1e-8)

                reference_slice = helper.aux_e2_nospltbas(
                    cell, auxcell, 0.5, j3c_order='ijL',
                    shls_slice=shls_slice, **kwargs)
                actual_slice = helper.aux_e2_nospltbas(
                    cell, auxcell, 0.5, j3c_order='Lij',
                    shls_slice=shls_slice, **kwargs)
                np.testing.assert_allclose(
                    actual_slice, reference_slice.transpose(0, 1, 3, 2),
                    atol=1e-12, rtol=0)
                expected_slice = reference.transpose(0, 1, 3, 2).reshape(
                    len(pairs), comp, naux, nao, nao)[:, :, l0:l1, i0:i1, j0:j1]
                np.testing.assert_allclose(
                    actual_slice, expected_slice.reshape(actual_slice.shape),
                    atol=1e-12, rtol=0)

    def test_gamma(self):
        self.check_order(np.zeros((1, 2, 3)))

    def test_shifted_single_kpt(self):
        self.assertIsNone(helper.kpts_to_kmesh(self.cell, self.kpts[:1]))
        self.check_order(np.array([[self.kpts[0], self.kpts[0]]]))

    def test_non_bvk_diagonal_kpts(self):
        kpts = self.kpts[[2, 0, 1]]
        self.check_order(np.stack([kpts, kpts], axis=1))

    def test_non_bvk_kpt_pairs(self):
        # A sparse, reordered list, including a diagonal and a repeated pair.
        pairs = self.kpts[np.array([[2, 0], [0, 1], [1, 2], [2, 2], [2, 0]])]
        self.check_order(pairs)

    def test_bvk_diagonal_kpts(self):
        kpts = self.cell.make_kpts([1, 1, 3])[[2, 0, 1]]
        self.check_order(np.stack([kpts, kpts], axis=1), bvk_kmesh=[1, 1, 3])


if __name__ == '__main__':
    unittest.main()
