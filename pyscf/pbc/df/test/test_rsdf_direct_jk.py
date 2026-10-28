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
from pyscf.pbc import df, gto


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.M(
            atom='He 0 0 0; He 1.1 0.3 0.2', a=np.eye(3) * 5,
            unit='Bohr', basis={'He': [[0, [1.2, 1.]], [1, [0.8, 1.]]]},
            precision=1e-9, verbose=0)
        cls.dfopts = dict(
            auxbasis={'He': [[0, [1.4, 1.]], [1, [1.0, 1.]], [2, [0.9, 1.]]]},
            omega=0.5, omega_j2c=0.5, exp_to_discard=0)

    def check_jk(self, kpts):
        rng = np.random.default_rng(14)
        x = rng.standard_normal((len(kpts), self.cell.nao, self.cell.nao))
        if np.any(kpts):
            x = x + 1j * rng.standard_normal(x.shape)
        dm = x @ x.conj().transpose(0, 2, 1) / self.cell.nao
        reference = df.RSDF(self.cell, kpts).set(**self.dfopts).build()
        ref_j, ref_k = reference.get_jk(dm, kpts=kpts, exxdiv=None)
        for use_bvk in [False, True]:
            for semidirect in [False, True]:
                with self.subTest(use_bvk=use_bvk, semidirect=semidirect):
                    direct = df.RSDF(self.cell, kpts).set(
                        **self.dfopts, direct=True, use_bvk=use_bvk,
                        semidirect=semidirect).build()
                    actual_j, actual_k = direct.get_jk(dm, kpts=kpts, exxdiv=None)
                    self.assertEqual(actual_j.shape, ref_j.shape)
                    self.assertEqual(actual_k.shape, ref_k.shape)
                    np.testing.assert_allclose(actual_j, ref_j, atol=1e-8, rtol=0)
                    np.testing.assert_allclose(actual_k, ref_k, atol=1e-8, rtol=0)

    def test_gamma(self):
        self.check_jk(np.zeros((1, 3)))

    def test_shifted_single_kpt(self):
        kpts = self.cell.get_abs_kpts(np.array([[0.13, 0.07, 0.11]]))
        self.check_jk(kpts)

    def test_multiple_kpts(self):
        self.check_jk(self.cell.make_kpts([1, 1, 3]))


if __name__ == '__main__':
    unittest.main()
