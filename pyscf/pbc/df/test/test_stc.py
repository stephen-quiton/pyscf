#!/usr/bin/env python

import unittest

import numpy as np

from pyscf.pbc import df, gto, scf, tools
from pyscf.pbc.df import fft_stc, rsdf_stc


def make_cell(mesh=(17, 15, 13), precision=1e-7, pseudo='gth-pade'):
    return gto.M(
        a=np.asarray([[4.0, .2, 0.], [0., 5.0, .1], [.1, 0., 6.0]]),
        atom='He 0 0 0',
        basis='gth-szv',
        pseudo=pseudo,
        mesh=mesh,
        precision=precision,
        verbose=0,
    )


class KnownValues(unittest.TestCase):
    def test_fft_kernel_equation_and_shifted_mesh(self):
        cell = make_cell()
        kpts = cell.make_kpts([2, 1, 1], scaled_center=[.5, 0., 0.])
        mydf = df.FFTDF_STC(cell, kpts).build()
        kpt = kpts[1] - kpts[0]
        coulG = fft_stc.get_coulG(mydf, kpt, cell.mesh)
        vtc, q2, omega = fft_stc._truncated_coulG(mydf, kpt, cell.mesh)
        damping = np.exp(-q2 / (4 * omega**2))
        ref = vtc * damping
        mask = q2 != 0
        ref[mask] += 4 * np.pi / q2[mask] * (1 - damping[mask])
        ref[~mask] = vtc[~mask] + np.pi / omega**2
        self.assertAlmostEqual(abs(coulG - ref).max(), 0., 12)
        coulG0 = fft_stc.get_coulG(mydf, np.zeros(3), cell.mesh)
        vtc0, q20, omega = fft_stc._truncated_coulG(
            mydf, np.zeros(3), cell.mesh)
        self.assertAlmostEqual(
            coulG0[q20 == 0].item(),
            (vtc0[q20 == 0] + np.pi / omega**2).item(), 12)
        self.assertTrue(np.array_equal(mydf._stc_kmesh, [2, 1, 1]))
        self.assertAlmostEqual(
            mydf.omega_stc,
            mydf.eta / tools.get_ws_inradius(cell.lattice_vectors(), [2, 1, 1]),
            12)
        copied = mydf.copy()
        self.assertIsNot(copied._ws_exx, mydf._ws_exx)
        self.assertIsNot(copied._ws_exx['vq_cache'], mydf._ws_exx['vq_cache'])
        copied.reset()
        self.assertIsNone(copied._ws_exx)

    def test_rsdf_long_range_kernel(self):
        cell = make_cell()
        kpts = cell.make_kpts([1, 1, 1])
        mydf = df.RSGDF_STC(cell, kpts, exxdiv='vcut_sph', rc_type='sph')
        mydf._rs_build()
        builder = rsdf_stc._RSGDFBuilder_STC(cell, mydf.auxcell, kpts)
        builder.__dict__.update(mydf.__dict__)
        weighted = builder.weighted_coulG(mesh=cell.mesh)
        vtc, q2, omega = fft_stc._truncated_coulG(builder, np.zeros(3), cell.mesh)
        ref = vtc * np.exp(-q2 / (4 * omega**2))
        ref[q2 == 0] = vtc[q2 == 0] + np.pi / omega**2
        weights = cell.get_Gv_weights(cell.mesh)[2]
        self.assertAlmostEqual(abs(weighted - ref * weights).max(), 0., 12)

    def test_j_is_ordinary_and_post_hf_is_rejected(self):
        cell = make_cell()
        kpts = cell.make_kpts([1, 1, 1])
        stc_df = df.RSGDF_STC(cell, kpts)
        j_df = df.GDF(cell, kpts)
        mf = rsdf_stc.density_fit(
            scf.KRHF(cell, kpts), with_df=stc_df, with_df_j=j_df)
        self.assertIs(mf.with_df, stc_df)
        self.assertIs(mf.with_df.with_df_j, j_df)
        dm = np.eye(cell.nao_nr())
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref = mf.with_df.with_df_j.get_jk(dm, kpts=kpts, with_k=False)[0]
        self.assertAlmostEqual(abs(vj - ref).max(), 0., 12)
        self.assertIsNone(mf.exxdiv)
        with self.assertRaisesRegex(NotImplementedError, 'HF exchange'):
            mf.with_df.get_eri()
        with self.assertRaisesRegex(RuntimeError, 'HF-exchange-only'):
            from pyscf.pbc.mp import kmp2
            kmp2.KMP2(mf)
        with self.assertRaisesRegex(RuntimeError, 'HF-exchange-only'):
            from pyscf.pbc.mp import kmp2_direct
            kmp2_direct.KMP2_direct(mf)

    def test_stored_rsdf_matches_fft_reference(self):
        cell = gto.M(
            a=np.eye(3) * 4., atom='He 0 0 0', basis='gth-szv',
            pseudo='gth-pade', mesh=[61] * 3, precision=1e-10, verbose=0)
        kpts = cell.make_kpts([1, 1, 1])
        mydf = df.RSGDF_STC(cell, kpts, exxdiv='vcut_sph', rc_type='sph')
        mydf.with_df_j = df.GDF(cell, kpts)
        mydf.build()
        fftdf = df.FFTDF_STC(cell, kpts, exxdiv='vcut_sph', rc_type='sph')
        dm = np.eye(cell.nao_nr())
        vk = mydf.get_jk(dm, kpts=kpts, with_j=False)[1]
        vk_ref = fftdf.get_jk(dm, kpts=kpts, with_j=False)[1]
        self.assertTrue(np.allclose(vk, vk_ref, rtol=1e-6, atol=1e-8))

        mf = scf.KRHF(cell, kpts, exxdiv=None)
        mf.with_df = mydf
        mf.conv_tol = 1e-10
        mf_ref = scf.KRHF(cell, kpts, exxdiv=None)
        mf_ref.with_df = fftdf
        mf_ref.conv_tol = 1e-10
        self.assertAlmostEqual(mf.kernel(), mf_ref.kernel(), 6)

    def test_band_point_keeps_original_mesh(self):
        cell = make_cell(precision=1e-6)
        kpts = cell.make_kpts([1, 1, 1])
        mydf = df.RSGDF_STC(cell, kpts)
        mydf.with_df_j = df.GDF(cell, kpts)
        mydf.build()
        dm = np.eye(cell.nao_nr())
        band = np.asarray([[.07, .03, 0.]])
        vk = mydf.get_jk(dm, kpts=kpts, kpts_band=band, with_j=False)[1]
        self.assertEqual(vk.shape, (1, 1, 1))
        self.assertTrue(np.array_equal(mydf._stc_kmesh, [1, 1, 1]))
        self.assertTrue(mydf._ws_exx['vq_cache'])

    def test_all_electron_scf_smoke(self):
        cell = gto.M(
            a=np.eye(3) * 5., atom='He 0 0 0', basis='sto-3g',
            precision=1e-7, verbose=0)
        kpts = cell.make_kpts([1, 1, 1])
        mf = rsdf_stc.density_fit(scf.KRHF(cell, kpts), eta=3.)
        mf.max_cycle = 2
        self.assertTrue(np.isfinite(mf.kernel()))

        # The single-k-point SCF driver must bypass its usual incore AO-ERI
        # shortcut because sTC uses separate J and K builders.
        mf = rsdf_stc.density_fit(scf.RHF(cell), eta=3.)
        vj, vk = mf.get_jk(dm=np.eye(cell.nao_nr()))
        self.assertEqual(vj.shape, (cell.nao_nr(), cell.nao_nr()))
        self.assertEqual(vk.shape, (cell.nao_nr(), cell.nao_nr()))

    def test_failures_and_compatibility_aliases(self):
        cell = make_cell()
        kpts = cell.make_kpts([1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'requires rc_type'):
            df.FFTDF_STC(cell, kpts, exxdiv='vcut_sph', rc_type='ws')
        mydf = df.RSGDF_STC(cell, kpts)
        mydf.omega_dot_Rc = 3
        mydf.Rc_type = 'ws'
        self.assertEqual(mydf.eta, 3)
        self.assertEqual(mydf.rc_type, 'ws')
        mydf.direct = True
        with self.assertRaisesRegex(NotImplementedError, 'stored CDERIs'):
            mydf.build()
        mydf.direct = False
        mydf.semidirect = True
        with self.assertRaisesRegex(NotImplementedError, 'semidirect'):
            mydf.build()
        mydf.semidirect = False
        with self.assertRaisesRegex(NotImplementedError, 'range-separated'):
            mydf.get_jk(np.eye(cell.nao_nr()), kpts=kpts, omega=.2)

        incomplete = cell.make_kpts([2, 1, 1])[:1]
        # One point alone is a valid Gamma-equivalent 1x1x1 mesh; use two
        # points from a 3-point mesh so the inferred mesh is incomplete.
        incomplete = cell.make_kpts([3, 1, 1])[:2]
        with self.assertRaisesRegex(RuntimeError, 'complete regular mesh'):
            df.FFTDF_STC(cell, incomplete).build()

        lowdim = make_cell()
        lowdim.dimension = 2
        with self.assertRaises(NotImplementedError):
            df.FFTDF_STC(lowdim).build()


if __name__ == '__main__':
    unittest.main()
