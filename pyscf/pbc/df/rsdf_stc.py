#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
#
# Authors: Hong-Zhou Ye <hzyechem@gmail.com>
#          Gengzhi Yang <genzyang17@gmail.com>

'''Stored RSDF implementation of smoothed truncated Coulomb exchange.'''

import h5py
import numpy as np

from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import rsdf
from pyscf.pbc.df.fft_stc import (_copy_ws_exx, _prepare_stc,
                                  _validate_cutoff, get_coulG,
                                  _post_hf_error)
from pyscf.pbc.lib.kpts_helper import unique


def _scf_kpts(mf):
    from pyscf.pbc.scf.khf import KSCF
    if isinstance(mf, KSCF):
        kpts = mf.kpts
    else:
        kpts = np.reshape(mf.kpt, (1, 3))
    return np.asarray(getattr(kpts, 'kpts', kpts)).reshape(-1, 3)


def _new_j_df(mf, kpts, auxbasis=None, mesh=None):
    with_df_j = df.GDF(mf.cell, kpts)
    with_df_j.max_memory = mf.max_memory
    with_df_j.stdout = mf.stdout
    with_df_j.verbose = mf.verbose
    with_df_j.auxbasis = auxbasis
    if mesh is not None:
        with_df_j.mesh = mesh
    return with_df_j


def density_fit(mf, auxbasis=None, mesh=None, with_df=None, with_df_j=None,
                exxdiv='vcut_ws', eta=4.0, rc_type='ws',
                omega_dot_Rc=None, Rc_type=None):
    '''Attach stored sTC exchange and an ordinary-GDF Hartree builder to SCF.

    ``omega_dot_Rc`` and ``Rc_type`` are compatibility aliases for ``eta`` and
    ``rc_type``.  The returned SCF object uses no additional Ewald correction;
    the complete finite sTC kernel is already represented by the exchange
    CDERIs.
    '''
    if omega_dot_Rc is not None:
        eta = omega_dot_Rc
    if Rc_type is not None:
        rc_type = Rc_type
    exxdiv, rc_type = _validate_cutoff(exxdiv, rc_type)
    kpts = _scf_kpts(mf)

    if with_df is None:
        with_df = RSGDF_STC(mf.cell, kpts, eta=eta, exxdiv=exxdiv,
                            rc_type=rc_type)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh
    elif not isinstance(with_df, RSGDF_STC):
        raise TypeError('with_df must be an RSGDF_STC object')
    else:
        with_df.eta = float(eta)
        with_df.exxdiv = exxdiv
        with_df.rc_type = rc_type

    if with_df_j is None:
        with_df_j = _new_j_df(mf, kpts, auxbasis, mesh)
    elif not isinstance(with_df_j, df.GDF):
        raise TypeError('with_df_j must be an ordinary GDF object')
    if getattr(with_df_j, 'supports_post_hf', True) is False:
        raise TypeError('with_df_j must not use the sTC interaction')
    with_df.with_df_j = with_df_j

    out = mf.copy().reset()
    out.with_df = with_df
    out.exxdiv = None
    out._eri = None
    return out


class RSGDF_STC(rsdf.RSGDF):
    '''Stored range-separated DF factors for sTC HF exchange only.'''

    supports_post_hf = False
    supports_ao_eri = False
    _keys = {'eta', 'exxdiv', 'rc_type', 'omega_stc', '_scf_kpts',
             'with_df_j'}

    def __init__(self, cell, kpts=np.zeros((1, 3)), eta=4.0,
                 exxdiv='vcut_ws', rc_type='ws'):
        super().__init__(cell, kpts)
        self._scf_kpts = np.array(self.kpts, copy=True).reshape(-1, 3)
        self.eta = float(eta)
        self.exxdiv, self.rc_type = _validate_cutoff(exxdiv, rc_type)
        self.omega_stc = None
        self._stc_rin = None
        self._stc_kmesh = None
        self._ws_exx = None
        self.with_df_j = None

    @property
    def omega_dot_Rc(self):
        return self.eta

    @omega_dot_Rc.setter
    def omega_dot_Rc(self, value):
        self.eta = float(value)

    @property
    def Rc_type(self):
        return self.rc_type

    @Rc_type.setter
    def Rc_type(self, value):
        self.rc_type = str(value).lower()

    def _check_supported(self):
        if self.direct or self.semidirect:
            raise NotImplementedError(
                'sTC requires stored CDERIs; direct and semidirect RSDF are unsupported')

    def _rs_build(self):
        self._check_supported()
        _prepare_stc(self)
        self.omega = self.omega_j2c = self.omega_stc
        super()._rs_build()

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('sTC cutoff = %s, rc_type = %s', self.exxdiv, self.rc_type)
        log.info('sTC eta = %.15g, R_in = %.15g, omega = %.15g',
                 self.eta, self._stc_rin, self.omega_stc)
        log.info('sTC SCF kmesh = %s, compact mesh = %s, j2c mesh = %s',
                 self._stc_kmesh, self.mesh_compact, self.mesh_j2c)
        return self

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None,
                  cderi_file=None):
        if cell is None:
            cell = self.cell
        if auxcell is None:
            auxcell = self.auxcell
        if cderi_file is None:
            cderi_file = self._cderi_to_save
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]
        dfbuilder = _RSGDFBuilder_STC(cell, auxcell, kpts_union)
        dfbuilder.__dict__.update(self.__dict__)
        # Integral bookkeeping needs the union, while the cutoff is defined by
        # the original complete regular SCF mesh retained in _scf_kpts.
        dfbuilder.kpts = kpts_union
        j_only = self._j_only or len(kpts_union) == 1
        # The legacy RSDF writer only removes this dataset when a separate
        # ``kpts`` dataset is present.  Stored band-point rebuilds do not have
        # one, so remove the stale index explicitly before replacing CDERIs.
        cderi_name = (cderi_file if isinstance(cderi_file, str)
                       else cderi_file.name)
        if h5py.is_hdf5(cderi_name):
            with h5py.File(cderi_name, 'a') as feri:
                if 'j3c-kptij' in feri:
                    del feri['j3c-kptij']
        dfbuilder.make_j3c(cderi_file, j_only=j_only,
                           dataname=self._dataname, kptij_lst=kptij_lst)
        self._ws_exx = dfbuilder._ws_exx

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        self._check_supported()
        if omega not in (None, 0):
            raise NotImplementedError(
                'sTC does not support range-separated hybrid omega requests')
        from pyscf.pbc.df.aft import _check_kpts
        kpts = _check_kpts(self, kpts)[0]
        vj = vk = None
        if with_k:
            # No Ewald correction: the sTC CDERIs contain the complete kernel.
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, None)
        if with_j:
            if self.with_df_j is None:
                raise RuntimeError(
                    'RSGDF_STC requires a companion ordinary GDF object for J')
            vj = df_jk.get_j_kpts(self.with_df_j, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = _post_hf_error
    ao2mo = get_mo_eri = _post_hf_error
    ao2mo_7d = _post_hf_error
    loop = update_mp = _post_hf_error

    def reset(self, cell=None):
        super().reset(cell)
        if cell is not None:
            self._scf_kpts = np.array(self.kpts, copy=True).reshape(-1, 3)
        self._ws_exx = None
        self._stc_rin = self._stc_kmesh = self.omega_stc = None
        if self.with_df_j is not None:
            self.with_df_j.reset(cell)
        return self

    def copy(self):
        out = super().copy()
        out._scf_kpts = self._scf_kpts.copy()
        out._ws_exx = _copy_ws_exx(self._ws_exx)
        if self.with_df_j is not None:
            out.with_df_j = self.with_df_j.copy()
        return out


class _RSGDFBuilder_STC(rsdf._RSGDFBuilder):
    def weighted_coulG(self, kpt=np.zeros(3), exx=None, mesh=None, omega=None):
        if mesh is None:
            mesh = self.mesh
        if omega is None:
            omega = self.omega
        # get_coulG derives omega from eta/R_in.  The explicit builder omega
        # must agree for both the j2c metric and j3c factors.
        coulG = get_coulG(self, kpt, mesh, long_range=True)
        if not np.isclose(omega, self.omega_stc):
            raise RuntimeError('inconsistent sTC range-separation parameter')
        return coulG * self.cell.get_Gv_weights(mesh)[2]


RSDF_STC = RSGDF_STC
