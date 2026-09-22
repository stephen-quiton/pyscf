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

'''FFT reference implementation of smoothed truncated Coulomb exchange.'''

from types import SimpleNamespace

import numpy as np

from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.lo.base import get_kmesh
from pyscf.pbc.tools.pbc import _Gv_wrap_around


def _copy_ws_exx(ws_exx):
    if ws_exx is None:
        return None
    out = ws_exx.copy()
    out['vq_cache'] = {key: value.copy()
                       for key, value in ws_exx['vq_cache'].items()}
    return out


def _regular_kmesh(cell, kpts):
    kpts = np.reshape(kpts, (-1, 3))
    try:
        kmesh = np.asarray(get_kmesh(cell, kpts), dtype=int)
    except RuntimeError as err:
        raise RuntimeError(
            'Input k-points do not form a complete regular mesh') from err
    scaled_kpts = cell.get_scaled_kpts(kpts - kpts[0])
    indices = np.rint(scaled_kpts * kmesh).astype(int) % kmesh
    if (len(kpts) != np.prod(kmesh) or
            len(np.unique(indices, axis=0)) != len(kpts)):
        raise RuntimeError('Input k-points do not form a complete regular mesh')
    return kmesh


def _validate_cutoff(exxdiv, rc_type):
    exxdiv = str(exxdiv).lower()
    rc_type = str(rc_type).lower()
    pairs = {'vcut_ws': 'ws', 'vcut_sph': 'sph'}
    if exxdiv not in pairs:
        raise ValueError("exxdiv must be 'vcut_ws' or 'vcut_sph'")
    if rc_type not in ('ws', 'sph'):
        raise ValueError("rc_type must be 'ws' or 'sph'")
    if pairs[exxdiv] != rc_type:
        raise ValueError(f'exxdiv={exxdiv!r} requires rc_type={pairs[exxdiv]!r}')
    return exxdiv, rc_type


def _prepare_stc(mydf):
    cell = mydf.cell
    if cell.dimension != 3:
        raise NotImplementedError(
            'smoothed truncated Coulomb exchange is only available for 3D cells')
    exxdiv, rc_type = _validate_cutoff(mydf.exxdiv, mydf.rc_type)
    kpts = np.reshape(mydf._scf_kpts, (-1, 3))
    kmesh = _regular_kmesh(cell, kpts)
    if rc_type == 'ws':
        rin = tools.get_ws_inradius(cell.lattice_vectors(), kmesh)
        if getattr(mydf, '_ws_exx', None) is None:
            mydf._ws_exx = tools.precompute_exx(cell, kpts)
    else:
        rin = (3 * len(kpts) * cell.vol / (4 * np.pi)) ** (1. / 3)
    mydf._stc_kmesh = kmesh
    mydf._stc_rin = rin
    mydf.omega_stc = float(mydf.eta) / rin
    return exxdiv, mydf.omega_stc


def _truncated_coulG(mydf, kpt, mesh):
    exxdiv, omega = _prepare_stc(mydf)
    cell = mydf.cell
    Gv = cell.get_Gv(mesh)
    context = SimpleNamespace(kpts=mydf._scf_kpts,
                              _ws_exx=getattr(mydf, '_ws_exx', None))
    coulG = tools.get_coulG(cell, kpt, exxdiv, context, mesh, Gv)
    if exxdiv == 'vcut_ws':
        mydf._ws_exx = context._ws_exx
    if abs(kpt).sum() > 1e-9:
        kG = _Gv_wrap_around(cell, Gv, kpt, mesh)
    else:
        kG = Gv
    absG2 = np.einsum('gi,gi->g', kG, kG)
    return coulG, absG2, omega


def get_coulG(mydf, kpt=np.zeros(3), mesh=None, long_range=False):
    '''Return the unweighted sTC kernel.

    If ``long_range`` is true, return the damped truncated term used by the
    RSDF reciprocal-space builder.  Otherwise return the complete sTC kernel.
    '''
    if mesh is None:
        mesh = mydf.mesh
    coulG, absG2, omega = _truncated_coulG(mydf, np.asarray(kpt), mesh)
    damping = np.exp(-absG2 * .25 / omega**2)
    g0 = absG2 == 0
    v0 = coulG[g0].copy()
    coulG *= damping
    if long_range:
        # The finite SR limit is retained here because the RSDF builder
        # removes the same charge term from its real-space erfc integrals.
        coulG[g0] = v0 + np.pi / omega**2
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            coulG += 4 * np.pi / absG2 * (1 - damping)
        coulG[g0] = v0 + np.pi / omega**2
    return coulG


def _post_hf_error(*args, **kwargs):
    raise NotImplementedError(
        'sTC density fitting is only available for HF exchange; use a separate '
        'ordinary GDF/RSDF object for post-HF integrals')


class FFTDF_STC(FFTDF):
    '''FFT numerical reference for smoothed truncated Coulomb exchange.'''

    supports_post_hf = False
    supports_ao_eri = False
    _keys = {'eta', 'exxdiv', 'rc_type', 'omega_stc', '_scf_kpts'}

    def __init__(self, cell, kpts=np.zeros((1, 3)), eta=4.0,
                 exxdiv='vcut_ws', rc_type='ws'):
        if cell.dimension != 3:
            raise NotImplementedError(
                'smoothed truncated Coulomb exchange is only available for 3D cells')
        super().__init__(cell, kpts)
        self._scf_kpts = np.array(self.kpts, copy=True).reshape(-1, 3)
        self.eta = float(eta)
        self.exxdiv, self.rc_type = _validate_cutoff(exxdiv, rc_type)
        self.omega_stc = None
        self._stc_rin = None
        self._stc_kmesh = None
        self._ws_exx = None

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

    def build(self):
        _prepare_stc(self)
        return super().build()

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        _prepare_stc(self)
        log = logger.new_logger(self, verbose)
        log.info('sTC cutoff = %s, rc_type = %s', self.exxdiv, self.rc_type)
        log.info('sTC eta = %.15g, R_in = %.15g, omega = %.15g',
                 self.eta, self._stc_rin, self.omega_stc)
        log.info('sTC SCF kmesh = %s, FFT mesh = %s', self._stc_kmesh, self.mesh)
        return self

    def _get_exchange_coulG(self, kpt, mesh):
        return get_coulG(self, kpt, mesh)

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega not in (None, 0):
            raise NotImplementedError(
                'sTC does not support range-separated hybrid omega requests')
        return super().get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                              omega=None, exxdiv=None)

    def get_k_e1(self, *args, **kwargs):
        raise NotImplementedError('analytical derivatives are not implemented for sTC')

    get_jk_e1 = get_k_e1

    get_eri = get_ao_eri = _post_hf_error
    ao2mo = get_mo_eri = _post_hf_error
    ao2mo_7d = _post_hf_error
    loop = _post_hf_error

    def reset(self, cell=None):
        super().reset(cell)
        if cell is not None:
            self._scf_kpts = np.array(self.kpts, copy=True).reshape(-1, 3)
        self._ws_exx = None
        self._stc_rin = self._stc_kmesh = self.omega_stc = None
        return self

    def copy(self):
        out = super().copy()
        out._scf_kpts = self._scf_kpts.copy()
        out._ws_exx = _copy_ws_exx(self._ws_exx)
        return out
