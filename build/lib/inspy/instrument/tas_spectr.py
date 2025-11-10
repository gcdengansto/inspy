# -*- coding: utf-8 -*-
r"""Define an instrument for resolution calculations

"""
import datetime as dt
import numpy as np
from scipy.linalg import block_diag as blkdiag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
######from skimage import measure
#from tvtk.api import tvtk
#from mayavi import mlab
import plotly.offline as plyplt
import plotly.subplots as plysub 
import plotly.graph_objects as go

from ..crystal import Sample
from ..energy  import Energy
from .exceptions import ScatteringTriangleError
from .mono import Mono
from .ana  import Ana
from .tools import get_tau, _cleanargs, _modvec, _scalar, _star, _voigt, get_bragg_widths


class SimpleComp:
    def __init__(self, width, height):
        self.width =width
        self.height=height
      
        

class TripleAxisSpectr:
    u"""An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    efixed : float, optional
        Fixed energy, either ei or ef, depending on the instrument
        configuration. Default: 14.7

    sample : obj, optional
        Sample lattice constants, parameters, mosaic, and orientation
        (reciprocal-space orienting vectors). Default: A crystal with
        a,b,c = 6,7,8 and alpha,beta,gamma = 90,90,90 and orientation
        vectors u=[1 0 0] and v=[0 1 0].

    hcol : list(4)
        Horizontal Soller collimations in minutes of arc starting from the
        neutron guide. Default: [40 40 40 40]

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the
        neutron guide. Default: [120 120 120 120]

    mono_tau : str or float, optional
        The monochromator reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common monochromator types.
        Default: 'PG(002)'

    mono_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    ana_tau : str or float, optional
        The analyzer reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common analyzer types.
        Default: 'PG(002)'

    ana_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    Attributes
    ----------


    """
    def __init__(self, efixed=14.7, mono=None, sample=None, ana=None, hcol=None, vcol=None, 
                  guide=None, monitor=None, detector=None, **kwargs):

        if 'instr_type' not in kwargs:
            kwargs['instr_type'] = 'tas'
          
        if sample is None:
            sample = Sample(6, 7, 8, 90, 90, 90)
            sample.u = [1, 0, 0]
            sample.v = [0, 1, 0]
        
        if mono is None:
            mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1)
            
        if ana is None:
            ana = Ana(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1, thickness =0.2, horifoc=-1)
        
        if guide is None:
            guide=SimpleComp(5, 18)
                    
        if monitor is None:
            monitor=SimpleComp(5, 18)

        if detector is None:
            detector=SimpleComp(2.5, 15)
            
        if hcol is None:
            hcol = [40, 40, 40, 40]

        if vcol is None:
            vcol = [120, 120, 120, 120]
            
        self.instr_type = kwargs['instr_type']
        self.mono       = mono
        self.ana        = ana
        self.hcol       = np.array(hcol)
        self.vcol       = np.array(vcol)
        self.efixed     = efixed
        self.sample     = sample
        self.orient1    = np.array(sample.u)
        self.orient2    = np.array(sample.v)
        self.guide      = guide
        self.monitor    = monitor
        self.detector   = detector
        self.method     =  1
        self.infin      = -1
        self.moncor     =  1
        self.description_string="No description yet"

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return "Instrument('tas', engine='inspy', efixed={0})".format(self.efixed)

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                print(value, right_parent_val)
                return False
        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    @property
    def mono(self):
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value

    @property
    def ana(self):
        return self._ana

    @ana.setter
    def ana(self, value):
        self._ana = value

    @property
    def method(self):
        """Selects the computation method.
        If method=0 or left undefined, a Cooper-Nathans calculation is performed. 
        For a Popovici calculation set method=1.
        """
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def moncor(self):
        """Selects the type of normalization used to calculate ``R0``
        If ``moncor=1`` or left undefined, ``R0`` is calculated in
        normalization to monitor counts (Section II C 2). 1/k\ :sub:`i` monitor
        efficiency correction is included automatically. To normalize ``R0`` to
        source flux (Section II C 1), use ``moncor=0``.
        """
        return self._moncor

    @moncor.setter
    def moncor(self, value):
        self._moncor = value

    @property
    def hcol(self):
        r""" The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. In case of a
        horizontally-focusing analyzer ``hcol[2]`` is the angular size of the
        analyzer, as seen from the sample position. If the beam divergence is
        limited by a neutron guide, the corresponding element of :attr:`hcol`
        is the negative of the guide’s *m*-value. For example, for a 58-Ni
        guide ( *m* = 1.2 ) before the monochromator, ``hcol[0]`` should be
        -1.2.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        """The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. If the beam
        divergence is limited by a neutron guide, the corresponding element of
        :attr:`vcol` is the negative of the guide’s *m*-value. For example, for
        a 58-Ni guide ( *m* = 1.2 ) before the monochromator, ``vcol[0]``
        should be -1.2.
        """
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        """distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and
        monochromator and monitor, respectively. The 5th element is only needed
        if moncor=1
        """
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = value

    @property
    def efixed(self):
        """the fixed incident or final neutron energy, in meV.
        """
        return self._efixed

    @efixed.setter
    def efixed(self, value):
        self._efixed = value

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def orient1(self):
        """Miller indexes of the first reciprocal-space orienting vector for
        the S coordinate system, as explained in Section II G.
        """
        return self._sample.u

    @orient1.setter
    def orient1(self, value):
        self._sample.u = np.array(value)

    @property
    def orient2(self):
        """Miller indexes of the second reciprocal-space orienting vector
        for the S coordinate system, as explained in Section II G.
        """
        return self._sample.v

    @orient2.setter
    def orient2(self, value):
        self._sample.v = np.array(value)

    @property
    def infin(self):
        """a flag set to -1 or left unassigned if the final energy is fixed, or
        set to +1 in a fixed-incident setup.
        """
        return self._infin

    @infin.setter
    def infin(self, value):
        self._infin = value

    @property
    def guide(self):
        r"""A structure that describes the source
        """
        return self._guide

    @guide.setter
    def guide(self, value):
        self._guide = value

    @property
    def detector(self):
        """A structure that describes the detector
        """
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def monitor(self):
        """A structure that describes the monitor
        """
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def Smooth(self):
        u"""Defines the smoothing parameters as explained in Section II H. Leave this
        field unassigned if you don’t want this correction done.

        * ``Smooth.E`` is the smoothing FWHM in energy (meV). A small number
          means “no smoothing along this direction”.

        * ``Smooth.X`` is the smoothing FWHM along the first orienting vector
          (x0 axis) in Å\ :sup:`-1`.

        * ``Smooth.Y`` is the smoothing FWHM along the y axis in Å\ :sup:`-1`.

        * ``Smooth.Z`` is the smoothing FWHM along the vertical direction in
          Å\ :sup:`-1`.

        """
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value
    


    def get_lattice(self):
        r"""Extracts lattice parameters from EXP and returns the direct and reciprocal lattice 
        parameters in the form used by _scalar.m, _star.m,etc.
        Returns
        -------
        [lattice, rlattice] : [class, class] Returns the direct and reciprocal lattice sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        lattice = Sample(self.sample.a,
                         self.sample.b,
                         self.sample.c,
                         np.deg2rad(self.sample.alpha),
                         np.deg2rad(self.sample.beta),
                         np.deg2rad(self.sample.gamma))
        rlattice = _star(lattice)[-1]

        return [lattice, rlattice]

    def _StandardSystem(self):
        r"""Returns rotation matrices to calculate resolution in the sample view
        instead of the instrument view

        Attributes
        ----------
        EXP : class
            Instrument class

        Returns
        -------
        [x, y, z, lattice, rlattice] : [array, array, array, class, class]
            Returns the rotation matrices and real and reciprocal lattice
            sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        [lattice, rlattice] = self.get_lattice()

        orient1 = self.orient1
        orient2 = self.orient2

        modx    = _modvec(orient1, rlattice)

        x       = orient1 / modx

        proj    = _scalar(orient2, x, rlattice)

        y       = orient2 - x * proj

        mody    = _modvec(y, rlattice)

        if len(np.where(mody <= 0)[0]) > 0:
            raise ScatteringTriangleError('Orienting vectors are colinear')

        y /= mody

        z = np.array([ x[1] * y[2] - y[1] * x[2],
                       x[2] * y[0] - y[2] * x[0],
                      -x[1] * y[0] + y[1] * x[0]], dtype=np.float64)

        proj   = _scalar(z, x, rlattice)

        z     -= x * proj

        proj   = _scalar(z, y, rlattice)

        z     -= y * proj

        modz   = _modvec(z, rlattice)

        z /= modz

        return [x, y, z, lattice, rlattice]

    
    def get_angles_and_Q(self, hkle):
        r"""Returns the Triple Axis Spectrometer angles and Q-vector given
        position in reciprocal space

        Parameters
        ----------
        hkle : list
            Array of the scattering vector and energy transfer at which the
            calculation should be performed

        Returns
        -------
        [A, Q] : list
            The angles A (A1 -- A5 in a list of floats) and Q (ndarray)

        """
        # compute all TAS angles (in plane)

        h, k, l, w = hkle
        # compute angles
        try:
            fx = 2 * int(self.infin == -1) + int(self.infin == 1)
        except AttributeError:
            fx = 2

        kfix = Energy(energy=self.efixed).wavevector
        f = 0.4826                                  # f converts from energy units into k^2, f=0.4826 for meV
        ki = np.sqrt(kfix ** 2 + (fx - 1) * f * w)  # kinematical equations.
        kf = np.sqrt(kfix ** 2 - (2 - fx) * f * w)

        # compute the transversal Q component, and A3 (sample rotation)
        # from McStas templateTAS.instr and TAS MAD ILL
        a     = np.array([self.sample.a, self.sample.b, self.sample.c]) / (2 * np.pi)
        alpha = np.deg2rad([self.sample.alpha, self.sample.beta, self.sample.gamma])
        cosa  = np.cos(alpha)
        sina  = np.sin(alpha)
        cc    = np.sum(cosa * cosa)
        cc    = 1 + 2 * np.product(cosa) - cc
        cc    = np.sqrt(cc)
        b     = sina / (a * cc)
        c1    = np.roll(cosa[np.newaxis].T, -1)
        c2    = np.roll(c1, -1)
        s1    = np.roll(sina[np.newaxis].T, -1)
        s2    = np.roll(s1, -1)
        cosb  = (c1 * c2 - cosa[np.newaxis].T) / (s1 * s2)
        sinb  = np.sqrt(1 - cosb * cosb)

        bb    = np.array([[b[0],                        0,              0],
                          [b[1] * cosb[2], b[1] * sinb[2],              0],
                          [b[2] * cosb[1], -b[2] * sinb[1] * cosa[0], 1 / a[2]]], dtype=np.float64)
        bb    = bb.T

        aspv  = np.hstack((self.orient1[np.newaxis].T, self.orient2[np.newaxis].T))

        vv    = np.zeros((3, 3))
        vv[0:2, :] = np.transpose(np.dot(bb, aspv))
        
        
        for m in range(2, 0, -1):
            vt = np.roll(np.roll(vv, -1, axis=0), -1, axis=1) * np.roll(np.roll(vv, -2, axis=0), -2, axis=1) - np.roll(
                np.roll(vv, -1, axis=0), -2, axis=1) * np.roll(np.roll(vv, -2, axis=0), -1, axis=1)
            vv[m, :] = vt[m, :]

        c      = np.sqrt(np.sum(vv * vv, axis=0))

        vv     = vv / np.tile(c, (3, 1))
        s      = vv.T * bb

        qt     = np.squeeze(np.dot(np.array([h, k, l]).T, s.T))

        qs     = np.sum(qt ** 2)
        Q      = np.sqrt(qs)

        sm     = self.mono.dir
        ss     = self.sample.dir
        sa     = self.ana.dir
        dm     = 2 * np.pi / get_tau(self.mono.tau)
        da     = 2 * np.pi / get_tau(self.ana.tau)
        thetaa = sa * np.arcsin(np.pi / (da * kf))  # theta angles for analyser
        thetam = sm * np.arcsin(np.pi / (dm * ki))  # and monochromator.
        thetas = ss * 0.5 * np.arccos((ki ** 2 + kf ** 2 - Q ** 2) / (2 * ki * kf))  # scattering angle from sample.

        A3     = -np.arctan2(qt[1], qt[0]) - np.arccos(
                     (np.dot(kf, kf) - np.dot(Q, Q) - np.dot(ki, ki)) / (-2 * np.dot(Q, ki)))
        A3     = ss * A3

        A1     = thetam
        A2     = 2 * A1
        A4     = 2 * thetas
        A5     = thetaa
        A6     = 2 * A5
        temp_a = np.float32([A1, A2, A3, A4, A5, A6])

        A      = np.squeeze(np.rad2deg(temp_a))

        return [A, Q]
    
    
    
    
    
    
    
    def CalcResMatQ(self, Q, W):
        r"""For a momentum transfer Q and energy transfers W, given experimental
        conditions specified in EXP, calculates the Cooper-Nathans or Popovici
        resolution matrix RM and resolution prefactor R0 in the Q coordinate
        system (defined by the scattering vector and the scattering plane).

        Parameters
        ----------
        Q : ndarray or list of ndarray
            The Q vectors in reciprocal space at which resolution should be
            calculated, in inverse angstroms

        W : float or list of floats
            The energy transfers at which resolution should be calculated in meV

        Returns
        -------
        [R0, RM] : list(float, ndarray)
            Resolution pre-factor (R0) and resolution matrix (RM) at the given
            reciprocal lattice vectors and energy transfers

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        CONVERT1 = np.pi / 60. / 180. / np.sqrt(8 * np.log(2))
        CONVERT2 = 2.072

        [length, Q, W] = _cleanargs(Q, W)

        RM       = np.zeros((length, 4, 4), dtype=np.float64)
        R0       = np.zeros(length, dtype=np.float64)
        RM_      = np.zeros((4, 4), dtype=np.float64)

        # The method to use
        method   = 0
        if hasattr(self, 'method'):
            method = self.method

        # Assign default values and decode parameters
        moncor   = 1
        if hasattr(self, 'moncor'):
            moncor = self.moncor

        alpha    = np.array(self.hcol) * CONVERT1
        beta     = np.array(self.vcol) * CONVERT1
        mono     = self.mono
        etam     = np.array(mono.mosaic) * CONVERT1
        etamv    = np.copy(etam)
        if hasattr(mono, 'vmosaic') and (method == 1 or method == 'Popovici'):
            etamv = np.array(mono.vmosaic) * CONVERT1

        ana      = self.ana
        etaa     = np.array(ana.mosaic) * CONVERT1
        etaav    = np.copy(etaa)
        if hasattr(ana, 'vmosaic'):
            etaav = np.array(ana.vmosaic) * CONVERT1

        sample   = self.sample
        infin    = -1
        if hasattr(self, 'infin'):
            infin = self.infin

        efixed = self.efixed

        monitorw = 1.
        monitorh = 1.
        beamw    = 1.
        beamh    = 1.
        monow    = 1.
        monoh    = 1.
        monod    = 1.
        anaw     = 1.
        anah     = 1.
        anad     = 1.
        detectorw = 1.
        detectorh = 1.
        sshapes   = np.repeat(np.eye(3, dtype=np.float64)[np.newaxis].reshape((1, 3, 3)), length, axis=0)
        sshape_factor = 12.
        L0     = 1.
        L1     = 1.
        L1mon  = 1.
        L2     = 1.
        L3     = 1.
        monorv = 1.e6
        monorh = 1.e6
        anarv  = 1.e6
        anarh  = 1.e6

        if hasattr(self, 'guide'):
            beam = self.guide
            if hasattr(beam, 'width'):
                beamw = beam.width ** 2 / 12.
            if hasattr(beam, 'height'):
                beamh = beam.height ** 2 / 12.
        bshape = np.diag([beamw, beamh])

        if hasattr(self, 'monitor'):
            monitor = self.monitor
            if hasattr(monitor, 'width'):
                monitorw = monitor.width ** 2 / 12.
            monitorh = monitorw
            if hasattr(monitor, 'height'):
                monitorh = monitor.height ** 2 / 12.
        monitorshape = np.diag([monitorw, monitorh])

        if hasattr(self, 'detector'):
            detector = self.detector
            if hasattr(detector, 'width'):
                detectorw = detector.width ** 2 / 12.
            if hasattr(detector, 'height'):
                detectorh = detector.height ** 2 / 12.
        dshape = np.diag([detectorw, detectorh])

        if hasattr(mono, 'width'):
            monow = mono.width ** 2 / 12.
        if hasattr(mono, 'height'):
            monoh = mono.height ** 2 / 12.
        if hasattr(mono, 'depth'):
            monod = mono.depth ** 2 / 12.
        mshape = np.diag([monod, monow, monoh])

        if hasattr(ana, 'width'):
            anaw = ana.width ** 2 / 12.
        if hasattr(ana, 'height'):
            anah = ana.height ** 2 / 12.
        if hasattr(ana, 'depth'):
            anad = ana.depth ** 2 / 12.
        ashape = np.diag([anad, anaw, anah])

        if hasattr(sample, 'shape_type'):
            if sample.shape_type   == 'cylindrical':
                sshape_factor = 16.
            elif sample.shape_type == 'rectangular':
                sshape_factor = 12.
        if hasattr(sample, 'width') and hasattr(sample, 'depth') and hasattr(sample, 'height'):
            _sshape = np.diag([sample.depth, sample.width, sample.height]).astype(np.float64) ** 2 / sshape_factor
            sshapes = np.repeat(_sshape[np.newaxis].reshape((1, 3, 3)), length, axis=0)
        elif hasattr(sample, 'shape'):
            _sshape = sample.shape.astype(np.float64) / sshape_factor
            if len(_sshape.shape) == 2:
                sshapes = np.repeat(_sshape[np.newaxis].reshape((1, 3, 3)), length, axis=0)
            else:
                sshapes = _sshape

        if hasattr(self, 'arms') and method == 1:
            arms = self.arms
            L0, L1, L2, L3 = arms[:4]
            L1mon = np.copy(L1)
            if len(arms) > 4:
                L1mon = np.copy(arms[4])

        if hasattr(mono, 'rv'):
            monorv = mono.rv

        if hasattr(mono, 'rh'):
            monorh = mono.rh

        if hasattr(ana, 'rv'):
            anarv  = ana.rv

        if hasattr(ana, 'rh'):
            anarh  = ana.rh

        taum = get_tau(mono.tau)
        taua = get_tau(ana.tau)

        horifoc = -1
        if hasattr(ana, 'horifoc'):
            horifoc = ana.horifoc

        if horifoc == 1:
            alpha[2] = alpha[2] * np.sqrt(8. * np.log(2.) / 12.)

        sm = self.mono.dir
        ss = self.sample.dir
        sa = self.ana.dir
        #print("shapes")
        #print(sshapes)
        for ind in range(length):
            sshape = sshapes[ind, :, :]
            # Calculate angles and energies
            w  = W[ind]
            q  = Q[ind]
            ei = efixed
            ef = efixed
            if infin > 0:
                ef = efixed - w
            else:
                ei = efixed + w
            ki = np.sqrt(ei / CONVERT2)
            kf = np.sqrt(ef / CONVERT2)

            thetam  = np.arcsin(taum / (2. * ki)) * sm
            thetaa  = np.arcsin(taua / (2. * kf)) * sa
            s2theta = np.arccos(np.complex128((ki ** 2 + kf ** 2 - q ** 2) / (2. * ki * kf))) * ss
            
            #print("ki:{},kf:{},thetam:{},thetaa:{},s2theta:{},q:{}".format(ki,kf,thetam,thetaa,s2theta,q))  # added by dgc
            
            
            if np.abs(np.imag(s2theta)) > 1e-12:
                raise ScatteringTriangleError(
                    'KI,KF,Q triangle will not close. Change the value of KFIX,FX,QH,QK or QL.')
            else:
                s2theta = np.real(s2theta)

            #correct sign of curvatures
            #monorh = monorh * sm
            #monorv = monorv * sm
            #anarh  = anarh * sa
            #anarv  = anarv * sa

            thetas = s2theta / 2.
            phi = np.arctan2(-kf * np.sin(s2theta), ki - kf * np.cos(s2theta))

            # Calculate beam divergences defined by neutron guides
            alpha[alpha < 0]   = -alpha[alpha < 0] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)
            beta[beta   < 0]   = -beta [beta  < 0] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)

            # Redefine sample geometry
            psi = thetas - phi  # Angle from sample geometry X axis to Q
            rot = np.matrix([[ np.cos(psi),  np.sin(psi), 0],
                             [-np.sin(psi),  np.cos(psi), 0],
                             [           0,            0, 1]], dtype=np.float64)

            # sshape=rot'*sshape*rot
            sshape = np.matrix(rot) * np.matrix(sshape) * np.matrix(rot).H

            # Definition of matrix G
            G = np.matrix(
                np.diag(1. / np.array([alpha[:2], beta[:2], alpha[2:], beta[2:]], dtype=np.float64).flatten() ** 2))

            
            # Definition of matrix F
            F = np.matrix(np.diag(1. / np.array([etam, etamv, etaa, etaav], dtype=np.float64) ** 2))
            
            # Definition of matrix A
            A = np.matrix([[ki / 2. / np.tan(thetam), -ki / 2. / np.tan(thetam), 0, 0, 0, 0, 0, 0],
                           [0, ki, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, ki, 0, 0, 0, 0],
                           [0, 0, 0, 0, kf / 2. / np.tan(thetaa), -kf / 2. / np.tan(thetaa), 0, 0],
                           [0, 0, 0, 0, kf, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, kf, 0]], dtype=np.float64)


            # Definition of matrix C
            C = np.matrix([[0.5, 0.5, 0, 0, 0, 0, 0, 0],
                           [0., 0., 1. / (2. * np.sin(thetam)), -1. / (2. * np.sin(thetam)), 0, 0, 0, 0],
                           [0, 0, 0, 0, 0.5, 0.5, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1. / (2. * np.sin(thetaa)), -1. / (2. * np.sin(thetaa))]],
                          dtype=np.float64)

            # Definition of matrix Bmatrix
            Bmatrix = np.matrix([[np.cos(phi), np.sin(phi), 0, -np.cos(phi - s2theta), -np.sin(phi - s2theta), 0],
                                 [-np.sin(phi), np.cos(phi), 0, np.sin(phi - s2theta), -np.cos(phi - s2theta), 0],
                                 [0, 0, 1, 0, 0, -1],
                                 [2. * CONVERT2 * ki, 0, 0, -2. * CONVERT2 * kf, 0, 0]], dtype=np.float64)
            """
            print("A:")    # by dgc
            print(A)
            print("C:")    # by dgc
            print(C)
            print("B:")    # by dgc
            print(Bmatrix)
            print("bshape:")
            print(bshape)
            print(mshape)
            print("sshape:")
            print(sshape)
            print(ashape)
            print(dshape)
            """
            #print(ki)
            #print(kf)
            #print(thetam)
            #print(thetaa)
            #print(s2theta)
            #print(phi)
            #print(Bmatrix)
            # Definition of matrix S
            Sinv = np.matrix(blkdiag(np.array(bshape, dtype=np.float64), mshape, sshape, ashape, dshape))  # S-1 matrix
            S    = Sinv.I

            # Definition of matrix T
            """
            T = np.zeros((4,13))
            T[1,1]=-1. / (2. * L0)
            T[1,3]=np.cos(thetam) * (1. / L1 - 1. / L0) / 2.
            T[1,4]=np.sin(thetam) * (1. / L0 + 1. / L1 - 2. / (monorh * np.sin(thetam))) / 2.
            T[1,6]=np.sin(thetas) / (2. * L1)
            T[1,7]=np.cos(thetas) / (2. * L1)
            T[2,2]=-1. / (2. * L0 * np.sin(thetam))
            T[2,5]=(1. / L0 + 1. / L1 - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
            T[2,8]=-1. / (2. * L1 * np.sin(thetam))
            T[3,6]=np.sin(thetas) / (2. * L2)
            T[3,7]=-np.cos(thetas) / (2. * L2)
            T[3,9]=np.cos(thetaa) * (1. / L3 - 1. / L2) / 2.
            T[3,9]=np.sin(thetaa) * (1. / L2 + 1. / L3 - 2. / (anarh * np.sin(thetaa))) / 2.
            T[3,12]= 1. / (2. * L3)
            T[4,8]=-1. / (2. * L2 * np.sin(thetaa))
            T[4,11]=(1. / L2 + 1. / L3 - 2. * np.sin(thetaa) / anarv) / (2. * np.sin(thetaa))
            T[4,13]=-1. / (2. * L3 * np.sin(thetaa))
            """
            T = np.matrix([[-1. / (2. * L0), 0, np.cos(thetam) * (1. / L1 - 1. / L0) / 2.,
                            np.sin(thetam) * (1. / L0 + 1. / L1 - 2. / (monorh * np.sin(thetam))) / 2., 0,
                            np.sin(thetas) / (2. * L1), np.cos(thetas) / (2. * L1), 0, 0, 0, 0, 0, 0],
                           [0, -1. / (2. * L0 * np.sin(thetam)), 0, 0,
                            (1. / L0 + 1. / L1 - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam)), 0, 0,
                            -1. / (2. * L1 * np.sin(thetam)), 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, np.sin(thetas) / (2. * L2), -np.cos(thetas) / (2. * L2), 0,
                            np.cos(thetaa) * (1. / L3 - 1. / L2) / 2.,
                            np.sin(thetaa) * (1. / L2 + 1. / L3 - 2. / (anarh * np.sin(thetaa))) / 2., 0,
                            1. / (2. * L3), 0],
                           [0, 0, 0, 0, 0, 0, 0, -1. / (2. * L2 * np.sin(thetaa)), 0, 0,
                            (1. / L2 + 1. / L3 - 2. * np.sin(thetaa) / anarv) / (2. * np.sin(thetaa)), 0,
                            -1. / (2. * L3 * np.sin(thetaa))]], dtype=np.float64)

            #print("T:")    # by dgc
            #print(T)
            # Definition of matrix D
            # Lots of index mistakes in paper for matrix D
            D = np.matrix([[-1. / L0, 0, -np.cos(thetam) / L0, np.sin(thetam) / L0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, np.cos(thetam) / L1, np.sin(thetam) / L1, 0, np.sin(thetas) / L1, np.cos(thetas) / L1,
                            0, 0, 0, 0, 0, 0],
                           [0, -1. / L0, 0, 0, 1. / L0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, -1. / L1, 0, 0, 1. / L1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, np.sin(thetas) / L2, -np.cos(thetas) / L2, 0, -np.cos(thetaa) / L2,
                            np.sin(thetaa) / L2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, np.cos(thetaa) / L3, np.sin(thetaa) / L3, 0, 1. / L3, 0],
                           [0, 0, 0, 0, 0, 0, 0, -1. / L2, 0, 0, 1. / L2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1. / L3, 0, 1. / L3]], dtype=np.float64)
            """
            print("D:")    # by dgc
            print(D)
            print("S:")    # by dgc
            print(S)
            print("F:")    # by dgc
            print(F)
            print("G:")    # by dgc
            print(G)
            """
            # Definition of resolution matrix M
            if method == 1 or method == 'popovici':
                K    = S + T.H * F * T
                H    = np.linalg.inv(D * np.linalg.inv(K) * D.H)
                Ninv = A * np.linalg.inv(H + G) * A.H
                
            else:
                H    = G + C.H * F * C
                Ninv = A * np.linalg.inv(H) * A.H
                # Horizontally focusing analyzer if needed
                if horifoc > 0:
                    Ninv = np.linalg.inv(Ninv)
                    Ninv[3:5, 3:5] = np.matrix([[(np.tan(thetaa) / (etaa * kf)) ** 2, 0],
                                                [0, (1 / (kf * alpha[2])) ** 2]], dtype=np.float64)
                    Ninv = np.linalg.inv(Ninv)

            Minv = Bmatrix * Ninv * Bmatrix.H
            #print(method)
            #print("Minv:")           # It has confirmed that Minv is corrected calculated. for both method=1 and 0
            #print(Minv)              # But RM_ is not correct.

            M   = np.linalg.inv(Minv)
            RM_ = np.copy(M)
            #print("M:")      # by dgc
            #print(M)
            #print("RM_:")    # by dgc
            #print(RM_)

            
            # by dgc
            RM_[0,2]=M[0,3]
            RM_[2,0]=M[3,0]
            RM_[2,2]=M[3,3]
            RM_[2,1]=M[3,1]
            RM_[1,2]=M[1,3]
    
            RM_[0,3]=M[0,2]
            RM_[3,0]=M[2,0]
            RM_[3,3]=M[2,2]
            RM_[3,1]=M[2,1]
            RM_[1,3]=M[1,2]
            
            #print("RM_:")    # by dgc
            #print(RM_)
            # Calculation of prefactor, normalized to source
            Rm  = ki ** 3 / np.tan(thetam)
            Ra  = kf ** 3 / np.tan(thetaa)
            R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * np.sin(thetam) * np.sin(thetaa))

            if method == 1 or method == 'popovici':
                # Popovici
                R0_ = R0_ * np.sqrt(np.linalg.det(F) / np.linalg.det(H + G))
            else:
                # Cooper-Nathans (popovici Eq 5 and 9)
                R0_ = R0_ * np.sqrt(np.linalg.det(F) / np.linalg.det(H))

            # Normalization to flux on monitor
            if moncor == 1:
                g = G[:4, :4]
                f = F[:2, :2]
                c = C[:2, :4]

                t = np.matrix([[-1. / (2. * L0), 0, np.cos(thetam) * (1. / L1mon - 1. / L0) / 2.,
                                np.sin(thetam) * (1. / L0 + 1. / L1mon - 2. / (monorh * np.sin(thetam))) / 2., 0, 0,
                                1. / (2. * L1mon)],
                               [0, -1. / (2. * L0 * np.sin(thetam)), 0, 0,
                                (1. / L0 + 1. / L1mon - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam)), 0, 0]],
                              dtype=np.float64)

                sinv = blkdiag(np.array(bshape, dtype=np.float64), mshape, monitorshape)  # S-1 matrix
                s = np.linalg.inv(sinv)

                d = np.matrix([[-1. / L0, 0, -np.cos(thetam) / L0, np.sin(thetam) / L0, 0, 0, 0],
                               [0, 0, np.cos(thetam) / L1mon, np.sin(thetam) / L1mon, 0, 0, 1. / L1mon],
                               [0, -1. / L0, 0, 0, 1. / L0, 0, 0],
                               [0, 0, 0, 0, -1. / L1mon, 0, 0]], dtype=np.float64)

                if method == 1 or method == 'popovici':
                    # Popovici
                    Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * np.sqrt(
                        np.linalg.det(f) / np.linalg.det(np.linalg.inv(d * np.linalg.inv(s + t.H * f * t) * d.H) + g))
                else:
                    # Cooper-Nathans
                    Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * np.sqrt(
                        np.linalg.det(f) / np.linalg.det(g + c.H * f * c))

                R0_ = R0_ / Rmon
                R0_ = R0_ * ki  # 1/ki monitor efficiency

            #print("R0:_{}".format(R0_))
            
            # Transform prefactor to Chesser-Axe normalization
            R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(RM_))
            # Include kf/ki part of cross section
            R0_ = R0_ * kf / ki
            
            #print("R0_:{}".format(R0_))

            # Take care of sample mosaic if needed
            # [S. A. Werner & R. Pynn, J. Appl. Phys. 42, 4736, (1971), eq 19]
            if hasattr(sample, 'mosaic'):
                etas  = sample.mosaic * CONVERT1
                etasv = np.copy(etas)
                if hasattr(sample, 'vmosaic'):
                    etasv = sample.vmosaic * CONVERT1
                R0_  = R0_ / np.sqrt((1 + (q * etas) ** 2 * RM_[3, 3]) * (1 + (q * etasv) ** 2 * RM_[1, 1]))
                Minv = np.linalg.inv(RM_)
                Minv[1, 1] = Minv[1, 1] + q ** 2 * etas ** 2
                Minv[3, 3] = Minv[3, 3] + q ** 2 * etasv ** 2
                RM_ = np.linalg.inv(Minv)

            # Take care of analyzer reflectivity if needed [I. Zaliznyak, BNL]
            if hasattr(ana, 'thickness') and hasattr(ana, 'Q'):
                KQ  = ana.Q
                KT  = ana.thickness
                toa = (taua / 2.) / np.sqrt(kf ** 2 - (taua / 2.) ** 2)
                smallest = alpha[3]
                if alpha[3] > alpha[2]:
                    smallest = alpha[2]
                Qdsint = KQ * toa
                dth  = (np.arange(1, 201) / 200.) * np.sqrt(2. * np.log(2.)) * smallest
                wdth = np.exp(-dth ** 2 / 2. / etaa ** 2)
                sdth = KT * Qdsint * wdth / etaa / np.sqrt(2. * np.pi)
                rdth = 1. / (1 + 1. / sdth)
                reflec = sum(rdth) / sum(wdth)
                R0_ = R0_ * reflec

            R0[ind] = R0_
            RM[ind] = RM_.copy()
            #print("R0_:{}".format(R0_))

        return [R0, RM]

    def CalcResMatHKL(self, hkle):
        r"""For a scattering vector (H,K,L) and  energy transfers W, given
        experimental conditions specified in EXP, calculates the Cooper-Nathans
        resolution matrix RMS and Cooper-Nathans Resolution prefactor R0 in a
        coordinate system defined by the crystallographic axes of the sample.

        Parameters
        ----------
        hkle : list
            Array of the scattering vector and energy transfer at which the
            calculation should be performed

        Notes
        -----
            Translated from ResLib, originally authored by A. Zheludev, 1999-2007,
            Oak Ridge National Laboratory

        """
        #print(hkle)
        self.HKLE    = hkle
        [H, K, L, W] = hkle

        [length, H, K, L, W]           = _cleanargs(H, K, L, W)
        self.H, self.K, self.L, self.W = H, K, L, W
        #print("length:{}".format(length))

        [x, y, z, sample, rsample] = self._StandardSystem()
        del z, sample

        Q = _modvec([H, K, L], rsample)
        #print("Q:{}".format(Q))
        uq = np.vstack((H / Q, K / Q, L / Q))

        xq = _scalar(x, uq, rsample)
        yq = _scalar(y, uq, rsample)

        tmat = np.array(
            [np.array([[xq[i], yq[i], 0, 0], [-yq[i], xq[i], 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]], dtype=np.float64) for i in range(len(xq))])

        RMS = np.zeros((length, 4, 4), dtype=np.float64)
        #rot = np.zeros((3, 3), dtype=np.float64)
        #print("tmat:shape")
        #print(tmat.shape)
        #print(tmat)
        
        # Sample shape matrix in coordinate system defined by scattering vector
        """
        #dgc change this part into a single sample rotation.
        #however, it is removed, because this is done in CalcResMatQ line 753
        sample = self.sample
        if hasattr(sample, 'shape'):
            #print("sampleshape:{}".format(sample.shape))
            rot[0, 0] = tmat[0, 0, 0]
            rot[1, 0] = tmat[0, 1, 0]
            rot[0, 1] = tmat[0, 0, 1]
            rot[1, 1] = tmat[0, 1, 1]
            rot[2, 2] = tmat[0, 2, 2]
            self.sample.shape = np.matrix(rot) * np.matrix(sample.shape) * np.matrix(rot).H
        """
        
        [R0, RM] = self.CalcResMatQ(Q, W)

        for i in range(length):
            RMS[i] = np.matrix(tmat[i]).H * np.matrix(RM[i]) * np.matrix(tmat[i])

        e = np.identity(4)
        for i in range(length):
            if hasattr(self, 'Smooth'):
                if self.Smooth.X:
                    mul    = np.diag([1 / (self.Smooth.X ** 2 / 8 / np.log(2)),
                                      1 / (self.Smooth.Y ** 2 / 8 / np.log(2)),
                                      1 / (self.Smooth.E ** 2 / 8 / np.log(2)),
                                      1 / (self.Smooth.Z ** 2 / 8 / np.log(2))])
                    R0[i]  = R0[i] / np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(RMS[i]))) * np.sqrt(
                        np.linalg.det(np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[i])))
                    RMS[i] = np.matrix(e) / (
                        np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[i]))

        self.R0, self.RMS, self.RM = [np.squeeze(item) for item in (R0, RMS, RM)]

    

    def ResConv(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r"""Numerically calculate the convolution of a user-defined cross-section function with the resolution function for a
        3-axis neutron scattering experiment.

        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 4D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, $\phi_2$, and $\phi_3$, and 2*ACCURACY[1]+1
            along $\phi_4$ (vertical direction). 'mc': 4D Monte Carlo
            integration. The cross section is sampled in 1000*ACCURACY
            randomly chosen points, uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

        """
        #print('this is resolution_convolution function')
        self.CalcResMatHKL(hkle)
        #print(hkle)
        [R0, RMS] = [np.copy(self.R0), self.RMS.copy()]
        #print(R0)
        #print(RMS)

        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        #print(len(W))
        [xvec, yvec, zvec] = self._StandardSystem()[:3]
        """
        Mxx = RMS[:, 0, 0]
        Mxy = RMS[:, 0, 1]
        Mxw = RMS[:, 0, 2]   #changed by dgc  was  0, 3]
        Myy = RMS[:, 1, 1]
        Myw = RMS[:, 1, 2]   #hanged by dgc was 1, 3]
        Mzz = RMS[:, 3, 3]   #changed by dgc was 2, 2]
        Mww = RMS[:, 2, 2]   #changed by dgc, was 3, 3]
        """
        Mxx = RMS[:, 0, 0]
        Mxy = RMS[:, 0, 1]
        Mxw = RMS[:, 0, 2]   #changed by dgc  was  0, 3]
        Myy = RMS[:, 1, 1]
        Myw = RMS[:, 1, 2]   #hanged    by dgc was 1, 3]
        Mzz = RMS[:, 3, 3]   #changed   by dgc was 2, 2]
        Mww = RMS[:, 2, 2]   #changed by dgc, was 3, 3]

        Mxx -= Mxw ** 2. / Mww
        Mxy -= Mxw * Myw / Mww
        Myy -= Myw ** 2. / Mww
        MMxx = Mxx - Mxy ** 2. / Myy

        detM = MMxx * Myy * Mzz * Mww

        tqz  = 1. / np.sqrt(Mzz)
        tqx  = 1. / np.sqrt(MMxx)
        tqyy = 1. / np.sqrt(Myy)
        tqyx = -Mxy / Myy / np.sqrt(MMxx)
        tqww = 1. / np.sqrt(Mww)
        tqwy = -Myw / Mww / np.sqrt(Myy)
        tqwx = -(Mxw / Mww - Myw / Mww * Mxy / Myy) / np.sqrt(MMxx)

        inte = sqw(H, K, L, W, p)
        #print(inte[0,:])
        [modes, points] = inte.shape
        #print(modes)
        #print(points)

        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number or output arguments in prefactor function')

        if METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = np.array([7, 0])
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            #print(dd1)
            convs = np.zeros((modes, length))
            conv  = np.zeros(length)
            [cw, cx, cy] = np.meshgrid(dd1, dd1, dd1, indexing='ij')   #deng bug? [cw, cx, cy] changed into  [cx, cy, cw]
            tx = np.tan(cx.flatten())  #dgc: i don't know why, use flatten() make everything correct, otherwise there are some shift in peaks.
            ty = np.tan(cy.flatten())  #dgc there was no flatten() in the origin neutronpy file
            tw = np.tan(cw.flatten())  #dgc with the flatten(), it is still not the same as matlab arra(1:end), however it becomes one dimension array
            tz = np.tan(dd2)
            #print(tz)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * np.exp(-0.5 * (tw ** 2)) * (
                1 + tw ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            

            for iz in range(len(tz)):
                for i in range(length):
                    dQ1 = tqx[i]  * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW  = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i]  * tz[iz]
                    H1  = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1  = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1  = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1  = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    """
                    if i==0:
                        print(dQ1[0:33])
                        print(dQ2[0:33])
                        print(dW[0:33])
                        print(inte[0,0:33])
                    """    
                    for j in range(modes):
                        add = inte[j, :] * norm * normz[iz]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])
            #print(conv)
            conv = conv * step1 ** 3 * step2 / np.sqrt(detM)
            #print(conv)
            if M[1] == 0:
                conv *= 0.79788
            if M[0] == 0:
                conv *= 0.79788 ** 3

        elif METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError('ACCURACY must be an int when using Monte Carlo method: {0}'.format(ACCURACY))
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r = np.random.randn(4, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    cw = r[3, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    tw = np.tan(cw)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2 + tw ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (
                        1 + tz ** 2) * (1 + tw ** 2)
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW  = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz
                    H1  = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1  = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1  = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1  = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000 * np.pi ** 4. / np.sqrt(detM)

        else:
            raise ValueError('Unknown METHOD: {0}. Valid options are: "fix",  "mc"'.format(METHOD))

        conv *= R0
        conv += bgr

        return conv

    def ResConvSMA(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r"""Numerically calculate the convolution of a user-defined single-mode
        cross-section function with the resolution function for a 3-axis
        neutron scattering experiment.

        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 3D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, and $\phi_2$, and 2*ACCURACY[1]+1 along $\phi_3$
            (vertical direction). 'mc': 3D Monte Carlo integration. The cross
            section is sampled in 1000*ACCURACY randomly chosen points,
            uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        self.CalcResMatHKL(hkle)
        [R0, RMS] = [np.copy(self.R0), self.RMS.copy()]

        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)

        [xvec, yvec, zvec] = self._StandardSystem()[:3]

        Mww = RMS[:, 3, 3]
        Mxw = RMS[:, 0, 3]
        Myw = RMS[:, 1, 3]

        GammaFactor = np.sqrt(Mww / 2)
        OmegaFactorx = Mxw / np.sqrt(2 * Mww)
        OmegaFactory = Myw / np.sqrt(2 * Mww)

        Mzz  = RMS[:, 2, 2]
        Mxx  = RMS[:, 0, 0]
        Mxx -= Mxw ** 2 / Mww
        Myy  = RMS[:, 1, 1]
        Myy -= Myw ** 2 / Mww
        Mxy  = RMS[:, 0, 1]
        Mxy -= Mxw * Myw / Mww

        detxy = np.sqrt(Mxx * Myy - Mxy ** 2)
        detz  = np.sqrt(Mzz)

        tqz   = 1. / detz
        tqy   = np.sqrt(Mxx) / detxy
        tqxx  = 1. / np.sqrt(Mxx)
        tqxy  = Mxy / np.sqrt(Mxx) / detxy

        [disp,    inte] = sqw(H, K, L, p)[:2]
        [modes, points] = disp.shape

        if pref is None:
            prefactor = np.ones(modes, points)
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number or output arguments in prefactor function')

        if METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError("ACCURACY (type: {0}) must be an 'int' when using Monte Carlo method".format(type(ACCURACY)))
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r   = np.random.randn(3, 1000) * np.pi - np.pi / 2
                    cx  = r[0, :]
                    cy  = r[1, :]
                    cz  = r[2, :]
                    tx  = np.tan(cx)
                    ty  = np.tan(cy)
                    tz  = np.tan(cz)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2)
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz
                    H1  = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1  = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1  = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points]  = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add   = inte[j, :] * _voigt(Omega, Gamma) * norm / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000. * np.pi ** 3

        elif METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = [7, 0]
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1   = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2   = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv  = np.zeros(length)
            [cy, cx] = np.meshgrid(dd1, dd1, indexing='ij')
            tx    = np.tan(cx.flatten())
            ty    = np.tan(cy.flatten())
            tz    = np.tan(dd2)
            norm  = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            for iz in range(tz.size):
                for i in range(length):
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz[iz]
                    H1  = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1  = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1  = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes,  points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm * normz[iz] / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 2 * step2

            if M[1] == 0:
                conv *= 0.79788
            if M[0] == 0:
                conv *= 0.79788 ** 2
        else:
            raise ValueError('Unknown METHOD: {0}. Valid options are: "fix" or "mc".'.format(METHOD))

        conv = conv * R0
        conv = conv + bgr

        return conv


    def ResolutionPlot(self, hkle=None, SMA=None, SMAp=None):
        if hkle is None:
            print("no enough parameters for plotting!")
            return
        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        #print(length)
        #print(W)
    
        center = int(round(length/2))

        #EXP=self.EXP(center)
        StyleOne = '-'
        StyleTwo = '--'

        XYAxesPos  = [0.1, 0.6, 0.36, 0.36]
        XEAxesPos  = [0.1, 0.1, 0.36, 0.36]
        YEAxesPos  = [0.6, 0.6, 0.36, 0.36]
        TxtAxesPos = [0.55, 0.0, 0.36, 0.36]
        GridPoints = 101
    
    
        # [R0,RMS]  =  ResMatS(H, K, L, W, EXP)
        self.CalcResMatHKL(hkle)
        [R0, RMS]  = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        #Q = _modvec([H, K, L], rsample)
        #uq = np.vstack((H / Q, K / Q, L / Q)) 
        uq = np.vstack((H , K , L ))
        
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W
        #print("qx:{}:".format(qx))
        #print("qy:{}:".format(qy))


        o1    = self.orient1
        o2    = self.orient2
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0
        #print(RMS.shape)
        #print(RMS)
        #print(fproject(RMS,0))
        #print(fproject(RMS,1))
        #print(fproject(RMS,2))
        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5
        #print(XWidth)
        #print(qx)
        #print(XMax)
        
        #========================================================================================================
        # plot XE projection
        fig = plt.figure()
        ((ax_xe, ax_ye), (ax_xy, ax_txt)) = fig.subplots(2, 2)
        ax_xe.set_position(XEAxesPos)
        ax_ye.set_position(YEAxesPos)
        ax_xy.set_position(XYAxesPos)
        ax_txt.set_position(TxtAxesPos)
        #xe_plot
        #ax_xe.set_position(XEAxesPos)
        ax_xe.set_xlim(XMin, XMax)
        ax_xe.set_ylim(WMin, WMax)
        #omax     =  XMax/_modvec(o1,rsample)
        #omin     =  XMin/_modvec(o2,rsample)
        #olab     =  'Qx (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
    
        ax_xe.set_xlabel('Qx[A-1]')
        ax_xe.set_ylabel('E [meV]')
    
        [proj,sec] = project(RMS,1)
        PlotEllipse(ax_xe, proj, qx, qw, StyleOne)
        PlotEllipse(ax_xe, sec,  qx, qw, StyleTwo)
    
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        Qygrid = np.linspace(YMin, YMax, GridPoints)#np.ones(Qxgrid.shape)*qy[center]
    
        Hgrid  = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid  = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid  = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
             [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
             [modes,points] =  dispersion.shape
             for mode in range(modes):
                 ax_xe.plot(Qxgrid,dispersion[mode,:],StyleOne)
   
        #ye plot
        #ax_ye.set_position(YEAxesPos)
        ax_ye.set_xlim(XMin, XMax)
        ax_ye.set_ylim(WMin, WMax)
        #omax     =  YMax/modvec(o2,rsample)
        #omin     =  YMin/modvec(o2,rsample)
        #olab     =  'Qx (units of [{}{}{}])'.format(o2[0],o2[1],o2[2])
    
        ax_ye.set_xlabel('Qx[A-1]')
        ax_ye.set_ylabel('E [meV]')
    
        [proj,sec]=project(RMS,0)
        PlotEllipse(ax_ye, proj, qy, qw, StyleOne)
        PlotEllipse(ax_ye, sec,  qy, qw, StyleTwo)
        
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        Qygrid = np.linspace(YMin, YMax, GridPoints)
        

        Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
            [dispersion, intensity, gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            [modes,points]                  =  dispersion.shape
            for mode in range(modes):
                ax_ye.plot(Qxgrid,dispersion[mode,:], StyleOne)

        #xy plot
        #ax_xy.set_position(XYAxesPos)
        ax_xy.set_xlim(XMin, XMax)
        ax_xy.set_ylim(YMin, YMax)
        #oxmax     =  YMax/modvec(o1,rsample)
        #oxmin     =  YMin/modvec(o1,rsample)
        #oymax     =  YMax/modvec(o2,rsample)
        #oymin     =  YMin/modvec(o2,rsample)
        #oxlab     =  'Qx (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
        #oylab     =  'Qy (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
    
        ax_xy.set_xlabel('Qx[A-1]')
        ax_xy.set_ylabel('Qy[A-1]')
    
        [proj,sec] = project(RMS,2)
        PlotEllipse(ax_xy, proj, qx, qy, StyleOne)
        PlotEllipse(ax_xy, sec,  qx, qy, StyleTwo)



        XWidth   =   fproject(RMS, 0)
        YWidth   =   fproject(RMS, 1)
        WWidth   =   fproject(RMS, 2)
        ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])

        XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
        YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
        WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])

        #print(center)
        matdet   =   np.linalg.det(RMS[int(center),:,:])
        
        ResVol   =   (2*np.pi)**2/np.sqrt(matdet)
        

        #ax_xy.set_position(TxtAxesPos)
        ax_txt.set_axis_off()
        ax_txt.text(0, 1.0, 'Scan center (point # {}):'.format(center))
        ax_txt.text(0, 0.88, 'H={0:.2f}  K={1:.2f}  L={2:.2f}  E={3:.2f} meV'.format(H[center], K[center], L[center], W[center]))
        ax_txt.text(0, 0.76, 'Projections on principal axes (FWHM):' )
        ax_txt.text(0, 0.64, 'Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2))
        ax_txt.text(0, 0.52, 'Bragg widths (FWHM):' )
        ax_txt.text(0, 0.40, 'Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2))
        ax_txt.text(0, 0.28, 'Resolution volume:{0:.4e}  meV/A3'.format(ResVol*2))
        ax_txt.text(0, 0.16, 'Intensity prefactor:{0:.4f}'.format(R0[center]))
        #plt.autoscale()
        plt.show()

    def PlotlyResPlot(self, hkle=None, SMA=None, SMAp=None, output_type = 'file'):
        if hkle is None:
            print("no enough parameters for plotting!")
            return
        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        print(length)

    
        center = int(round(length/2))

        #EXP=self.EXP(center)
        StyleOne = '-'
        StyleTwo = '--'

        XYAxesPos  = [0.1, 0.6, 0.36, 0.36]
        XEAxesPos  = [0.1, 0.1, 0.36, 0.36]
        YEAxesPos  = [0.6, 0.6, 0.36, 0.36]
        TxtAxesPos = [0.55, 0.0, 0.36, 0.36]
        GridPoints = 101
    
    
        # [R0,RMS]  =  ResMatS(H, K, L, W, EXP)
        self.CalcResMatHKL(hkle)
        [R0, RMS]  = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        #Q = _modvec([H, K, L], rsample)
        #uq = np.vstack((H / Q, K / Q, L / Q)) 
        uq = np.vstack((H , K , L ))
        
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W



        o1    = self.orient1
        o2    = self.orient2
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0

        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5
        #========================================================================================================
       
        plotlyfig = plysub.make_subplots(rows=2, cols=2)
               
        #xe_plot
        [proj, sec] = project(RMS, 1)

        

        [r1, x1, y1]=ProduceEllipse(proj, qx, qw)
        [r2, x2, y2]=ProduceEllipse(sec, qx, qw)

        print(x1)
        
       

        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x1[:,index], y=y1[:,index], line=dict(width=2)), row=1, col=1)
            plotlyfig.add_trace(go.Scatter(x=x2[:,index], y=y2[:,index], line=dict(width=2, dash='dash')), row=1, col=1)
        #plotlyfig.add_trace(go.Scatter(x=x1, y=y1, line=dict(width=4)), row=1, col=1)
        #plotlyfig.add_trace(go.Scatter(x=x2, y=y2, line=dict(width=4, dash='dash')), row=1, col=1)
        #plotlyfig.add_trace(go.Scatter(x2, y2, line=dict(color='firebrick', width=4), row=1, col=1)
    
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        Qygrid = np.linspace(YMin, YMax, GridPoints) 
    
        Hgrid  = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid  = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid  = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
             [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
             [modes,points] =  dispersion.shape
             for mode in range(modes):
                 plotlyfig.add_trace(go.Scatter(Qxgrid, dispersion[mode,:], line=dict(width=2,  dash='dot'), row=1, col=1))
   
        #ye_plot
        [proj,sec]=project(RMS,0)
        [r3,x3,y3]=ProduceEllipse(proj, qy, qw)
        [r4,x4,y4]=ProduceEllipse(sec, qy, qw)

        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x3[:,index], y=y3[:,index], line=dict(width=2)), row=1, col=2)
            plotlyfig.add_trace(go.Scatter(x=x4[:,index], y=y4[:,index], line=dict(width=2, dash='dash')), row=1, col=2)
        
        
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        Qygrid = np.linspace(YMin, YMax, GridPoints)
        
        Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
            [dispersion, intensity, gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            [modes,points]                  =  dispersion.shape
            for mode in range(modes):
                plotlyfig.add_trace(go.Scatter(Qxgrid, dispersion[mode,:], line=dict(width=2,  dash='dot'), row=1, col=2))
        #xy_plot
        [proj,sec] = project(RMS,2)
        [r5,x5,y5]=ProduceEllipse(proj, qx, qy)
        [r6,x6,y6]=ProduceEllipse(sec,  qx, qy)

        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x5[:,index], y=y5[:,index], line=dict(width=2)), row=2, col=1)
            plotlyfig.add_trace(go.Scatter(x=x6[:,index], y=y6[:,index], line=dict(width=2, dash='dash')), row=2, col=1)
        
                            
        # Update xaxis properties
        plotlyfig.update_xaxes(title_text="Qx[A-1]", range=[XMin, XMax], row=1, col=1)
        plotlyfig.update_xaxes(title_text="Qy[A-1]", range=[YMin, YMax], row=1, col=2)
        plotlyfig.update_xaxes(title_text="Qx[A-1]", range=[XMin, XMax], row=2, col=1)
        
        
        # Update yaxis properties
        plotlyfig.update_yaxes(title_text="E [meV]", range=[WMin, WMax], row=1, col=1)
        plotlyfig.update_yaxes(title_text="E [meV]", range=[WMin, WMax], row=1, col=2)
        plotlyfig.update_yaxes(title_text="Qy[A-1]", range=[YMin, YMax], row=2, col=1)
       

        XWidth   =   fproject(RMS, 0)
        YWidth   =   fproject(RMS, 1)
        WWidth   =   fproject(RMS, 2)
        ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])

        XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
        YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
        WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])

        #print(center)
        matdet   =   np.linalg.det(RMS[int(center),:,:])
        
        ResVol   =   (2*np.pi)**2/np.sqrt(matdet)
        

        
        text     ='            Scan center (point # {}):<br>'.format(center)
        text=text+'            H={0:.2f}  K={1:.2f}  L={2:.2f}  E={3:.2f} meV<br>'.format(H[center], K[center], L[center], W[center])
        text=text+'            Projections on principal axes (FWHM):<br>'
        text=text+'            Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1<br>'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2)
        text=text+'            Bragg widths (FWHM):<br>'
        text=text+'            Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1<br>'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2)
        text=text+'            Resolution volume:{0:.4e}  meV/A3<br>'.format(ResVol*2)
        text=text+'            Intensity prefactor:{0:.4f} <br>'.format(R0[center])
        plotlyfig.add_annotation(text=text, xref="paper", yref="paper", x=0.66, y=0.20,  align="left", showarrow=False)
      
        """
        plotlyfig.add_annotation(text='Scan center (point # {}):'.format(center), xref="paper", yref="paper", x=0.6, y=0.45,  align="left", showarrow=False)
        plotlyfig.add_annotation(text='H={0:.2f}  K={1:.2f}  L={2:.2f}  E={3:.2f} meV'.format(H[center], K[center], L[center], W[center]), xref="paper", yref="paper", x=0.65, y=0.4, align="left", showarrow=False)
        plotlyfig.add_annotation(text='Projections on principal axes (FWHM):', xref="paper", yref="paper", x=0.65, y=0.35, showarrow=False)
        plotlyfig.add_annotation(text='Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2), xref="paper", yref="paper", x=0.7, y=0.3, align="left",  showarrow=False)
        plotlyfig.add_annotation(text='Bragg widths (FWHM):', xref="paper", yref="paper", x=0.56, y=0.25, align="left", showarrow=False)
        plotlyfig.add_annotation(text='Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2), xref="paper", yref="paper", x=0.7, y=0.2, align="left",  showarrow=False)
        plotlyfig.add_annotation(text='Resolution volume:{0:.4e}  meV/A3'.format(ResVol*2), xref="paper", yref="paper", x=0.65, y=0.15, align="left",  showarrow=False)
        plotlyfig.add_annotation(text='Intensity prefactor:{0:.4f}'.format(R0[center]), xref="paper", yref="paper", x=0.6, y=0.1, align="left",  showarrow=False)
        """
        plotlyfig.update_layout(
                    autosize=True,
                    width=800,
                    height=700,
                    margin=dict(
                        l=20,
                        r=20,
                        b=20,
                        t=20,
                        pad=4
                    ),
                    paper_bgcolor="LightSteelBlue",
                    showlegend = False,
                )

        plot_div = plyplt.plot(plotlyfig, output_type = output_type)
        return plot_div





    def ResolutionPlotProj(self, ax=None, qslice='QxQy',hkle=None, SMA=None, SMAp=None):
        if hkle is None:
            print("no enough parameters for plotting!")
            return

        H, K, L, W   = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        #print(length)
        #print(W)
    
        center = int(round(length/2))

    
        #EXP=self.EXP(center)
        StyleOne   = '-'
        StyleTwo   = '--'
        GridPoints = 101
    
    
        # [R0,RMS]  =  ResMatS(H, K, L, W, EXP)
        self.CalcResMatHKL(hkle)
        [R0, RMS] = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        #Q = _modvec([H, K, L], rsample)
        #uq = np.vstack((H / Q, K / Q, L / Q)) 
        uq = np.vstack((H , K , L ))
        
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W
        #print("qx:{}:".format(qx))
        #print("qy:{}:".format(qy))


        o1    = self.orient1
        o2    = self.orient2
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0
        #print(RMS.shape)
        #print(RMS)
        #print(fproject(RMS,0))
        #print(fproject(RMS,1))
        #print(fproject(RMS,2))
        
        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5
        #print(XWidth)
        #print(qx)
        #print(XMax)
        
        #========================================================================================================
        # plot XE projection
        if qslice == "QxE":

            #ax.set_position(AxPos)
            ax.set_xlim(XMin, XMax)
            ax.set_ylim(WMin, WMax)
            #omax     =  XMax/_modvec(o1,rsample)
            #omin     =  XMin/_modvec(o2,rsample)
            #olab     =  'Qx (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
        
            ax.set_xlabel('Qx[A-1]')
            ax.set_ylabel('E [meV]')
        
            [proj,sec]=project(RMS,1)
            PlotEllipse(ax, proj, qx, qw, StyleOne)
            PlotEllipse(ax, sec,  qx, qw, StyleTwo)
        
            Qxgrid = np.linspace(XMin, XMax, GridPoints)
            Qygrid = np.linspace(YMin, YMax, GridPoints)
        
            Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
            Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
            Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
        
            if SMA is not None:
                 [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
                 [modes,points] =  dispersion.shape
                 for mode in range(modes):
                     ax.plot(Qxgrid,dispersion[mode,:],StyleOne)
   
        elif qslice == "QyE": 
            #ye plot
            #ax.set_position(AxPos)
            ax.set_xlim(XMin, XMax)
            ax.set_ylim(WMin, WMax)
            #omax     =  YMax/modvec(o2,rsample)
            #omin     =  YMin/modvec(o2,rsample)
            #olab     =  'Qx (units of [{}{}{}])'.format(o2[0],o2[1],o2[2])
        
            ax.set_xlabel('Qy[A-1]')
            ax.set_ylabel('E [meV]')
        
            [proj,sec]=project(RMS,0)
            PlotEllipse(ax, proj, qy, qw, StyleOne)
            PlotEllipse(ax, sec,  qy, qw, StyleTwo)
            
            Qxgrid = np.linspace(XMin, XMax, GridPoints)
            Qygrid = np.linspace(YMin, YMax, GridPoints)
            
        
            Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
            Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
            Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
        
            if SMA is not None:
                [dispersion,intensity,gamma]  =  SMA(Hgrid, Kgrid, Lgrid, SMAp)
                [modes,points] =  dispersion.shape
                for mode in range(modes):
                    ax.plot(Qxgrid,dispersion[mode,:], StyleOne)

        elif qslice == "QxQy": 
            #xy plot
            #ax.set_position(AxPos)
            ax.set_xlim(XMin, XMax)
            ax.set_ylim(YMin, YMax)
            #oxmax     =  YMax/modvec(o1,rsample)
            #oxmin     =  YMin/modvec(o1,rsample)
            #oymax     =  YMax/modvec(o2,rsample)
            #oymin     =  YMin/modvec(o2,rsample)
            #oxlab     =  'Qx (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
            #oylab     =  'Qy (units of [{}{}{}])'.format(o1[0],o1[1],o1[2])
        
            ax.set_xlabel('Qx[A-1]')
            ax.set_ylabel('Qy[A-1]')
        
            [proj,sec]=project(RMS,2)
            PlotEllipse(ax, proj, qx, qy, StyleOne)
            PlotEllipse(ax, sec,  qx, qy, StyleTwo)

        elif qslice == "Txt": 

            XWidth   =   fproject(RMS, 0)
            YWidth   =   fproject(RMS, 1)
            WWidth   =   fproject(RMS, 2)
            ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])
    
            XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
            YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
            WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])
    
            #print(center)
            matdet   =   np.linalg.det(RMS[int(center),:,:])
            
            ResVol   =   (2*np.pi)**2/np.sqrt(matdet)
            
    
    
            #ax_xy.set_position(TxtAxesPos)
            ax.set_axis_off()
            strtemp='Scan center (point # {}):'.format(center)
            ax.text(0, 1.0, strtemp)
            #self.description_string='Instrument Description\n'+strtemp
            strtemp='H={0:.2f}  K={1:.2f}  L={2:.2f}  E={3:.2f} meV'.format(H[center], K[center], L[center], W[center])
            ax.text(0, 0.88, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Projections on principal axes (FWHM):'
            ax.text(0, 0.76, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2)
            ax.text(0, 0.64, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Bragg widths (FWHM):'
            ax.text(0, 0.52,  strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Qx: {0:.4f} A-1 Qy: {1:.4f} A-1 Qz: {2:.4f} A-1'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2)
            ax.text(0, 0.40, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Resolution volume:{0:.4e}  meV/A3'.format(ResVol*2)
            ax.text(0, 0.28, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp
            strtemp='Intensity prefactor:{0:.4f}'.format(R0[center])
            ax.text(0, 0.16, strtemp)
            #self.description_string=self.description_string+'\n'+strtemp 
            #print(self.description_string)         

        else:
            print("unknown qslice, please give only QxE, QyE, QxQy, or Txt")
        
        try:
            method = ['Cooper-Nathans', 'Popovici'][self.method]
        except AttributeError:
            method = 'Cooper-Nathans'
        frame = '[Q1,Q2,Qz,E]'

        if hasattr(self, 'infin'):
            FX = 2 * int(self.infin == -1) + int(self.infin == 1)
        else:
            FX = 2

        if self.RMS.shape == (4, 4):
            NP   = self.RMS
            R0   = float(self.R0)
            hkle = self.HKLE
        else:
            NP   = self.RMS[0]
            R0   = self.R0[0]
            hkle = [self.H[0], self.K[0], self.L[0], self.W[0]]

        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP))
        bragg_widths = get_bragg_widths(NP)

        if getattr(self, "instr_type") == "tas":
            angles = self.get_angles_and_Q(hkle)[0]

            text_format = ['Method: {0}'.format(method),
                           'Position HKLE [{0}]\n'.format(dt.datetime.now().strftime('%d-%b-%Y %H:%M:%S')),
                           ' [Q_H, Q_K, Q_L, E] = {0} \n'.format(self.HKLE),
                           'Resolution Matrix M in {0} (M/10^4):'.format(frame),
                           '[[{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 0] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 1] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 2] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]]\n'.format(*NP[:, 3] / 1.0e4),
                           'Resolution volume:   V_0={0:.6f} meV/A^3'.format(2 * ResVol),
                           'Intensity prefactor: R_0={0:.3f}'.format(R0),
                           'Bragg width in [Q_1,Q_2,E] (FWHM):',
                           ' dQ_1={0:.3f} dQ_2={1:.3f} [A-1] dE={2:.3f} [meV]'.format(bragg_widths[0], bragg_widths[1],
                                                                                       bragg_widths[4]),
                           ' dQ_z={0:.3f} Vanadium width V={1:.3f} [meV]'.format(*bragg_widths[2:4]),
                           'Instrument parameters:',
                           ' DM  =  {0:.3f} ETAM= {1:.3f} SM={2}'.format(self.mono.d, self.mono.mosaic, self.mono.dir),
                           ' KFIX=  {0:.3f} FX  = {1} SS={2}'.format(Energy(energy=self.efixed).wavevector, FX,
                                                                       self.sample.dir),
                           ' DA  =  {0:.3f} ETAA= {1:.3f} SA={2}'.format(self.ana.d, self.ana.mosaic, self.ana.dir),
                           ' A1= {0:.2f} A2={1:.2f} A3={2:.2f} A4={3:.2f} A5={4:.2f} A6={5:.2f} [deg]'.format(*angles),
                           'Collimation [arcmin]:',
                           ' Horizontal: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.hcol),
                           ' Vertical: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.vcol),
                           'Sample:',
                           ' a, b, c  =  [{0}, {1}, {2}] [Angs]'.format(self.sample.a, self.sample.b, self.sample.c),
                           ' Alpha, Beta, Gamma  =  [{0}, {1}, {2}] [deg]'.format(self.sample.alpha, self.sample.beta,
                                                                                  self.sample.gamma),
                           ' U  =  {0} [rlu]   V  =  {1} [rlu]'.format(self.orient1, self.orient2)]        
        
        self.description_string = '\n'.join(text_format)
        
        plt.show()






    

    def ResolutionPlot3D(self, hkle=None, RANGE=None, EllipStyle=None, XYStyle=None, XEStyle=None, YEStyle=None, SMA=None, SMAp=None, SXg=None, SYg=None):
        # this is to plot the resolution function ellipsoid in Q-enery space
        

        plotly_clr_list = ['Bluered', 'Greens','Hot', 'Jet', 'Earth' ]
        matplot_clrs    = ['r','b','g','y','c'] 
        cmap_list       = ['Spectral', 'PuOr', 'PiYG', 'BrBG', 'BuPu']
        
        SMAGridPoints   = 21
        EllipGridPoints = 21
        
        bPlotInQSpace   = 1
        
        
        if len(RANGE) < 6    :  print('Range must have the form [Xmin Xmax Ymin Ymax Emin Emax]')
        if EllipStyle is None: EllipStyle = 'g'
        if XYStyle    is None: XYStyle    = '-p'
        if XEStyle    is None: XYStyle    = '-c'
        if YEStyle    is None: XYStyle    = '-b'
        if SMA        is None: print('SMA is not provided')

        if SMA is not None and (SXg is None or SYg is None):
            SX    =    np.linspace(RANGE(0),RANGE(1),SMAGridPoints)
            SY    =    np.linspace(RANGE(2),RANGE(3),SMAGridPoints)
            [SXg, SYg] = np.meshgrid(SX,SY)


        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
 
        center = int(round(length/2)) #you must convert to integer

        self.CalcResMatHKL(hkle)
        RMS   =   self.RMS.copy()

        #xvec y are orient1 and orient2 vector zvec is vertical on sample
        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        
        #convert HKL into Qx(A-1) in scattering plane x and y
        uq = np.vstack([H, K, L])
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W


        plotlyfig=go.Figure()#plotly method
        
        fig=plt.figure()
        """
        ######ax3d=fig.add_subplot(111, projection='3d')
        if bPlotInQSpace   == 1:
            ######ax3d.set_xlabel('Qx (A-1)')
            ######ax3d.set_ylabel('Qy (A-1)')
        else:
            ######ax3d.set_xlabel('Qx (rlu)')
            ######ax3d.set_ylabel('Qy (rlu)')            
        ######ax3d.set_zlabel('E (meV)')
        """
       
        #plot ellipsoids
        wx=fproject(RMS,0)
        wy=fproject(RMS,1)
        ww=fproject(RMS,2)

        for point in range(length):

            xlow  = -wx[point]*1.5 + qx[point]
            xhigh =  wx[point]*1.5 + qx[point]
            ylow  = -wy[point]*1.5 + qy[point]
            yhigh =  wy[point]*1.5 + qy[point]
            xcenter  =  qx[point]
            ycenter  =  qy[point]
            zcenter  =  qw[point]
            
            if bPlotInQSpace == 0: #plot in HKL

                Hlow = xlow*xvec[0]+ylow*yvec[0]
                Klow = xlow*xvec[1]+ylow*yvec[1]
                Llow = xlow*xvec[2]+ylow*yvec[2]
                
                xlow = Hlow*self.orient1[0]+Klow*self.orient1[1]+Llow*self.orient1[2]
                ylow = Hlow*self.orient2[0]+Klow*self.orient2[1]+Llow*self.orient2[2]
                
                Hhigh=xhigh*xvec[0]+yhigh*yvec[0]
                Khigh=xhigh*xvec[1]+yhigh*yvec[1]
                Lhigh=xhigh*xvec[2]+yhigh*yvec[2]
                
                xhigh=Hhigh*self.orient1[0]+Khigh*self.orient1[1]+Lhigh*self.orient1[2]
                yhigh=Hhigh*self.orient2[0]+Khigh*self.orient2[1]+Lhigh*self.orient2[2]

                Hcenter=qx[point]*xvec[0]+qy[point]*yvec[0]
                Kcenter=qx[point]*xvec[1]+qy[point]*yvec[1]
                Lcenter=qx[point]*xvec[2]+qy[point]*yvec[2]
                
                xcenter=Hcenter*self.orient1[0]+Kcenter*self.orient1[1]+Lcenter*self.orient1[2]
                ycenter=Hcenter*self.orient2[0]+Kcenter*self.orient2[1]+Lcenter*self.orient2[2]               
            
            x = np.linspace(xlow,  xhigh, EllipGridPoints)
            y = np.linspace(ylow,  yhigh, EllipGridPoints)
            z = np.linspace(-ww[point]*1.5, ww[point]*1.5, EllipGridPoints) + zcenter
            [xg,yg,zg] = np.meshgrid(x,y,z)
            #print(x.shape)
            #print(xg.shape)
            
            ee = ( RMS[point,0,0]*(xg-xcenter)**2 + 
                   RMS[point,1,1]*(yg-ycenter)**2 +
                   RMS[point,2,2]*(zg-zcenter)**2 + 
                   2*RMS[point,0,1]*(xg-xcenter)*(yg-ycenter) +
                   2*RMS[point,0,2]*(xg-xcenter)*(zg-zcenter) +
                   2*RMS[point,2,1]*(zg-zcenter)*(yg-ycenter))

           
            ######[verts, faces, normals, values] = measure.marching_cubes_lewiner(ee, 2*np.log(2), spacing=((xhigh-xlow)/(EllipGridPoints-1), (yhigh-ylow)/(EllipGridPoints-1), ww[point]*3/(EllipGridPoints-1)))
            #verts is the coorditnates xyz of the ellipsoids around the center of (0 0 0) and count the ellipsoids width 
            #the marching_cubes,_lewiner calculate not in a center of the ellipsoids but in a box [0, 3wx] [0, 3wy], [0 3ww] so have to take away that
            ######verts = verts + [xcenter, ycenter, zcenter]-[ (xhigh-xlow)/2, (yhigh-ylow)/2, ww[point]*1.5]   #translate this coordinates to the center of the current (qx,qy and w)
            ######ax3d.plot_trisurf(verts[:,0],verts[:,1],faces, verts[:,2],cmap=cmap_list[point],lw=1, zorder=point*2)

            #the following line is another way to plot it  by using matplotlib
            #mesh = Poly3DCollection(verts[faces])
            #mesh.set_edgecolor('g')
            #mesh.set_facecolor(matplot_clrs[point])
            #ax3d.add_collection3d(mesh)
            

            
            #********************use plotly to plot into web browser*****************
            
            plotlyfig.add_trace(go.Isosurface(
                    x=xg.flatten(), y=yg.flatten(), z=zg.flatten(), 
                    value=ee.flatten(),colorscale=plotly_clr_list[point], isomin=2*np.log(2), isomax=2*np.log(2)))
            
            plotlyfig.update_layout(xaxis_title='Qx [A-1]')
            plotlyfig.update_yaxes(title_text='Qy [A-1]')
            plotlyfig.update_zaxes(title_text='E [meV]')
                       
            #************************************************************************            
            #********************use mayavi to plot *********************************
            pts = np.empty(xg.shape + (3,), dtype=float)
            pts[..., 0] = xg
            pts[..., 1] = yg/5
            pts[..., 2] = zg/100
            #print(pts[...,0].shape)
            pts = pts.transpose(0, 1, 2, 3).copy()
            #print(pts.shape)
            pts.shape = [int(pts.size / 3), 3]
            #print(pts.shape)
            ee = ee.T.copy()
            #print(xg.shape)
            #print(pts[...,0].shape)

            # Create the dataset.
            ####sg = tvtk.StructuredGrid(dimensions=xg.shape, points=pts)
            ####sg.point_data.scalars = ee.ravel()
            ####sg.point_data.scalars.name = 'ellipsoid'


            # Now visualize the data.
            ####d = mlab.pipeline.add_dataset(sg)
            #gx = mlab.pipeline.grid_plane(d)
            #gx.grid_plane.axis = 'x'
            #gy = mlab.pipeline.grid_plane(d)
            #gy.grid_plane.axis = 'y'
            #gz = mlab.pipeline.grid_plane(d)
            #gz.grid_plane.axis = 'z'
            ####mlab.pipeline.iso_surface(d,contours=[2*np.log(2),], opacity=0.5)
            #src=mlab.pipeline.scalar_field(ee)
            #mlab.pipeline.iso_surface(src, contours=[2*np.log(2),], opacity=0.5)
            #mlab.pipeline.iso_surface(src, contours=[ee.max()-0.1*ee.ptp(),], opacity=0.5)
            #mlab.contour3d(xg, yg, zg, ee, vmax=2*np.log(2), vmin=2*np.log(2))
            #mlab.axes()
            #mlab.show()
            


        
        #plot dispersion surfaces  need to be tested
        if SMA is not None:
            Hgrid = SXg*xvec[0]+SYg*yvec[0]
            Kgrid = SXg*xvec[1]+SYg*yvec[1]
            Lgrid = SXg*xvec[2]+SYg*yvec[2]
            
            Agrid=Hgrid*self.orient1[0]+Kgrid*self.orient1[1]+Lgrid*self.orient1[2]
            Bgrid=Hgrid*self.orient2[0]+Kgrid*self.orient2[1]+Lgrid*self.orient2[2]
    
    
            [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            #print(Hgrid.shape)

            [xpoints, ypoints, modes] = dispersion.shape
            for mode in range(modes):
                SZ=dispersion[:,:,mode]
                #print(SZ)
                #if  SZ > RANGE[5]: SZ = RANGE[5]
                #if  SZ < RANGE[4]: SZ = RANGE[4]
                if bPlotInQSpace ==1:
                    ######ax3d.plot_surface(SXg,SYg,SZ, alpha=0.5, color='g')
                    #disp_verts = [list(zip(SXg, SYg, SZ))]    
                    #disp_surf = Poly3DCollection(disp_verts)
                    #ax3d.add_collection3d(disp_surf)
                    #mlab.contour_surf(SXg,SYg,SZ)
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity=0.2))
                else: 
                    ######ax3d.plot_surface(Agrid,Bgrid,SZ, zorder=-1)
                    
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity=0.2))

        
       
        #plot projections
        [proj3,sec]  =  project(RMS,2)
        [proj2,sec]  =  project(RMS,1)
        [proj1,sec]  =  project(RMS,0)
        phi =np.linspace(0.1, 2*np.pi+0.1,1001) # 0.1:2*pi/3000:2*pi+0.1
        
        for i in range(length):
            r3  =  np.sqrt(2*np.log(2)/(proj3[i,0,0]*np.cos(phi)**2+proj3[i,1,1]*np.sin(phi)**2+2*proj3[i,0,1]*np.cos(phi)*np.sin(phi)))
            r2  =  np.sqrt(2*np.log(2)/(proj2[i,0,0]*np.cos(phi)**2+proj2[i,1,1]*np.sin(phi)**2+2*proj2[i,0,1]*np.cos(phi)*np.sin(phi)))
            r1  =  np.sqrt(2*np.log(2)/(proj1[i,0,0]*np.cos(phi)**2+proj1[i,1,1]*np.sin(phi)**2+2*proj1[i,0,1]*np.cos(phi)*np.sin(phi)))
            xproj3 = r3*np.cos(phi)+qx[i]
            yproj3 = r3*np.sin(phi)+qy[i]
            zproj3 = np.ones(xproj3.shape)*RANGE[4]
            xproj2 = r2*np.cos(phi)+qx[i]
            zproj2 = r2*np.sin(phi)+qw[i]
            yproj2 = np.ones(xproj2.shape)*RANGE[3]
            yproj1 = r1*np.cos(phi)+qy[i]
            zproj1 = r1*np.sin(phi)+qw[i]
            xproj1 = np.ones(yproj1.shape)*RANGE[0]

            #convet to the HKL space rather than Q (A -1)
            if bPlotInQSpace == 1: #plot in Q(A-1)
           
                ######ax3d.plot(xproj1,yproj1,zproj1)
                ######ax3d.plot(xproj2,yproj2,zproj2)
                ######ax3d.plot(xproj3,yproj3,zproj3)
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=xproj1,y=yproj1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj2,y=yproj2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj3,y=yproj3,z=zproj3, mode='lines'))
                #************************************************************************
                ####mlab.plot3d(xproj1,yproj1/5,zproj1/100, tube_radius=None, color=(1, 0.4, 0))
                #gx = mlab.pipeline.grid_plane(pa)
                ####mlab.plot3d(xproj2,yproj2/5,zproj2/100, tube_radius=None, color=(0.2, 1, 0))
                ####mlab.plot3d(xproj3,yproj3/5,zproj3/100, tube_radius=None, color=(0, 0, 1))
            else: #plot in HKL
                H3 = xproj3*xvec[0]+yproj3*yvec[0]
                K3 = xproj3*xvec[1]+yproj3*yvec[1]
                L3 = xproj3*xvec[2]+yproj3*yvec[2]
                A3 = H3*self.orient1[0]+K3*self.orient1[1]+L3*self.orient1[2]
                B3 = H3*self.orient2[0]+K3*self.orient2[1]+L3*self.orient2[2]
    
                H2 = xproj2*xvec[0]+yproj2*yvec[0]
                K2 = xproj2*xvec[1]+yproj2*yvec[1]
                L2 = xproj2*xvec[2]+yproj2*yvec[2]
                A2 = H2*self.orient1[0]+K2*self.orient1[1]+L2*self.orient1[2]
                B2 = H2*self.orient2[0]+K2*self.orient2[1]+L2*self.orient2[2]
    
                H1 = xproj1*xvec[0]+yproj1*yvec[0]
                K1 = xproj1*xvec[1]+yproj1*yvec[1]
                L1 = xproj1*xvec[2]+yproj1*yvec[2] 
                A1 = H1*self.orient1[0]+K1*self.orient1[1]+L1*self.orient1[2]
                B1 = H1*self.orient2[0]+K1*self.orient2[1]+L1*self.orient2[2]
           
                ######ax3d.plot(A1,B1,zproj1)
                ######ax3d.plot(A2,B2,zproj2)
                ######ax3d.plot(A3,B3,zproj3)
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=A1,y=B1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A2,y=B2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A3,y=B3,z=zproj3, mode='lines'))
                #************************************************************************
                ####mlab.plot3d(A1,B1,zproj1/100)
                ####mlab.plot3d(A2,B2,zproj2/100)
                ####mlab.plot3d(A3,B3,zproj3/100)

           

           
        ######ax3d.set_xlim([ RANGE[0], RANGE[1] ])
        ######ax3d.set_ylim([ RANGE[2], RANGE[3] ])
        ######ax3d.set_zlim([ RANGE[4], RANGE[5] ])
        ######ax3d.autoscale()
        print("updated 202504")
        plt.set_cmap('Spectral')
        plt.show()
        
        plyplt.plot(plotlyfig)
        ####mlab.show()



    def PlotlyResPlot3D(self, hkle=None, RANGE=None, EllipStyle=None, XYStyle=None, XEStyle=None, YEStyle=None, SMA=None, SMAp=None, SXg=None, SYg=None, output_type='file'):
        # this function use plotly to plot the resolution of a tas instrument
        
        plotly_clr_list = ['Bluered', 'Greens','Hot', 'Jet', 'Earth' ]

        SMAGridPoints   = 21
        EllipGridPoints = 21
        
        bPlotInQSpace   = 1
        
        if len(RANGE) < 6    :  print('Range must have the form [Xmin Xmax Ymin Ymax Emin Emax]')
        if EllipStyle is None: EllipStyle = 'g'
        if XYStyle    is None: XYStyle    = '-p'
        if XEStyle    is None: XYStyle    = '-c'
        if YEStyle    is None: XYStyle    = '-b'
        if SMA        is None: print('SMA is not provided')

        if SMA is not None and (SXg is None or SYg is None):
            SX    =    np.linspace(RANGE(0),RANGE(1),SMAGridPoints)
            SY    =    np.linspace(RANGE(2),RANGE(3),SMAGridPoints)
            [SXg, SYg] = np.meshgrid(SX,SY)


        H, K, L, W = hkle
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
 
        center = int(round(length/2))             #you must convert to integer

        self.CalcResMatHKL(hkle)
        RMS   =   self.RMS.copy()

        #xvec y are orient1 and orient2 vector zvec is vertical on sample
        [xvec,yvec,zvec,sample,rsample]  =  self._StandardSystem()
        
        #convert HKL into Qx(A-1) in scattering plane x and y
        uq = np.vstack([H, K, L])
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W


        plotlyfig=go.Figure()  #plotly method
        
        #plot ellipsoids
        wx=fproject(RMS,0)
        wy=fproject(RMS,1)
        ww=fproject(RMS,2)

        for point in range(length):

            xlow  = -wx[point]*1.5 + qx[point]
            xhigh =  wx[point]*1.5 + qx[point]
            ylow  = -wy[point]*1.5 + qy[point]
            yhigh =  wy[point]*1.5 + qy[point]
            xcenter  =  qx[point]
            ycenter  =  qy[point]
            zcenter  =  qw[point]
            
            if bPlotInQSpace == 0: #plot in HKL

                Hlow = xlow*xvec[0]+ylow*yvec[0]
                Klow = xlow*xvec[1]+ylow*yvec[1]
                Llow = xlow*xvec[2]+ylow*yvec[2]
                
                xlow = Hlow*self.orient1[0]+Klow*self.orient1[1]+Llow*self.orient1[2]
                ylow = Hlow*self.orient2[0]+Klow*self.orient2[1]+Llow*self.orient2[2]
                
                Hhigh=xhigh*xvec[0]+yhigh*yvec[0]
                Khigh=xhigh*xvec[1]+yhigh*yvec[1]
                Lhigh=xhigh*xvec[2]+yhigh*yvec[2]
                
                xhigh=Hhigh*self.orient1[0]+Khigh*self.orient1[1]+Lhigh*self.orient1[2]
                yhigh=Hhigh*self.orient2[0]+Khigh*self.orient2[1]+Lhigh*self.orient2[2]

                Hcenter=qx[point]*xvec[0]+qy[point]*yvec[0]
                Kcenter=qx[point]*xvec[1]+qy[point]*yvec[1]
                Lcenter=qx[point]*xvec[2]+qy[point]*yvec[2]
                
                xcenter=Hcenter*self.orient1[0]+Kcenter*self.orient1[1]+Lcenter*self.orient1[2]
                ycenter=Hcenter*self.orient2[0]+Kcenter*self.orient2[1]+Lcenter*self.orient2[2]               
            
            x = np.linspace(xlow,  xhigh, EllipGridPoints)
            y = np.linspace(ylow,  yhigh, EllipGridPoints)
            z = np.linspace(-ww[point]*1.5, ww[point]*1.5, EllipGridPoints) + zcenter
            [xg,yg,zg] = np.meshgrid(x,y,z)
            #print(x.shape)
            #print(xg.shape)
            
            ee = ( RMS[point,0,0]*(xg-xcenter)**2 + 
                   RMS[point,1,1]*(yg-ycenter)**2 +
                   RMS[point,2,2]*(zg-zcenter)**2 + 
                   2*RMS[point,0,1]*(xg-xcenter)*(yg-ycenter) +
                   2*RMS[point,0,2]*(xg-xcenter)*(zg-zcenter) +
                   2*RMS[point,2,1]*(zg-zcenter)*(yg-ycenter))

            #********************use plotly to plot into web browser*****************
            plotlyfig.add_trace(go.Isosurface(
                    x=xg.flatten(), y=yg.flatten(), z=zg.flatten(), 
                    value=ee.flatten(),colorscale=plotly_clr_list[point], isomin=2*np.log(2), isomax=2*np.log(2)))
            
            plotlyfig.update_layout(xaxis_title='Qx [A-1]')
            plotlyfig.update_yaxes(title_text='Qy [A-1]')
            #plotlyfig.update_zaxes(title_text='E [meV]')
                       
            #************************************************************************            
        
        #plot dispersion surfaces  need to be tested
        if SMA is not None:
            Hgrid = SXg*xvec[0]+SYg*yvec[0]
            Kgrid = SXg*xvec[1]+SYg*yvec[1]
            Lgrid = SXg*xvec[2]+SYg*yvec[2]
            
            Agrid=Hgrid*self.orient1[0]+Kgrid*self.orient1[1]+Lgrid*self.orient1[2]
            Bgrid=Hgrid*self.orient2[0]+Kgrid*self.orient2[1]+Lgrid*self.orient2[2]
    
    
            [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            #print(Hgrid.shape)

            [xpoints, ypoints, modes] = dispersion.shape
            for mode in range(modes):
                SZ=dispersion[:,:,mode]
                #print(SZ)
                #if  SZ > RANGE[5]: SZ = RANGE[5]
                #if  SZ < RANGE[4]: SZ = RANGE[4]
                if bPlotInQSpace ==1:
                    ######ax3d.plot_surface(SXg,SYg,SZ, alpha=0.5, color='g')
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity=0.2))
                else: 
                    ######ax3d.plot_surface(Agrid,Bgrid,SZ, zorder=-1)
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity=0.2))

        #plot projections
        [proj3,sec]  =  project(RMS,2)
        [proj2,sec]  =  project(RMS,1)
        [proj1,sec]  =  project(RMS,0)
        phi =np.linspace(0.1, 2*np.pi+0.1,1001) # 0.1:2*pi/3000:2*pi+0.1
        
        for i in range(length):
            r3     =  np.sqrt(2*np.log(2)/(proj3[i,0,0]*np.cos(phi)**2+proj3[i,1,1]*np.sin(phi)**2+2*proj3[i,0,1]*np.cos(phi)*np.sin(phi)))
            r2     =  np.sqrt(2*np.log(2)/(proj2[i,0,0]*np.cos(phi)**2+proj2[i,1,1]*np.sin(phi)**2+2*proj2[i,0,1]*np.cos(phi)*np.sin(phi)))
            r1     =  np.sqrt(2*np.log(2)/(proj1[i,0,0]*np.cos(phi)**2+proj1[i,1,1]*np.sin(phi)**2+2*proj1[i,0,1]*np.cos(phi)*np.sin(phi)))
            xproj3 = r3*np.cos(phi)+qx[i]
            yproj3 = r3*np.sin(phi)+qy[i]
            zproj3 = np.ones(xproj3.shape)*RANGE[4]
            xproj2 = r2*np.cos(phi)+qx[i]
            zproj2 = r2*np.sin(phi)+qw[i]
            yproj2 = np.ones(xproj2.shape)*RANGE[3]
            yproj1 = r1*np.cos(phi)+qy[i]
            zproj1 = r1*np.sin(phi)+qw[i]
            xproj1 = np.ones(yproj1.shape)*RANGE[0]

            #convet to the HKL space rather than Q (A -1)
            if bPlotInQSpace == 1: #plot in Q(A-1)
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=xproj1,y=yproj1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj2,y=yproj2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj3,y=yproj3,z=zproj3, mode='lines'))
                #************************************************************************

            else: #plot in HKL
                H3 = xproj3*xvec[0]+yproj3*yvec[0]
                K3 = xproj3*xvec[1]+yproj3*yvec[1]
                L3 = xproj3*xvec[2]+yproj3*yvec[2]
                A3 = H3*self.orient1[0]+K3*self.orient1[1]+L3*self.orient1[2]
                B3 = H3*self.orient2[0]+K3*self.orient2[1]+L3*self.orient2[2]
    
                H2 = xproj2*xvec[0]+yproj2*yvec[0]
                K2 = xproj2*xvec[1]+yproj2*yvec[1]
                L2 = xproj2*xvec[2]+yproj2*yvec[2]
                A2 = H2*self.orient1[0]+K2*self.orient1[1]+L2*self.orient1[2]
                B2 = H2*self.orient2[0]+K2*self.orient2[1]+L2*self.orient2[2]
    
                H1 = xproj1*xvec[0]+yproj1*yvec[0]
                K1 = xproj1*xvec[1]+yproj1*yvec[1]
                L1 = xproj1*xvec[2]+yproj1*yvec[2] 
                A1 = H1*self.orient1[0]+K1*self.orient1[1]+L1*self.orient1[2]
                B1 = H1*self.orient2[0]+K1*self.orient2[1]+L1*self.orient2[2]
           
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=A1,y=B1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A2,y=B2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A3,y=B3,z=zproj3, mode='lines'))
                #************************************************************************

        plot_div = plyplt.plot(plotlyfig, output_type = output_type)
        return plot_div
        ####mlab.show()  
        
        
        

    def S2R(self, qx, qy, qz):
        #===================================================================================
        # Given cartesian coordinates of a vector in the S-system, calculate its
        # reciprocal-space coordinates (Miller indexes).
        [xvec,yvec,zvec, sample, rsample]=self._StandardSystem()
        H=qx*xvec[0] + qy*yvec[0]+qz*zvec[0]
        K=qx*xvec[1] + qy*yvec[1]+qz*zvec[1]
        L=qx*xvec[2] + qy*yvec[2]+qz*zvec[2]
        q = np.sqrt(qx**2 + qy**2 + qz**2)
        
        return [H, K, L, q]
    
    
    def R2S(self, H,K,L):
        # Given reciprocal-space coordinates of a vector, calculates its cartesian
        #coordinates in the S-System.
        uq = np.vstack([H, K, L])    
        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        qx = _scalar(uq,xvec,rsample)
        qy = _scalar(uq,yvec,rsample)
        qz = _scalar(uq,zvec,rsample)
        q  = _modvec(uq,rsample)
        
        return [qx, qy, qz, q]
    
 




def  fproject(mat, i):
    
    if i==0:
        v,j=2,1
    elif i==1:
        v,j=0,2
    elif i==2:
        v,j=0,1
    else:
        print("Error: improper i value!, it should be 0 or 1 or 2")
        return
    if not isinstance(mat, np.ndarray):
        mat=np.array(mat)
    if mat.ndim<3:
        mat=np.reshape(mat, [-1, mat.shape[0],mat.shape[1]])
    #print(mat.shape)    
    
    [a,b,c]=mat.shape
    #print(mat.dimension)
    #print(a, b, c)
    #print(i, j, v)
    #print(mat[i,i,:])
    #print(mat[v,v,:])
    proj=np.zeros([a,2,2])

    proj[:,0,0] = mat[:,i,i] - mat[:,i,v]**2/mat[:,v,v]
    proj[:,0,1] = mat[:,i,j] - mat[:,i,v]*mat[:,j,v]/mat[:,v,v]
    proj[:,1,0] = mat[:,j,i] - mat[:,j,v]*mat[:,i,v]/mat[:,v,v]
    proj[:,1,1] = mat[:,j,j] - mat[:,j,v]**2/mat[:,v,v]
    hwhm = proj[:,0,0]-proj[:,0,1]**2/proj[:,1,1]

    hwhm = np.sqrt(2*np.log(2))/np.sqrt(hwhm)

    return hwhm
    


def  project(mat, v):
    #RMS shape in neutronpy is (n,4,4) different from the matlab Reslib34.
    #thus, we have some changes here in order to keep this 
    if v == 2:
        i,j=0,1
    elif v == 0:
        i,j=1,2
    elif v == 1:
        i,j=0,2
    else:
        print("Error: improper i value!, it should be 0 or 1 or 2")
        return
    if not isinstance(mat, np.ndarray):
        mat=np.array(mat)
    if mat.ndim<3:
        mat=np.reshape(mat, [-1, mat.shape[0],mat.shape[1]])
    #print(mat.shape)   
    [a,b,c]=mat.shape
    #print(a, b, c)
    #print(i, j, v)
    #print(mat[i,i,:])
    #print(mat[v,v,:])
    proj=np.zeros([a,2,2])
    sec =np.zeros([a,2,2])
    proj[:,0,0] = mat[:,i,i] - mat[:,i,v]**2/mat[:,v,v]
    proj[:,0,1] = mat[:,i,j] - mat[:,i,v]*mat[:,j,v]/mat[:,v,v]
    proj[:,1,0] = mat[:,j,i] - mat[:,j,v]*mat[:,i,v]/mat[:,v,v]
    proj[:,1,1] = mat[:,j,j] - mat[:,j,v]**2/mat[:,v,v]
    sec[:,0,0]  = mat[:,i,i]
    sec[:,0,1]  = mat[:,i,j]
    sec[:,1,0]  = mat[:,j,i]
    sec[:,1,1]  = mat[:,j,j]

    return [proj, sec]


def PlotEllipse(ax,mat,x0,y0,style):
    
    if ax is None:
        print("ax is not valid")
        fig=plt.figure()
        ax=fig.add_subplot(111)
        
    [a,b,c]=mat.shape
    
    phi=np.linspace(0, 2*np.pi,1000)
    for i in range(a):
        r = np.sqrt(2*np.log(2)/(mat[i,0,0]*np.cos(phi)**2+mat[i,1,1]*np.sin(phi)**2+2*mat[i,0,1]*np.cos(phi)*np.sin(phi)))
        x = r*np.cos(phi)+x0[i]
        y = r*np.sin(phi)+y0[i]

        ax.plot(x,y,style)
        ax.autoscale()
        #plt.show()


def ProduceEllipse(mat,x0,y0):
            
    [a,b,c]=mat.shape
    r=np.zeros(shape=(1000, a))
    x=np.zeros(shape=(1000, a))
    y=np.zeros(shape=(1000, a))
    
    phi=np.linspace(0, 2*np.pi,1000)
    for i in range(a):
        r[:,i] = np.sqrt(2*np.log(2)/(mat[i,0,0]*np.cos(phi)**2+mat[i,1,1]*np.sin(phi)**2+2*mat[i,0,1]*np.cos(phi)*np.sin(phi)))
        x[:,i]  = r[:,i]*np.cos(phi)+x0[i]
        y[:,i]  = r[:,i]*np.sin(phi)+y0[i]
    #print("shape of x:{}".format(x.shape))
    return [r, x, y]