# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

# Handles lattice geometries to find rotations and transformations

import numpy as np

class Lattice(object):
    #Class to describe a generic lattice system defined by lattice six parameters.


    def __init__(self, a, b, c, alpha, beta, gamma):
        self.abc = [a, b, c]
        self.abg = [alpha, beta, gamma]

    def __repr__(self):
        return "Lattice({0}, {1}, {2}, {3}, {4}, {5})".format(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    @property
    def a(self):
 
        return self.abc[0]

    @a.setter
    def a(self, a):
        self.abc[0] = a

    @property
    def b(self):

        return self.abc[1]

    @b.setter
    def b(self, b):
        self.abc[1] = b

    @property
    def c(self):

        return self.abc[2]

    @c.setter
    def c(self, c):
        self.abc[2] = c

    @property
    def alpha(self):

        return self.abg[0]

    @alpha.setter
    def alpha(self, alpha):
        self.abg[0] = alpha

    @property
    def beta(self):

        return self.abg[1]

    @beta.setter
    def beta(self, beta):
        self.abg[1] = beta

    @property
    def gamma(self):

        return self.abg[2]

    @gamma.setter
    def gamma(self, gamma):
        self.abg[2] = gamma

    @property
    def alpha_rad(self):

        return self.abg_rad[0]

    @property
    def beta_rad(self):

        return self.abg_rad[1]

    @property
    def gamma_rad(self):

        return self.abg_rad[2]

    @property
    def astar(self):

        return self.b * self.c * np.sin(self.alpha_rad) / self.volume * 2 * np.pi

    @property
    def bstar(self):

        return self.a * self.c * np.sin(self.beta_rad) / self.volume * 2 * np.pi

    @property
    def cstar(self):

        return self.a * self.b * np.sin(self.gamma_rad) / self.volume * 2 * np.pi

    @property
    def alphastar_rad(self):

        return np.arccos((np.cos(self.beta_rad) * np.cos(self.gamma_rad) -
                          np.cos(self.alpha_rad)) /
                         (np.sin(self.beta_rad) * np.sin(self.gamma_rad)))

    @property
    def betastar_rad(self):

        return np.arccos((np.cos(self.alpha_rad) *
                          np.cos(self.gamma_rad) -
                          np.cos(self.beta_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.gamma_rad)))

    @property
    def gammastar_rad(self):

        return np.arccos((np.cos(self.alpha_rad) * np.cos(self.beta_rad) -
                          np.cos(self.gamma_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.beta_rad)))

    @property
    def alphastar(self):

        return np.rad2deg(self.alphastar_rad)

    @property
    def betastar(self):

        return np.rad2deg(self.betastar_rad)

    @property
    def gammastar(self):

        return np.rad2deg(self.gammastar_rad)

    @property
    def reciprocal_abc(self):

        return [self.astar, self.bstar, self.cstar]

    @property
    def reciprocal_abg(self):

        return [self.alphastar, self.betastar, self.gammastar]

    @property
    def reciprocal_abg_rad(self):

        return [self.alphastar_rad, self.betastar_rad, self.gammastar_rad]

    @property
    def abg_rad(self):

        return np.deg2rad(self.abg)

    @property
    def lattice_type(self):

        if len(np.unique(self.abc)) == 3 and len(np.unique(self.abg)) == 3:
            return 'triclinic'
        elif len(np.unique(self.abc)) == 3 and self.abg[1] != 90 and np.all(np.array(self.abg)[:3:2] == 90):
            return 'monoclinic'
        elif len(np.unique(self.abc)) == 3 and np.all(np.array(self.abg) == 90):
            return 'orthorhombic'
        elif len(np.unique(self.abc)) == 1 and len(np.unique(self.abg)) == 1 and np.all(
                        np.array(self.abg) < 120) and np.all(np.array(self.abg) != 90):
            return 'rhombohedral'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg) == 90):
            return 'tetragonal'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg)[0:2] == 90) and \
                        self.abg[2] == 120:
            return 'hexagonal'
        elif len(np.unique(self.abc)) == 1 and np.all(np.array(self.abg) == 90):
            return 'cubic'
        else:
            raise ValueError('Provided lattice constants and angles do not resolve to a valid Bravais lattice')

    @property
    def volume(self):

        return np.sqrt(np.linalg.det(self.G))

    @property
    def reciprocal_volume(self):

        return np.sqrt(np.linalg.det(self.Gstar))

    @property
    def G(self):
        #Metric tensor of the real space lattice


        a, b, c = self.abc
        alpha, beta, gamma = self.abg_rad

        return np.matrix([[a ** 2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                          [a * b * np.cos(gamma), b ** 2, b * c * np.cos(alpha)],
                          [a * c * np.cos(beta), b * c * np.cos(alpha), c ** 2]], dtype=float)

    @property
    def Gstar(self):
        #Metric tensor of the reciprocal lattice

        return np.linalg.inv(self.G) * 4 * np.pi ** 2

    @property
    def Bmatrix(self):
        #Cartesian basis matrix in reciprocal units such that Bmatrix*Bmatrix.T = Gstar


        return np.matrix([[self.astar, self.bstar * np.cos(self.gammastar_rad),  self.cstar * np.cos(self.betastar_rad)],
                          [0, self.bstar * np.sin(self.gammastar_rad), -self.cstar * np.sin(self.betastar_rad) * np.cos(self.alpha_rad)],
                          [0, 0, self.cstar * np.sin(self.betastar_rad) * np.sin(self.alphastar_rad)]], dtype=float)

    def get_d_spacing(self, hkl):
        #Returns the d-spacing of a given reciprocal lattice vector.

        hkl = np.array(hkl)

        return float(1 / np.sqrt(np.dot(np.dot(hkl, self.Gstar / 4 / np.pi ** 2), hkl)))

    def get_angle_between_planes(self, v1, v2):
        #Returns the angle :math:`\phi` between two reciprocal lattice vectors

        v1, v2 = np.array(v1), np.array(v2)

        return float(np.rad2deg(np.arccos(np.inner(np.inner(v1, self.Gstar), v2) /
                                          np.sqrt(np.inner(np.inner(v1, self.Gstar), v1)) /
                                          np.sqrt(np.inner(np.inner(v2, self.Gstar), v2)))))

    def get_two_theta(self, hkl, wavelength):
        #Returns the detector angle  for a given reciprocal lattice vector and incident wavelength.

        return 2 * np.rad2deg(np.arcsin(wavelength / 2 /
                                        self.get_d_spacing(hkl)))

    def get_q(self, hkl):
        #Returns the magnitude of *Q* for a given reciprocal lattice
 
        return 2 * np.pi / self.get_d_spacing(hkl)
