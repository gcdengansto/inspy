# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

# Material constructor

import inspy.constants as const
import numpy as np

from .atom import Atom, MagneticAtom
from .sample import Sample
from .structure_factors import MagneticStructureFactor, NuclearStructureFactor
from .symmetry import SpaceGroup

# from ..scattering.pattern import HKLGenerator


class MagneticUnitCell(Sample):
    # defining a magnetic unit cell

    def __init__(self, unit_cell):
        if 'chirality' not in unit_cell:
            self.chirality = 0
        if 'ph' not in unit_cell:
            self.phase = 0
        self.atoms = []
        if 'space_group' in unit_cell:
            self.space_group = SpaceGroup(unit_cell['space_group'])
            for atom in unit_cell['atoms']:
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(MagneticAtom(atom['ion'],
                                                   pos,
                                                   atom['moment'],
                                                   atom['occupancy']))
        else:
            for atom in unit_cell['atoms']:
                self.atoms.append(MagneticAtom(atom['ion'],
                                               atom['pos'],
                                               atom['moment'],
                                               atom['occupancy']))

        self.propagation_vector = unit_cell['propagation_vector']

        a, b, c = unit_cell['lattice']['abc']
        alpha, beta, gamma = unit_cell['lattice']['abg']
        super(MagneticUnitCell, self).__init__(a, b, c, alpha, beta, gamma)

    def __repr__(self):
        return "MagneticUnitCell('{0}')".format(self.propagation_vector)


class Material(Sample, NuclearStructureFactor, MagneticStructureFactor):
    #Class for the Material being supplied for the structure factor calculation


    def __init__(self, crystal):
        self.name = crystal['name']

        if 'formulaUnits' not in crystal:
            crystal['formulaUnits'] = 1.

        self.muCell = 0.
        for item in crystal['composition']:
            if 'occupancy' not in item:
                item['occupancy'] = 1.
            self.muCell += const.periodic_table()[item['ion']]['mass'] * item['occupancy']

        self.Mcell = self.muCell * crystal['formulaUnits']

        if 'lattice' in crystal:
            a, b, c = crystal['lattice']['abc']
            alpha, beta, gamma = crystal['lattice']['abg']

        if 'wavelength' in crystal:
            self.wavelength = crystal['wavelength']
        else:
            self.wavelength = 2.359

        if 'space_group' in crystal:
            self.space_group = SpaceGroup(crystal['space_group'])
            self.atoms = []
            for atom in crystal['composition']:
                if 'Uiso' not in atom:
                    atom['Uiso'] = 0
                if 'Uaniso' not in item:
                    atom['Uaniso'] = np.matrix(np.zeros((3, 3)))
                if 'occupancy' not in item:
                    atom['occupancy'] = 1.
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(Atom(atom['ion'],
                                           pos,
                                           atom['occupancy'],
                                           self.Mcell,
                                           crystal['massNorm'],
                                           atom['Uiso'],
                                           atom['Uaniso']))
        else:
            self.atoms = []
            for item in crystal['composition']:
                if 'Uiso' not in item:
                    item['Uiso'] = 0
                if 'Uaniso' not in item:
                    item['Uaniso'] = np.matrix(np.zeros((3, 3)))
                if 'occupancy' not in item:
                    item['occupancy'] = 1.
                self.atoms.append(Atom(item['ion'],
                                       item['pos'],
                                       item['occupancy'],
                                       self.Mcell,
                                       crystal['massNorm'],
                                       item['Uiso'],
                                       item['Uaniso']))

        if 'magnetic_unit_cell' in crystal:
            self.magnetic_unit_cell = MagneticUnitCell(crystal['magnetic_cell'])

        if 'mosaic' not in crystal:
            crystal['mosaic'] = None
        if 'vmosaic' not in crystal:
            crystal['vmosaic'] = None
        if 'u' not in crystal:
            crystal['u'] = None
        if 'v' not in crystal:
            crystal['v'] = None
        if 'dir' not in crystal:
            crystal['dir'] = 1

        super(Material, self).__init__(a, b, c, alpha, beta, gamma, crystal['mosaic'], crystal['vmosaic'],
                                       crystal['dir'], crystal['u'], crystal['v'])

    def __repr__(self):
        return "Material('{0}')".format(self.name)

    @property
    def total_scattering_cross_section(self):
        r"""Returns total scattering cross-section of unit cell
        """
        total = 0
        for atom in self.atoms:
            total += (atom.coh_xs + atom.inc_xs)
        return total

    def N_atoms(self, mass):
        #Number of atoms in the defined Material, given the mass of the sample.

        return const.N_A * mass / self.muCell

    def calc_optimal_thickness(self, energy=25.3, transmission=1 / np.exp(1)):
        #Calculates the optimal sample thickess to avoid problems with
        #extinction, multiple coherent scattering and absorption.


        sigma_coh = np.sum([atom.occupancy * atom.coh_xs for atom in self.atoms])
        sigma_inc = np.sum([atom.occupancy * atom.inc_xs for atom in self.atoms])
        sigma_abs = np.sum([atom.occupancy * atom.abs_xs for atom in self.atoms])

        sigma_T = (sigma_coh + sigma_inc + sigma_abs * np.sqrt(25.3 / energy)) / self.volume

        return -np.log(transmission) / sigma_T

    def calc_incoh_elas_xs(self, mass=None):
        #Calculates the incoherent elastic cross section.


        INC_XS = 0
        for atom in self.atoms:
            INC_XS += atom.inc_xs * np.exp(-8 * np.pi ** 2 * atom.Uiso * np.sin(
                np.deg2rad(self.get_two_theta(atom.pos, self.wavelength) / 2.)) ** 2 / self.wavelength ** 2)

        if mass is not None:
            return self.N_atoms(mass) / (4 * np.pi) * INC_XS
        else:
            return INC_XS / (4 * np.pi)
