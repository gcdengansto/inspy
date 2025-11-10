# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:27:29 2020
@author: dgc
"""


from __future__ import absolute_import
import sys
import warnings
import pkg_resources

from . import constants
from . import instrument
from .crystal import Lattice
from .crystal import Material
from .crystal import Sample
from .crystal import symmetry
#from .data import Data
from .energy import Energy
from .instrument import TripleAxisSpectr, TimeOfFlightSpectr, tools
from .insfit import FitConv
try:
    from .gui.main_gui import main
except ImportError:
    warnings.warn('PyQt5 not found, cannot run Resolution GUI')

__version__ = pkg_resources.require("inspy")[0].version

if sys.version_info[:2] == (2, 6) or sys.version_info[:2] == (3, 3):
    warnings.warn('Support for Python 2.6 and Python 3.3 is depreciated and will be dropped in inspy 0.1.0',
                  DeprecationWarning)
