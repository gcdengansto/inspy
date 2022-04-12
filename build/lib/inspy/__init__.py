r"""NeutronPy: open source python library for neutron scattering data analysis

"""
from __future__ import absolute_import
import sys
import warnings
import pkg_resources

from . import constants
from . import fileio
from . import functions
from . import instrument
from . import models
from . import spurion
from . import scattering
from .crystal import Lattice
from .crystal import Material
from .crystal import Sample
from .crystal import symmetry
from .data import Data
from .energy import Energy
from .instrument import TripleAxisSpectr, TimeOfFlightSpectr, tools
from .lsfit import Fitter, FitConv
try:
    from . import gui, main_gui
except ImportError:
    warnings.warn('PyQt5 not found, cannot run Resolution GUI')

__version__ = pkg_resources.require("inspy")[0].version

if sys.version_info[:2] == (2, 6) or sys.version_info[:2] == (3, 3):
    warnings.warn('Support for Python 2.6 and Python 3.3 is depreciated and will be dropped in inspy 0.1.0',
                  DeprecationWarning)
