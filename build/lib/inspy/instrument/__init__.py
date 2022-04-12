r"""Instrument resolution calculations

"""

from .mono import Mono
from .ana  import Ana
from .components import Chopper, Detector, Guide
from .tas_spectr import TripleAxisSpectr, SimpleComp
from .tof_spectr import TimeOfFlightSpectr
from .tools import get_tau, fproject, _cleanargs, _ellipse, _modvec, _scalar, _star, get_angle_ki_Q, get_bragg_widths, get_phonon_width, get_kfree, chop, calc_proj_hwhm, project_into_plane
from . import tools
from . import exceptions
