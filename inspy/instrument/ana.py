# -*- coding: utf-8 -*-
from .mono import Mono


class Ana(Mono):
    u"""Class containing analyzer information.

    Parameters
    ----------
    tau : float or str
        Tau value for the analyzer

    mosaic : int
        Mosaic of the analyzer in arc minutes

    direct : ±1, optional
        Direction of the analyzer (left or right, -1 or +1, respectively).
        Default: -1 (left-handed coordinate frame).

    vmosaic : int, optional
        Vertical mosaic of the analyzer in arc minutes. Default: None

    height : float, optional
        Height of the analyzer in cm. Default: None

    width : float, optional
        Width of the analyzer in cm. Default: None

    depth : float, optional
        Depth of the analyzer in cm. Default: None

    horifoc : int, optional
        Set to 1 if horizontally focusing analyzer is used. Default: -1

    thickness : float, optional
        Thickness of Analyzer crystal in cm. Required for analyzer
        reflectivity calculation. Default: None

    Q : float, optional
        Kinematic reflectivity coefficient. Required for analyzer
        reflectivity calculation. Default: None

    """
    
    u"""A structure that describes the analyzer and contains fields as in
        :attr:`mono` plus optional fields.

        Attributes
        ----------
        thickness: float
            The analyzer thickness in cm for ideal-crystal reflectivity
            corrections (Section II C 3). If no reflectivity corrections are to
            be made, this field should remain unassigned or set to a negative
            value.

        Q : float
            The kinematic reflectivity coefficient for this correction. It is
            given by

            .. math::    Q = \\frac{4|F|**2}{V_0} \\frac{(2\\pi)**3}{\\tau**3},

            where V0 is the unit cell volume for the analyzer crystal, F is the
            structure factor of the analyzer reflection, and τ is the analyzer
            reciprocal lattice vector. For PG(002) Q = 0.1287. Leave this field
            unassigned or make it negative if you don’t want the correction
            done.

        horifoc : bool
            A flag that is set to 1 if a horizontally focusing analyzer is used
            (Section II D). In this case ``hcol[2]`` (see below) is the angular
            size of the analyzer, as seen from the sample position. If the
            field is unassigned or equal to -1, a flat analyzer is assumed.
            Note that this option is only available with the Cooper-Nathans
            method.

        dir : int
            Direction of the crystal (left or right, -1 or +1, respectively).
            Default: -1 (left-handed coordinate frame).

        rh : float
            Horizontal curvature of the analyzer in cm.

        rv : float
            Vertical curvature of the analyzer in cm.

        """

    def __init__(self, tau, mosaic, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None,
                 horifoc=-1, thickness=None,  direct=-1, Q=None):
        super(Ana, self).__init__(tau=tau, mosaic=mosaic, vmosaic=vmosaic, height=height, width=width, depth=depth, rh=rh, rv=rv, direct=direct)
        if thickness is not None:
            self.thickness = thickness
        if Q is not None:
            self.Q = Q
        self.horifoc = horifoc

    def __repr__(self):
        args = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in ['tau', 'mosaic']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            [ 'vmosaic', 'height', 'width', 'depth', 'rh', 'rv', 'horifoc', 'thickness', 'direct','Q']
                            if getattr(self, key, None) is not None])
        return "Analyzer({0})".format(', '.join([args, kwargs]))
