#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)
# Version number
__version__ = "0.01"

__all__ = ["cross_correlation", "manipulate_spectra", "orbital_properties",
           "photon_counts", "pipeline", "remove_tellurics", "retrieve_sky"]


# Was coronagraph imported from setup.py?
try:
    __HRT_SETUP__
except NameError:
    __HRT_SETUP__ = False

if not __HRT_SETUP__:
    # This is a regular coronagraph run
    from . import *
    from .cross_correlation import *
    from .manipulate_spectra import *
    from .orbital_properties import *
    from .photon_counts import *
    from .pipeline import *
    from .remove_tellurics import *
    from .retrieve_sky import *
