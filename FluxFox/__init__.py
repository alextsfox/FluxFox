# Author: Alex Fox
# Created: 2024-06-17
"""
FluxFox eddy covariance science package

Includes modules for
- AmeriFlux data handling (`ameriflux`)
- Post-processing (`postproc`)
- Prep-processing (`prepproc`)  # TODO
"""

from . import ameriflux
from . import thermo
from . import postproc
from . import utils

__all__ = ["ameriflux", "thermo", "postproc", "utils"]