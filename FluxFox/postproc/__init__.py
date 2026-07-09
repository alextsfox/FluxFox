# Author: Alex Fox
# Created: 2024-06-17
"""
Post-processing functions for eddy flux data.

Includes modules for
- U* filtering (`ustar`)
- Gap-filling (`gapfill`)
- Despiking (`despike`)
- Partitioning (`partition`)
- Footprint estimation (`footprint`) # TODO
- Plotting (`plot`)
- Other utility functions, such as day/night detection (`utils`)

A typical workflow involves 
1. Despiking
2. U* filtering
3. Gap-filling
4. Partitioning
5. Plotting 
"""

from .utils import compute_isday
from .ustar import ustar_papale_2006
from .gapfill import gapfill_xgb_liu_2025, gapfill_mds_reichstein_2005
from .despike import despike_mad_papale_2006
from .partition import gpp_night_reichstein_2005, gpp_night_falge_2001, gpp_day_lasslop_2010
from .plot import plot_fingerprint
    
__all__ = [
    "ustar_papale_2006", 
    "gapfill_xgb_liu_2025", "gapfill_mds_reichstein_2005", 
    "despike_mad_papale_2006", 
    "gpp_night_reichstein_2005", "gpp_night_falge_2001", "gpp_day_lasslop_2010",
    "plot_fingerprint",
    "compute_isday"
]