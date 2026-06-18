# Author: Alex Fox
# Created: 2026-06-17

"""
Partitioning utilities
"""

from typing import Any

import pandas as pd
import numpy as np
from scipy import optimize

from .utils import compute_isday

def _sum_abs_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred))

def _lloyd_taylor_1994(T: np.ndarray, R_ref: float, E0: float) -> np.ndarray:
    """
    Lloyd and Taylor (1994) model of ecosystem respiration as a function of temperature.
    """
    T_ref = 10 + 273.15
    T0 = -46.02 + 273.15
    return R_ref * np.exp(E0 * (1. / (T_ref - T0) - 1. / (T - T0)))

def gpp_falge_2001(
    df: pd.DataFrame,
    nee_col: str,
    t_col: str,
    lat: float, lon: float, elev: float = 0,
    sw_thresh: float = 20,
    **curve_fit_kwargs,
)-> tuple[pd.DataFrame, Any]:
    """
    Partitioning of NEE into GPP and Reco based on 
    Falge et al. (2001). "Gap filling strategies for defensible annual sums of net ecosystem exchange," Agricultural and Forest Meteorology

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NEE data to be partitioned, along with air temperature.
        Must have a DatetimeIndex.
    nee_col: str
        Name of the column containing NEE data.
    t_col: str
        Name of the column containing air temperature (or soil temperature) data.
    lat : float
        Latitude of the site.
    lon : float
        Longitude of the site.
    elev : float, optional
        Elevation of the site in meters. Default is 0.
    sw_thresh : float, optional
        Threshold for incoming shortwave radiation to classify daytime. Default is 20.
    **curve_fit_kwargs : 
        Additional keyword arguments to pass to scipy.optimize.curve_fit (in addition to func, x_data, y_data, and p0).
    
    Returns
    -------
    tuple[pd.DataFrame, Any]
        DataFrame containing partitioned GPP and Reco, and the optimization result (from scipy.optimize.curve_fit)
    """
    
    # Classify into daytime and nighttime
    isday = compute_isday(df, lat=lat, lon=lon, elev=elev, sw_thresh=sw_thresh)

    # fit nighttime FC as a nonlinear function of air or soil temperature.
    # Typically uses a Lloyd-Taylor model with a temp sensitivity parameter and a reference respiration parameter

    # Estimate seasonal variation in respiration parameters
    # Allow the respiration normalization parameter, something like R_ref to vary through time, while keeping the temperature sensitivity stable or globally contrained.
    # The Reddyproc/Reichstein workflow estimates temperature sensitivity first, then estimates the seasonal course of reference respiration with sliding windows.

    # Predict ecosystem respiration for both daytime and nigthttime records, including periods with missing NEE (but flag for low-quality)

    # For daytime gap-filled NEE, optinally use a light-response regression


    # select nighttime records
    # TODO: add jacobian
    # TODO: add bounds
    # TODO: allow user to provide uncertainty for FC (e.g., standard deviation)
    nighttime_FC = df.loc[~isday, [nee_col, t_col]].dropna()
    res = optimize.curve_fit(
        _lloyd_taylor_1994, 
        x_data=nighttime_FC[t_col], y_data=nighttime_FC[nee_col],
        p0=[2.0, 200.0], 
        **curve_fit_kwargs,
    )
    R_ref, E0 = res[0]
    
    df_out = pd.DataFrame(index=df.index)
    df_out['Reco'] = _lloyd_taylor_1994(df[t_col].values, R_ref=R_ref, E0=E0)
    df_out['GPP'] = -df[nee_col] + df_out['Reco']

    # cap to non-negative values
    df_out['GPP'] = df_out['GPP'].clip(lower=0)
    df_out['Reco'] = df_out['Reco'].clip(lower=0)
    
    return df_out, res
    


