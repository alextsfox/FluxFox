"""
Functions for compensating time lags in high frequency data.
"""

from scipy import signal
import pandas as pd
import numpy as np

def timelag_nominal(obs: pd.Series, n_wind: float, e_wind: float, z_wind: float, n_sep: float, e_sep: float, z_sep: float, acq_freq: float, use_vertical : bool = False, min_ws : float = 0.1) -> tuple[pd.Series, int]:
    """
    Compute the nominal time lag between two time series from 2 instruments based on the wind speed and separation distance between the instruments.

    This method is not recommended as the primary method for computing time lags, especially for closed path instruments, but can be used as a first guess or to check the plausibility of other methods.

    Parameters
    ----------
    obs : pd.Series
        The observed time series: this one will be shifted to maximize correlation with the reference series.
    n_wind : float
        The northerly wind component (m/s) (wind originating in the north).
    e_wind : float
        The easterly wind component (m/s) (wind originating in the east).
    z_wind : float
        The vertical wind component (m/s) (wind originating from above).
    n_sep : float
        The north-south separation distance between the two instruments (m). Positive if the observed instrument is north of the reference instrument.
    e_sep : float
        The east-west separation distance between the two instruments (m). Positive if the observed instrument is east of the reference instrument.
    z_sep : float
        The vertical separation distance between the two instruments (m). Positive if the observed instrument is above the reference instrument.
    acq_freq : float
        The acquisition frequency of the time series (Hz).
    use_vertical : bool, optional
        Whether to include the vertical wind component in the time lag calculation. Default is False.
    min_ws : float, optional
        The minimum wind speed (m/s) below which the time lag is considered unreliable and will return a lag of 0. Default is 0.1 m/s.

    Returns
    -------
    obs_shifted : pd.Series
        The observed time series shifted by the computed time lag.
    lag : int
        The computed time lag (in number of samples) based on the wind speed and separation distances. A positive lag indicates that the observed series lags behind the reference series, while a negative lag indicates that the observed series leads the reference series.
    """

    acq_period = 1 / acq_freq

    if use_vertical:
        wind_sq = n_wind**2 + e_wind**2 + z_wind**2
        dot = n_sep*n_wind + e_sep*e_wind + z_sep*z_wind
    else:
        wind_sq = n_wind**2 + e_wind**2
        dot = n_sep*n_wind + e_sep*e_wind

    speed = np.sqrt(wind_sq)
    if speed < min_ws:
        lag = 0
    else:
        tau = -dot / wind_sq
        lag = int(round(tau / acq_period))

    obs_shifted = obs.shift(-lag)
    return obs_shifted, lag

def timelag_covmax(obs: pd.Series, ref: pd.Series, default_lag: int = 0, max_lag: int | None = None, max_gap: int | None = None) -> tuple[pd.Series, int]:
    """
    Computes the time lag between two time series using cross-correlation.

    Parameters
    ----------
    obs : pd.Series
        The observed time series: this one will be shifted to maximize correlation with the reference series.
    ref : pd.Series
        The reference time series.
    default_lag : int, optional
        The default lag (in number of samples) to return if the cross-correlation cannot be computed or if max lag is exceeded. default is 0.
        Negative values indicate that the observed series leads the reference series, while positive values indicate that the observed series lags behind the reference series.
    max_lag : int, optional
        The maximum lag (in number of samples) to consider for the cross-correlation. If None, the maximum lag will be set to half the length of the shorter series.
    max_gap : int, optional
        The maximum number of consecutive NaN values to interpolate over before computing the cross-correlation. If None, set no limit in interpolation. Set to 0 to not interpolate at all: this will cause the function to return a lag of 0 if there are any NaN values in the input series.
    
    Returns
    -------
    obs_shifted : pd.Series
        The observed time series shifted by the computed time lag.
    lag : int
        The computed time lag (in number of samples) that maximizes the cross-correlation between the observed and reference series. A positive lag indicates that the observed series lags behind the reference series, while a negative lag indicates that the observed series leads the reference series.
    """
    if not isinstance(obs, pd.Series) or not isinstance(ref, pd.Series):
        raise TypeError("Both obs and ref must be pandas Series.")

    if max_lag is None:
        max_lag = min(obs.shape[0], ref.shape[0]) // 2

    try:
        correlation = signal.correlate(
            obs.interpolate(limit=max_gap, limit_area="inside", method="linear").to_numpy(), 
            ref.interpolate(limit=max_gap, limit_area="inside", method="linear").to_numpy(), 
            mode='full'
        )
        lags = signal.correlation_lags(len(obs), len(ref), mode='full')
        lag = int(lags[np.argmax(correlation)])
        if abs(lag) > max_lag:
            lag = default_lag
            # TODO: log that max lag was exceeded
    except ValueError:
        lag = default_lag
        # TODO: log that cross-correlation could not be computed
    obs_shifted = obs.shift(-lag)
    return obs_shifted, lag

def timelag_auto_opt_fratini():
    """
    Placeholder for future implementation of the eddypro automatic timelag optimization method, likely attributable to Fratini.

    https://www.licor.com/support/EddyPro/topics/time-lag-detect-correct.html

    Method, roughly:
    1. For each time period, compute the covmax timelag to get a first guess of the time lag for each period.
    2. In a moving window (of a month or more), compute the median timelag and tau_med, and a plausibility window of t_nom ± z*MAD(tau)/0.6745, where z is the z-score (default 1.5)
    3. When tau_i exceeds the plausibility window, replace it with tau_med.
    4. Apply the timelag to the data.

    For water vapor, the time lag depends on humidity, which is especially important for closed path systems. The same system applies, except
    the data is split into humidity bins (by quantile? by absolute value?) and the timelag is computed for each bin separately. The plausibility window is then computed for each bin separately, and the timelag is applied to the data in each bin separately.

    Follow the design pattern of RotatePlanarFit.
    """

__all__ = ["timelag_covmax", "timelag_nominal"]