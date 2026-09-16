"""
Miscellaneous utility functions for FluxFox.
This module is intended as a grab-bag of useful functions that don't fit neatly into other modules.
"""

import pandas as pd
import numpy as np
import solarpy

def compute_WY_DOWY(timestamps:pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute the water year and day of water year for a given set of timestamps.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        The timestamps for which to compute the water year and day of water year.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: 'WY' for water year and 'DOWY' for day of water year.
    """

    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("timestamps must be a pandas DatetimeIndex")

    wy = np.where(timestamps.month.isin(range(1, 10)), timestamps.year, timestamps.year + 1)
    dowy = (timestamps - pd.to_datetime((wy - 1).astype(str) + "-10-01")).days + 1

    return pd.DataFrame({"WY": wy, "DOWY": dowy}, index=timestamps, dtype=int)

def compute_meridian_offset_timezone(lon: float) -> str:
    """
    Compute the meridian offset timezone for a given longitude.

    Parameters
    ----------
    lon : float
        The longitude in decimal degrees. Positive for east, negative for west.

    Returns
    -------
    str
    """

    if lon < -180 or lon >= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    return f"UTC{int(np.round(lon / 15)):+d}"

def compute_isday(timestamps: pd.DatetimeIndex, lat: float, lon: float, elev: float=0, sw_thresh: float=20) -> pd.Series:
    """
    Compute whether or not it's daytime based on theoretical insolation at a given latitude, longitude, and elevation.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        the timestamps representing each datapoint in your timeseries. 
        This measurement should reflect the meridian offset timezone, not necessarily local time. 
        For example, Anchorage, AK, USA has a meridian offset timezone of UTC-10, but the local time is UTC-9 (-8 during daylight savings).
        If you are unsure of the meridian offset timezone, you can compute it using `compute_meridian_offset_timezone()`.
    lat : float
        the latitude of the site (decimal degrees)
    lon : float
        the longitude of the site (decimal degrees)
    elev : float
        the elevation above sea level of the site (meters). Default 0m.
    sw_thresh : float
        the threshold for theoretical incoming shortwave radiation (W m-2) on a horizontal surface, below which it is considered nighttime. Default 20 W m-2.
        
    Returns
    -------
    isday : pd.Series
        a pandas series of type `bool`, indexed by `timestamps`. `True` if daytime, `False` if nighttime.
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("timestamps must be a pandas DatetimeIndex")
    timestamps = timestamps.sort_values()
    dt = timestamps.diff().median()

    # compute theoretical incoming shortwave radiation on a horizontal surface using solarpy
    # do this for 365 days of a year.
    panel = solarpy.solar_panel(1, 1, id_name="noname")  # surface, efficiency and name
    panel.set_orientation(np.array([0, 0, -1]))  # upwards
    panel.set_position(lat, lon, elev)
    dates = pd.date_range("2019-01-01", "2019-12-31 23:59:59", freq=dt)
    powers = []
    for d in dates:
        panel.set_datetime(d.to_pydatetime())
        powers.append(panel.power())
    powers = pd.DataFrame({"SW_IN_POT": powers, "doy": dates.dayofyear, "h": dates.hour, "m":dates.minute})

    # merge the theoretical powers with the timestamps
    if timestamps.name is None:
        timestamps = timestamps.rename("TIMESTAMP")
    df = pd.DataFrame({'doy':timestamps.dayofyear, 'h':timestamps.hour, 'm':timestamps.minute})
    df = df.set_index(timestamps)
    df = (
        df
        .reset_index(names=timestamps.name)
        .merge(powers, on=['doy', 'h', 'm'], how='left')
        .drop(columns=["doy", "h", "m"])
        .set_index(timestamps.name)
        .reindex(timestamps)
    )

    # leap days
    dates = df.index[df["SW_IN_POT"].isna()]
    for d in dates:
        if ~pd.isna(d):
            panel.set_datetime(d.to_pydatetime())
            df.loc[d, "SW_IN_POT"] = panel.power()

    isday = df["SW_IN_POT"] > sw_thresh

    return isday.astype(bool).loc[timestamps]