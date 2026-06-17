# Author: Alex Fox
# Created: 2026-06-17
"""
Utility functions for post-processing eddy flux data.
"""

import pandas as pd
import numpy as np
import solarpy
from tqdm import tqdm

def compute_isday(timestamps: pd.DatetimeIndex, lat: float, lon: float, elev: float=0, sw_thresh: float=20) -> pd.Series:
    """
    Compute whether or not it's daytime based on theoretical insolation

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        the timestamps representing each datapoint in your timeseries
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
    timestamps = timestamps.sort_values()

    panel = solarpy.solar_panel(1, 1, id_name="noname")  # surface, efficiency and name
    panel.set_orientation(np.array([0, 0, -1]))  # upwards
    panel.set_position(lat, lon, elev)
    dates = pd.date_range("2019-01-01", "2019-12-31 23:59:59", freq=timestamps[1] - timestamps[0])
    powers = []
    for d in tqdm(dates):
        panel.set_datetime(d.to_pydatetime())
        powers.append(panel.power())
    powers = pd.DataFrame({"SW_IN_POT": powers, "doy": dates.dayofyear, "h": dates.hour, "m":dates.minute})

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
    for d in tqdm(dates):
        panel.set_datetime(d.to_pydatetime())
        df.loc[d, "SW_IN_POT"] = panel.power()

    isday = df["SW_IN_POT"] > sw_thresh
    return isday.astype(bool).loc[timestamps]