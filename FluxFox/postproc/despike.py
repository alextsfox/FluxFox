# Author: Alex Fox
# Created: 2026-06-17
"""
Postprocessing despiking methods
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import compute_isday

def mad_despike_papale_2006(
    df: pd.DataFrame,
    cols_to_despike: list[str],
    lat: float, lon: float, elev: float=0,
    z: float=4,
    nighttime_swin: float=20,
    chunksize_days: int=13,
)->pd.Series:
    # 1. split into day/night
    isday = compute_isday(df.index, lat, lon, elev, nighttime_swin)
    # add twilight_paddingtime periods of padding to blend day and night to avoid abrupt transitions
    isnight = ~isday
    sunset_indices = np.where((isday.values[:-1]) & (~isday.values[1:]))[0]
    sunrise_indices = np.where((~isday.values[:-1]) & (isday.values[1:]))[0]
    isday.iloc[sunset_indices + 1] = True
    isday.iloc[sunrise_indices] = True
    isnight.iloc[sunset_indices] = True
    isnight.iloc[sunrise_indices + 1] = True

    night_df = df.loc[isnight, cols_to_despike]
    day_df = df.loc[isday, cols_to_despike]

    spike_daynight = []
    for daysplit_df in (night_df, day_df):
        # assume all points are spikes initially
        spikes = pd.DataFrame(True, index=df.index, columns=cols_to_despike)
        # 2. chunk into chunksize_days blocks
        for chunk_start, chunk_group in daysplit_df.groupby(pd.Grouper(freq=f'{chunksize_days}D')):
            # 3. Compute double-difference (d = (X_i-X_i-1) - (X_i+1-X_i))
            double_diff = chunk_group.diff() - chunk_group.shift(-1).diff()
            
            # 4. Compute MAD within block: MAD = median(|d_i - median(d)|)
            mad = (double_diff - double_diff.median()).abs().median()

            # 5. Flag outside z-range
            max_deviation = z * mad / 0.6745
            # set points within z-range as non-spikes
            # this seems redundant, but note that NA values will always return False in comparisons
            # so they will be considered spikes this way
            chunk_non_spikes = (
                (double_diff >= double_diff.median() - max_deviation) 
                & (double_diff <= double_diff.median() + max_deviation)
            )
            spikes.loc[chunk_group.index, cols_to_despike] = ~chunk_non_spikes
        spike_daynight.append(spikes)
    # call something a spike if both day and night agree that it is a spike. This can resolve conflicts at day/night boundaries
    # note that during the nighttime, the daytime series always indicates spikes
    spikes = spike_daynight[0] & spike_daynight[1]
    # invert: True indicates OK
    spike_flag = ~spikes
    return spike_flag


__all__ = ["mad_despike_papale_2006"]