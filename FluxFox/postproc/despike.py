# Author: Alex Fox
# Created: 2026-06-17
"""
Postprocessing despiking methods
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import _check_common_args

def despike_mad_papale_2006(
    df: pd.DataFrame,
    isday: pd.Series,
    cols_to_despike: list[str],
    z: float=4,
    chunksize_days: int=13,
)->pd.Series:
    """
    Despike the specified columns in the dataframe using the method described by Papale et al. (2006), based on the median absolute deviation (MAD) of double differences within chunks of data.
    
    Overview of method
    1. Split the data into daytime and nighttime based on the `isday` series, adding padding around sunrise and sunset to avoid abrupt transitions.
    2. Chunk the data into blocks of `chunksize_days`.
    3. Compute the double difference within each block: d = (X_i - X_i-1) - (X_i+1 - X_i)
    4. Compute the MAD of the double differences within each block.
    5. Flag points as spikes if they deviate from the median by more than z standard deviations (using the MAD to estimate the standard deviation of a normal distribution).
    6. Combine day and night spike flags. During dawn and dusk, a point is only considered a spike if both day and night agree that it is a spike.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data to be despiked.
    isday : pd.Series
        Boolean series indicating daytime observations. Must align with df. utils.compute_isday can be used to generate this series.
    cols_to_despike : list[str]
        List of column names to apply the despiking method to.
    z : float, default=4
        Threshold multiplier for the MAD-based spike detection.
    chunksize_days : int, default=13
        Size of the chunks (in days) to compute the MAD within.
    
    Returns
    -------
    pd.Series
        Boolean series indicating non-spike points (`True` for OK, `False` for spikes).
    """
    _check_common_args(df, isday)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()
    if not z > 0:
        msg = f"z must be > 0, got {z}"
        raise ValueError(msg)
    
    if not chunksize_days == int(chunksize_days):
        msg = f"chunksize_days must be an integer, got {chunksize_days}"
        raise ValueError(msg)
    if chunksize_days <= 0:
        msg = f"chunksize_days must be > 0, got {chunksize_days}"
        raise ValueError(msg)
    chunksize_days = int(chunksize_days)

    for c in cols_to_despike:
        if c not in df.columns:
            msg = f"Column {c} not found in dataframe."
            raise ValueError(msg)
        
    # 1. split into day/night
    
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


__all__ = ["despike_mad_papale_2006"]