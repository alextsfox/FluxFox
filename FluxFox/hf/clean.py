from typing import Literal

import pandas as pd
import numpy as np

def _despike_window(chunk_df, z_thresh):
        double_diff = chunk_df.diff() - chunk_df.shift(-1).diff()
        median = double_diff.median()
        mad = (double_diff - median).abs().median()
        z = (double_diff - median) / mad
        z = z.fillna(0).replace([np.inf, -np.inf], 0)
        chunk_keep = z.abs() < z_thresh
        return chunk_keep.to_numpy()

def despike_mds(df: pd.DataFrame, z_thresh: float | list[float], chunksize: int) -> pd.DataFrame:
    """
    2nd derivative MAD despiker.

    * Loop through the data in non-overlapping windows of size ``chunksize``. In each window:
        1. Compute the modified z-score as ..math:: z = (x - \text{median}(x))/\text{MAD}(x), where ..math:: \text{MAD}(x) = \text{median}(|x - \text{median}(x)|).
        2. Flag values as spikes if their absolute z-score exceeds the corresponding threshold in ``z_thresh*0.6745`` (scaling factor converts from std units to MAD units).
    * Repeat the loop with an offset of 2 datapoints to reduce edge effects
    * Remove datapoints flagged as spikes by BOTH passes.
    * Linearly interpolate removed spikes

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame to be despiked.
    z_thresh: float | list[float]
        The threshold(s) for despiking, given as non-negative z-scores ( (x - µ) / σ ). Will be scaled by 0.6745 to convert to mad units.
        If z_thresh=+inf, no despiking is applied to that column
    chunksize: int
        The size of chunks to process at a time.
   
    Returns
    -------
    pd.DataFrame
        A boolean data frame indicating which values in the input dataframe are to be kept (True) or despiked (False).
    """

    if "__len__" not in dir(z_thresh):
        z_thresh = [z_thresh] * df.shape[1]
    if len(z_thresh) != df.shape[1]:
        raise ValueError("Length of z_thresh must match the number of columns in df")
    z_thresh = [z*0.6745 for z in z_thresh]

    step_size = 2
    if chunksize <= step_size*2:
        raise ValueError(f"chunksize must be greater than {step_size*2}")

    npoints = df.shape[0]

    # despike in 2 passes: first pass starts at the beginning, second pass starts at an offset
    # we then flag only the spikes that were detected in BOTH passes to eliminate edge effects.
    keep_1 = pd.DataFrame(True, index=df.index, columns=df.columns)
    start = 0
    while start < npoints:
        end = min(start + chunksize, npoints)
        chunk_df = df.iloc[start:end]
        chunk_keep = _despike_window(chunk_df, z_thresh)
        keep_1.iloc[start:end] = chunk_keep
        start += chunksize

    # second pass: start at an offset to 
    keep_2 = pd.DataFrame(True, index=df.index, columns=df.columns)
    start = step_size
    while start < npoints:
        end = min(start + chunksize, npoints)
        chunk_df = df.iloc[start:end]
        chunk_keep = _despike_window(chunk_df, z_thresh)
        keep_2.iloc[start:end] = chunk_keep
        start += chunksize
    keep_1.iloc[-step_size:] = False  # edge case: default to keep_2 for last step_size datapoints
    keep_2.iloc[:step_size] = False  # edge case: default to keep_1 for first step_size datapoints

    return keep_1 | keep_2


__all__ = ["despike_mds"]