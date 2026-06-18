from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

def fingerprint_plot(
    y: pd.Series,
    smear_days: int = 0,
    na_color: str | tuple = "lightgrey",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[int, int]] = (6, 8),
    **pcolormesh_kwargs
) -> plt.Axes:
    """
    Plots a fingerprint plot of the given time series data.

    Parameters
    ----------
    y : pd.Series
        The time series data to plot with a DatetimeIndex.
    smear_days : int, optional
        The number of days to smear the data over. This can be used to smooth the data.
    na_color: str | tuple, optional
        The color to use for NaN values in the plot. Can either be a matplotlib named color or an RGB(A) tuple.
    ax : Optional[plt.Axes], optional
        The matplotlib Axes object to plot on. If None, a new figure and axes are created.
    figsize : Optional[tuple[int, int]], optional
        The size of the figure if a new figure is created (used when ax is None).
    **pcolormesh_kwargs : dict
        Additional keyword arguments to pass to the `ax.pcolormesh` function, such as `cmap`, `alpha`, `norm`, `vmin`, `vmax`, etc.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the fingerprint plot.
    """
    # sort y by the index, drop duplicates, and turn into a contiguous timeseries
    y = y.sort_index()
    y = y.loc[~y.index.duplicated(keep='last')]
    dt = y.index.diff().median()
    dt_hours = dt.total_seconds() / 3600
    fill_idx = pd.date_range(y.index[0].floor("1D"), y.index[-1].ceil("1D") - dt, freq=dt)
    y = y.reindex(fill_idx)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    time_axis = np.arange(0, 24, dt_hours)
    date_axis = pd.date_range(y.index[0], y.index[-1], freq="1D")

    TT, DD = np.meshgrid(time_axis, date_axis)
    YY = y.values.reshape(TT.shape)

    if smear_days > 0:
        window = np.ones(smear_days)/smear_days
        if smear_days >= 7:
            window = np.hanning(smear_days + 2)
            window /= window.sum()
        for i in range(YY.shape[1]):
            YY[:, i] = np.convolve(YY[:, i], window, mode='same')


    pcolormesh_kwargs = dict() if pcolormesh_kwargs is None else pcolormesh_kwargs
    if "cmap" not in pcolormesh_kwargs:
        pcolormesh_kwargs["cmap"] = "RdBu"
    if "norm" not in pcolormesh_kwargs and "vmin" not in pcolormesh_kwargs and "vmax" not in pcolormesh_kwargs:
        minmax = max(np.abs(np.nanquantile(YY, 0.025)), np.abs(np.nanquantile(YY, 0.975)))
        from matplotlib import colors as mcolors
        pcolormesh_kwargs["norm"] = mcolors.TwoSlopeNorm(vmin=-minmax, vcenter=0, vmax=minmax)
    
    # plot NaNs
    if not isinstance(pcolormesh_kwargs["cmap"], mpl.colors.Colormap):
        data_cmap = plt.cm.get_cmap(pcolormesh_kwargs.get("cmap", "RdBu")).copy()
        data_cmap.set_bad(color=na_color)
        pcolormesh_kwargs["cmap"] = data_cmap
    ax.pcolormesh(TT, DD, YY, **pcolormesh_kwargs)
    ax.set_ylabel("Date")

    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("Hour of Day")
    ax.set_title(y.name if y.name else "Fingerprint Plot")
    
    return ax

__all__ = ["fingerprint_plot"]