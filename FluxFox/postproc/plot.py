from typing import Optional

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

def plot_fingerprint(
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
    im : matplotlib.collections.QuadMesh
        The matplotlib QuadMesh object returned by `ax.pcolormesh` representing the fingerprint plot.
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
    im = ax.pcolormesh(TT, DD, YY, **pcolormesh_kwargs)
    ax.set_ylabel("Date")

    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("Hour of Day")
    ax.set_title(y.name if y.name else "Fingerprint Plot")
    
    return im

_INTEGRITY_GRADIENT_STEPS = 256
_LOW_MISSING_THRESHOLD = 0.05
_HIGH_MISSING_THRESHOLD = 0.95
_COLORBAR_TICKS = [0, _LOW_MISSING_THRESHOLD, 0.25, 0.5, 0.75, _HIGH_MISSING_THRESHOLD, 1]
_COLORBAR_LABELS = ["0%", "5%", "25%", "50%", "75%", "95%", "100%"]


def _validate_integrity_dataframe(df):
    """Validate dataframe requirements for integrity visualization."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a pandas DatetimeIndex.")
    if df.empty:
        raise ValueError("df must contain at least one row.")
    if df.index.hasnans:
        raise ValueError("df index must not contain NaT values.")
    if not df.columns.is_unique:
        raise ValueError("df column names must be unique.")


def _select_integrity_columns(df, columns, regex):
    """Return dataframe columns selected by explicit names and an optional regex."""
    if columns is None:
        selected_columns = list(df.columns)
    elif isinstance(columns, str):
        selected_columns = [columns]
    else:
        selected_columns = list(columns)

    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found in df: {missing_columns}")

    if regex is not None:
        try:
            regex_columns = set(df.filter(regex=regex).columns)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid column-selection regex {regex!r}.") from error
        selected_columns = [
            column for column in selected_columns if column in regex_columns
        ]

    if not selected_columns:
        raise ValueError("Column selection did not match any dataframe columns.")
    return selected_columns


def _build_integrity_colormap(cmap, low_missing_color, high_missing_color):
    """Create the three-zone colormap used by the missingness heatmap."""
    middle_colormap = plt.get_cmap(cmap)
    middle_positions = np.linspace(0, 1, _INTEGRITY_GRADIENT_STEPS)
    middle_range = _HIGH_MISSING_THRESHOLD - _LOW_MISSING_THRESHOLD
    # Adjacent stops create sharp transitions into the solid endpoint bands.
    color_stops = [
        (0.0, low_missing_color),
        (np.nextafter(_LOW_MISSING_THRESHOLD, 0), low_missing_color),
        *[
            (
                _LOW_MISSING_THRESHOLD + middle_range * position,
                middle_colormap(position),
            )
            for position in middle_positions
        ],
        (np.nextafter(_HIGH_MISSING_THRESHOLD, 1), high_missing_color),
        (1.0, high_missing_color),
    ]
    return LinearSegmentedColormap.from_list("data_integrity", color_stops)


def plot_data_integrity(
    df,
    columns=None,
    regex=None,
    time_interval="ME",
    figsize=None,
    cmap="cool",
    low_missing_color="black",
    high_missing_color="white",
    title=None,
    ax=None,
):
    """Visualize variable availability and missingness over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a DatetimeIndex.
    columns : sequence of str, optional
        Explicit column names to inspect. When combined with ``regex``, only
        explicitly named columns that also match the expression are included.
    regex : str, optional
        Regular expression used to select columns, such as ``r"^FC"``.
    time_interval : str
        Pandas resampling frequency for the missingness heatmap. Example valid 
        values include "D" for daily, "H" for hourly, "ME" for monthly, etc, "3ME" for every three months, and so on.
    figsize : tuple of float, optional
        Figure dimensions in inches. A height based on the number of selected
        columns is used when omitted.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used between 5% and 95% missingness.
    low_missing_color : color, optional
        Color for periods with less than 5% missingness.
    high_missing_color : color, optional
        Color for periods with more than 95% missingness.
    title : str, optional
        Figure title.
    ax : array-like of matplotlib.axes.Axes, optional
        Axes to plot on. If None, new axes are created.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel data-integrity dashboard.
    """
    _validate_integrity_dataframe(df)
    selected_columns = _select_integrity_columns(df, columns, regex)
    selected = df.loc[:, selected_columns].sort_index()
    try:
        # Empty resampling bins represent periods with no observations at all.
        missing_by_period = selected.isna().resample(time_interval).mean().fillna(1.0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"time_interval {time_interval!r} is not a valid pandas frequency."
        ) from error

    availability = (1 - selected.isna().mean()).sort_values()

    if figsize is None:
        figsize = (16, max(8, 3 + 0.28 * len(selected_columns)))

    if ax is None:
        figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        if title is not None:
            figure.suptitle(title, fontsize=16)
    else:
        axes = ax
        figure = ax[0].figure

    availability_percent = availability * 100
    axes[0].barh(availability_percent.index, availability_percent, color="tab:blue")
    axes[0].invert_yaxis()
    axes[0].set(
        title="Data availability by variable",
        xlabel="Available observations (%)",
        xlim=(0, 100),
    )

    integrity_colormap = _build_integrity_colormap(
        cmap,
        low_missing_color,
        high_missing_color,
    )
    heatmap = axes[1].imshow(
        missing_by_period.T.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=integrity_colormap,
        vmin=0,
        vmax=1,
    )
    heatmap_positions = np.linspace(
        0, len(missing_by_period.index) - 1, min(8, len(missing_by_period.index)), dtype=int
    )
    axes[1].set(
        title=f"Data Integrity",
        ylabel="Variable",
        xticks=heatmap_positions,
        xticklabels=[
            missing_by_period.index[position].strftime("%Y-%m-%d")
            for position in heatmap_positions
        ],
        yticks=np.arange(len(selected_columns)),
        yticklabels=selected_columns,
    )
    axes[1].tick_params(axis="x", rotation=45)
    colorbar = figure.colorbar(
        heatmap,
        ax=axes[1],
        ticks=_COLORBAR_TICKS,
        label="Missing observations",
    )
    colorbar.minorticks_off()
    colorbar.ax.set_yticklabels(_COLORBAR_LABELS)
    return figure

__all__ = ["plot_fingerprint", "plot_data_integrity"]