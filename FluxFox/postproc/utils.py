# Author: Alex Fox
# Created: 2026-06-17
"""
Utility functions for post-processing eddy flux data.
"""

import pandas as pd
import numpy as np
import solarpy
from typing import Optional
import warnings

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
    for d in dates:
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
    for d in dates:
        panel.set_datetime(d.to_pydatetime())
        df.loc[d, "SW_IN_POT"] = panel.power()

    isday = df["SW_IN_POT"] > sw_thresh
    return isday.astype(bool).loc[timestamps]

def _check_common_args(
    df: pd.DataFrame,
    isday: pd.Series,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(isday, pd.Series):
        raise TypeError("isday must be a pandas Series")
    if not isday.index.equals(df.index):
        raise ValueError("isday must have the same index as df")
    if not isday.dtype == bool:
        raise TypeError("isday must be of boolean dtype")
    if not df.index.is_unique:
        raise ValueError("df must have a unique index")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    

def season_to_month(season: int, n_seasons: int) -> int:
    """
    Convert a season number to the starting month of that season.

    Parameters
    ----------
    season : int
        The season number (1-based).
    n_seasons : int
        The total number of seasons in a year.

    Returns
    -------
    month : int
        The starting month of the given season.
    """
    return (season - 1) * (12 // n_seasons) + 1

def month_to_season(month: int, n_seasons: int) -> int:
    """
    Convert a month number to the corresponding season number.

    Parameters
    ----------
    month : int
        The month number (1-based).
    n_seasons : int
        The total number of seasons in a year.

    Returns
    -------
    season : int
        The season number corresponding to the given month.
    """
    return ((month - 1) // (12 // n_seasons)) + 1

def compute_storage_single_point(
    df: pd.DataFrame,
    rho_col: str,
    co2_col: Optional[str] = None,
    h2o_col: Optional[str] = None,
    t_col: Optional[str] = None,
    co2_zm: Optional[float] = None,
    h2o_zm: Optional[float] = None,
    t_zm: Optional[float] = None
) -> pd.DataFrame:
    r"""
    Simple function to compute single-point storage fluxes for CO2, H2O, and temperature based on the provided measurement columns and measurement heights.
    
    Storage flux is computed as

    $$
    S_x = \frac{\Delta x}{\Delta t} \cdot z_m
    $$

    for gas storage flux,

    $$
    S_H = \rho_a c_p \frac{\Delta T}{\Delta t} \cdot z_m
    $$

    for sensible heat storage flux, and

    $$
    S_L = \lambda \frac{\Delta H2O}{\Delta t} \cdot z_m
    $$

    where $\lambda$ is the latent heat of vaporization for water.

    This assumes there is not gradient in temperature, concentration, or air density between the sensor and the ground.

    Can be turned into a multi-point storage flux calculation by applying this function to multiple measurement heights and combining the results appropriately in a Riemann sum. In such a case, zm would represent the height difference between consecutive measurement heights, and the storage flux would represent the contribution from that layer to the total storage flux.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the measurement data. Must have a DatetimeIndex.
    rho_col : str
        The column of air density measurements. Report in kg m-3.
    co2_col : Optional[str]
        The column of CO2 concentration measurements. Report in mole fraction (ppm).
        REQUIRED for computing SC
    h2o_col : Optional[str]
        The column of H2O concentration measurements. Report in mole fraction (ppt).
        REQUIRED for computing SLE and SH2O
        REQUIRED for computing SC.
    t_col : Optional[str]
        The column of temperature measurements. Report in degrees Celsius.
        REQUIRED for computing SH.
        PREFERRED for computing SLE and SH2O (enables precise calculation of $\lambda_v$ and $\rho_w$.)
    co2_zm : Optional[float]
        The measurement height for CO2 concentration measurements. Required if computing the CO2 storage flux. Report in meters.
    h2o_zm : Optional[float]
        The measurement height for H2O concentration measurements. Required if computing the H2O storage flux. Report in meters.
    t_zm : Optional[float]
        The measurement height for temperature measurements. Required if computing the temperature storage flux. Report in meters.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed storage fluxes.
        If co2_col is provided, the output dataframe will have an "SC" column corresponding to the computed CO2 storage flux (µmol m-2 s-1).
        If h2o_col is provided, the output dataframe will have an "SH2O" column corresponding to the computed H2O storage flux (mmol m-2 s-1) and SLE for latent heat storage flux (assuming the latent heat of vaporization for water) (W m-2).
        If t_col is provided, the output dataframe will have an "SH" column corresponding to the computed sensible heat storage flux (W m-2).
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.index.is_unique:
        raise ValueError("df must have a unique index")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if rho_col is None:
        raise ValueError("rho_col must be provided for all storage flux calculations.")
    if rho_col not in df.columns:
        raise ValueError(f"Column {rho_col} not found in dataframe")

    dt = df.index.diff().total_seconds().median()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=pd.Timedelta(seconds=dt))).sort_index()
    
    SC = None
    if co2_col is not None:
        if co2_zm is None:
            raise ValueError("co2_zm must be provided if co2_col is specified")
        if co2_col not in df.columns:
            raise ValueError(f"Column {co2_col} not found in dataframe")

        # compute molar volume of air
        # M = Md*(1 - h2o) + Mv*h2o
        M = 0.028965 * (1 - df[h2o_col]/1000) + 0.018016 * (df[h2o_col]/1000)  # kg mol-1
        Vm = M / df[rho_col]  # m3 mol-1
        SC = df[co2_col].diff() / dt * co2_zm / Vm  # µmol m-2 s-1
    
    SH2O = None
    SLE = None
    if h2o_col is not None:
        if h2o_zm is None:
            raise ValueError("h2o_zm must be provided if h2o_col is specified")
        if h2o_col not in df.columns:
            raise ValueError(f"Column {h2o_col} not found in dataframe")
        
        M = 0.028965 * (1 - df[h2o_col]/1000) + 0.018016 * (df[h2o_col]/1000)  # kg mol-1
        Vm = M / df[rho_col]  # m3 mol-1
        SH2O = df[h2o_col].diff() / dt * h2o_zm / Vm

        if t_col is not None and t_col in df:
            lam = 2.501e6 - 2361.4*df[t_col]
            # Thermal expansivity: 0 if T < 0, 2.07e-4 if T >= 20, linearly interpolate in between
            beta = np.select(
                [df[t_col] <= 0, df[t_col] >= 20],
                [0, 2.07e-4],
                default = 2.07e-4 * df[t_col] / 20
            )
            rho_w = 998.203 / (1 + beta*(df[t_col] - 20))
        else:
            lam = 2.454e6
            rho_w = 998.203
            warnings.warn("t_col not provided or not in dataframe, defaulting to lambda = 2.454e6 J / kg, rho_w = 998.203 kg m-3 at 20°C.")
        
        # convert mmol m-2 s-1 to W m-2
        SLE = SH2O * rho_w/1000 * lam

    SH = None
    if t_col is not None:
        if t_col not in df.columns:
            raise ValueError(f"Column {t_col} not found in dataframe")
        if t_zm is None:
            raise ValueError("t_zm must be provided if t_col is specified")

        # compute cp: linterp between 1003 @-23.15C & 1008 @78.85C
        c_p = np.select(
            [df[t_col] <= -23.15, df[t_col] >= 78.85],
            [1003, 1008],
            default = 1003 + (1008 - 1003) * (df[t_col] + 23.15) / (78.85 + 23.15)
        )
        SH = df[rho_col] * c_p * df[t_col].diff() / dt * t_zm

    df_out = pd.DataFrame()
    if SC is not None:
        df_out["SC"] = SC
    if SH2O is not None:
        df_out["SH2O"] = SH2O
    if SLE is not None:
        df_out["SLE"] = SLE
    if SH is not None:
        df_out["SH"] = SH
    return df_out.reindex(df.index)
    