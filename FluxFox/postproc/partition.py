# Author: Alex Fox
# Created: 2026-06-17

"""
Partitioning utilities
"""

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import pandas as pd
import numpy as np
from scipy import optimize

from .utils import _check_common_args

def _lloyd_taylor_1994(T: np.ndarray, R_ref: float, E0: float) -> np.ndarray:
    """
    Lloyd and Taylor (1994) model of ecosystem respiration as a function of temperature.
    """
    T_ref = 10 + 273.15
    T0 = -46.02 + 273.15
    return R_ref * np.exp(E0 * (1. / (T_ref - T0) - 1. / (T - T0 + 273.15)))

def _fit_lt94(x_data: np.ndarray, y_data: np.ndarray, fix_E0: Optional[float] = None) -> Any:
    def objective(p: np.ndarray) -> float:
        if fix_E0 is not None:
            E0 = fix_E0
            R_ref = p
        else:
            R_ref, E0 = p
        y_hat = _lloyd_taylor_1994(x_data, R_ref=R_ref, E0=E0)
        # return np.sum((y_data - y_hat) ** 2)
        return np.sum(np.abs(y_data - y_hat))
    if fix_E0 is not None:
        res = optimize.minimize(
            objective,
            x0=np.array([2.0]), bounds=optimize.Bounds(lb=[-np.inf], ub=[np.inf]), 
            method="Nelder-Mead", 
            options={'maxiter': 10_000, 'xatol': 1e-6, 'fatol':1e-6}
        )
    else:
        res = optimize.minimize(
            objective,
            x0=np.array([2.0, 200.0]), bounds=optimize.Bounds(lb=[-np.inf, 0.0], ub=[np.inf, 650.0]), 
            method="Nelder-Mead", 
            options={'maxiter': 10_000, 'xatol': 1e-6, 'fatol':1e-6}
        )
    return res

@dataclass
class FalgeResult:
    GPP: pd.Series
    Reco: pd.Series
    res: optimize.OptimizeResult

def gpp_night_falge_2001(
    df: pd.DataFrame,
    isday: pd.Series,
    nee_col: str,
    t_col: str,
)-> FalgeResult:
    r"""
    Partitioning of NEE into GPP and Reco based on 
    Falge et al. (2001). "Gap filling strategies for defensible annual sums of net ecosystem exchange," Agricultural and Forest Meteorology

    This is a "quick and dirty" method, using nighttime NEE to estimate ecosystem respiration ($R_{eco}$) and then partitioning daytime NEE into GPP and Reco based on temperature observations, as in Lloyd and Taylor (1994).
    This method is largely outdated.

    $$
    R_{eco} = R_{ref} * \exp\left( E_0  \left( \frac{1}{T_ref - T_0} - \frac{1}{T - T_0} \right) \right)
    $$

    where $T_{ref}$=10°C and $T_0$=-46.02°C. $E_0$ and $R_{ref}$ are fitted parameters.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NEE data to be partitioned, along with air temperature.
        Must have a DatetimeIndex.
    isday: pd.Series
        Boolean series indicating daytime observations. Must align with df. utils.compute_isday can be used to generate this series.
    nee_col: str
        Name of the column containing NEE data.
    t_col: str
        Name of the column containing air temperature (or soil temperature) data.
    
    Returns
    -------
    FalgeResult
        Object containing partitioned GPP and Reco, and the optimization result (from scipy.optimize.minimize)
    """
    
    _check_common_args(df, isday)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()

    nighttime_NEE = df.loc[~isday, [nee_col, t_col]].dropna()
    res = _fit_lt94(x_data=nighttime_NEE[t_col], y_data=nighttime_NEE[nee_col])
    
    R_ref, E0 = res.x

    if R_ref < 1e-6 or E0 < 1e-7:
        warnings.warn(f"Fitted R_ref ({R_ref}) or E0 ({E0}) are very small. Check data quality and fitting procedure. This is often caused by NEE being biased negative, especially at night.")
    
    df_out = pd.DataFrame(index=df.index)
    df_out['Reco'] = _lloyd_taylor_1994(df[t_col].values, R_ref=R_ref, E0=E0)
    df_out['GPP'] = -df[nee_col] + df_out['Reco']

    # cap to non-negative values
    df_out['GPP'] = df_out['GPP'].clip(lower=0)
    df_out['Reco'] = df_out['Reco'].clip(lower=0)
    
    return FalgeResult(
        GPP=df_out['GPP'],
        Reco=df_out['Reco'],
        res=res
    )
    
@dataclass
class ReichsteinResult:
    """
    Diagnostics for the Reichstein et al. (2005) partitioning method.
    Contains the fitted GPP and Reco timeseries, E0 parameter, the time-varying R_ref estimates, and diagnostic DataFrames for both E0 and R_ref displaying the optimize.OptimizeResult objects (Nelder-Mead method)
    """
    GPP: pd.Series
    Reco: pd.Series
    E0: float
    R_ref: pd.Series
    E0_diag: pd.DataFrame
    R_ref_diag: pd.DataFrame

def gpp_night_reichstein_2005(
    df: pd.DataFrame,
    isday: pd.Series,
    nee_col: str,
    t_col: str,
    lat: float, lon: float, elev: float = 0,
    sw_thresh: float = 20,
    E0_window_width_days: int = 14,
    R_ref_window_width_days: int = 7,
    n_best_E0: int = 3,
    min_datapoints: int = 6,
    min_temp_range: float = 5.0,
)-> ReichsteinResult:
    r"""
    Partitioning of NEE into GPP and Reco based on 
    Reichstein et al. (2005). "On the separation of net ecosystem exchange into assimilation and ecosystem respiration: review and improved algorithm," Global Change Biology

    This method is a more sophisticated approach compared to Falge 2001, as it accounts for temporal variability in ecosystem respiration.

    This method fits a Lloyd-Taylor model to nighttime NEE data to estimate ecosystem respiration parameters in 2 stages:
    1. Estimate E0:
        Within a moving window of size `E0_window_width_days`, the E0 parameter is estimated from nighttime NEE data. Using a LOO-CV approach, the uncertainty of E0 within each window is assessed.
        Windows with sample size less than `min_datapoints` or temperature range less than `min_temp_range` are skipped.
        Then, the best `n_best_E0` estimates (based on the smallest standard error) are selected to represent the E0 parameter, and averaged.
    2. Estimate R_ref:
        Within a moving window of size `R_ref_window_width_days`, the R_ref parameter is estimated from nighttime NEE data using the previously determined E0.
        Windows with sample size less than `min_datapoints` or temperature range less than `min_temp_range` are skipped.
        The resulting R_ref estimates are then linearly interpolated.
    
    Reco is then calculated using the Lloyd Taylor model (as a function of temperature) with the estimated E0 and interpolated R_ref for all timepoints (including nighttime ones).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NEE data to be partitioned, along with air temperature.
        Must have a DatetimeIndex.
    isday : pd.Series
        Boolean series indicating daytime observations. Must align with df. utils.compute_isday can be used to generate this series.
    nee_col: str
        Name of the column containing (storage-corrected) NEE data.
    t_col: str
        Name of the column containing air temperature (or soil temperature) data.
    E0_window_width_days : int, default=14
        Width of the moving window (in days) used to estimate E0
    R_ref_window_width_days : int, default=7
        Width of the moving window (in days) used to estimate R_ref
    n_best_E0 : int, default=3
        Number of best E0 estimates to consider based on the smallest standard error.
    min_datapoints : int, default=6
        Minimum number of data points required within a moving window to perform the estimation.
    min_temp_range : float, default=5.0
        Minimum temperature range (in °C) required within a moving window to perform the estimation.
    
    Returns
    -------
    ReichsteinResult
        Object containing partitioned GPP and Reco, and a ReichsteinDiagnostics object diagnostics to evaluate the partitioning.
    """

    _check_common_args(df, isday)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()
    if not E0_window_width_days > 0:
        raise ValueError("E0_window_width_days must be positive.")
    if not int(E0_window_width_days) == E0_window_width_days:
        raise ValueError("E0_window_width_days must be an integer.")
    if not R_ref_window_width_days > 0:
        raise ValueError("R_ref_window_width_days must be positive.")
    if not int(R_ref_window_width_days) == R_ref_window_width_days:
        raise ValueError("R_ref_window_width_days must be an integer.")
    if not n_best_E0 > 0:
        raise ValueError("n_best_E0 must be positive.")
    if not int(n_best_E0) == n_best_E0:
        raise ValueError("n_best_E0 must be an integer.")
    if not min_datapoints > 0:
        raise ValueError("min_datapoints must be positive.")
    if not int(min_datapoints) == min_datapoints:
        raise ValueError("min_datapoints must be an integer.")
    if not min_temp_range > 0:
        raise ValueError("min_temp_range must be positive.")
    E0_window_width_days = int(E0_window_width_days)
    R_ref_window_width_days = int(R_ref_window_width_days)
    n_best_E0 = int(n_best_E0)
    min_datapoints = int(min_datapoints)

    nighttime_NEE = df.loc[~isday, [nee_col, t_col]].dropna()

    # estimate E0 from a collection of short nighttime windows
    E0_results = []
    for idx, window in nighttime_NEE[[t_col, nee_col]].groupby(pd.Grouper(freq=f"{E0_window_width_days}D")):
        window = window.dropna()
        npoints = window.shape[0]
        trange = window[t_col].max() - window[t_col].min()
        if npoints < min_datapoints or trange < min_temp_range:
            continue
        # loo_idx = rng.choice(window.index, size=min(20, window.shape[0]), replace=False)
        # for loo_iter, iloo in enumerate(loo_idx):
        #     window_loo = window.drop(index=iloo)
        #     npoints = window_loo.shape[0]
        #     trange = window_loo[t_col].max() - window_loo[t_col].min()
        #     if npoints < min_datapoints or trange < min_temp_range:
        #         continue
        try:
            res = _fit_lt94(x_data=window[t_col].to_numpy(), y_data=window[nee_col].to_numpy())
        except Exception as e:
            warnings.warn(f"E0 fitting failed for window starting at {idx}: {e}")
            continue
        success, status, fun, R_ref, E0, nit, nfev = res.success, res.status, res.fun, *res.x, res.nit, res.nfev
        E0_results.append([idx, success, status, fun, R_ref, E0, nit, nfev, npoints, trange])
    E0_results = pd.DataFrame(E0_results, columns=['idx', 'success', 'status', 'fun', 'R_ref', 'E0', 'nit', 'nfev', 'npoints', 'trange'])
    E0_results = E0_results.loc[E0_results['success']].dropna()
    if E0_results.shape[0] < n_best_E0:
        raise ValueError(f"Not enough successful E0 fits to compute a reliable estimate. Required: {n_best_E0}, available: {E0_results.shape[0]}")
    E0 = E0_results[["E0", "fun"]].sort_values("fun").iloc[:n_best_E0].mean()["E0"]
    if E0 < 1e-6:
        warnings.warn(f"Fitted E0 ({E0}) is very small. Check data quality and fitting procedure. This is often caused by NEE being biased negative, especially at night.")

    # Estimate R_ref over time using the moving window approach.
    R_ref_results = []
    for idx, window in nighttime_NEE[[t_col, nee_col]].groupby(pd.Grouper(freq=f"{R_ref_window_width_days}D")):
        window = window.dropna()
        if window.shape[0] < min_datapoints or window[t_col].max() - window[t_col].min() < min_temp_range:
            continue
        try:
            res = _fit_lt94(x_data=window[t_col].to_numpy(), y_data=window[nee_col].to_numpy(), fix_E0=E0)
        except Exception as e:
            warnings.warn(f"R_ref fitting failed for window starting at {idx}: {e}")
            continue
        success, status, fun, R_ref, nit, nfev = res.success, res.status, res.fun, *res.x, res.nit, res.nfev
        R_ref_results.append([idx, success, status, fun, R_ref, E0, nit, nfev, window.shape[0], window[t_col].max() - window[t_col].min()])
    R_ref_results = pd.DataFrame(R_ref_results, columns=['idx', 'success', 'status', 'fun', 'R_ref', 'E0', 'nit', 'nfev', 'npoints', 'trange']).set_index("idx")
    
    R_ref_results = R_ref_results.loc[R_ref_results['success']]
    R_ref = R_ref_results["R_ref"]
    R_ref = R_ref.reindex(df.index).interpolate(method="time")
    if R_ref.quantile(0.95) < 1e-6:
        warnings.warn(f">=95% of fitted R_ref ({R_ref.quantile(0.95)}) values are very small. Check data quality and fitting procedure. This is often caused by NEE being biased negative, especially at night.")
    elif R_ref.quantile(0.05) < 1e-6:
        warnings.warn(f">=5% of fitted R_ref ({R_ref.quantile(0.05)}) values are very small. Check data quality and fitting procedure. This is often caused by NEE being biased negative, especially at night.")

    # apply lloyd taylor
    R_eco = _lloyd_taylor_1994(T=df[t_col], R_ref=R_ref, E0=E0)
    GPP = -df[nee_col] + R_eco

    # clip to 0
    GPP = GPP.clip(lower=0)

    results = ReichsteinResult(
        GPP=GPP,
        Reco=R_eco,
        E0=E0,
        R_ref=R_ref.astype("float32"),
        E0_diag=E0_results,
        R_ref_diag=R_ref_results
    )

    return results



__all__ = ["ReichsteinResult", "FalgeResult", "gpp_night_reichstein_2005", "gpp_night_falge_2001"]