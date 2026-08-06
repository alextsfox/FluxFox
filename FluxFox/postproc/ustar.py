# Author: Alex Fox
# Created: 2026-06-17
"""
Creates a U* filter for a given dataset based on air temperature, U*, and NEE
"""


import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils import _check_common_args, month_to_season, season_to_month

@dataclass
class UStarFilterResult:
    """Container for U* filter output and diagnostics.
    
    Attributes
    ----------
    flag : pd.Series
        Boolean series indicating which values pass the U* filter (False = fail)
    thresholds : pd.DataFrame
        DataFrame of U* thresholds by season and year.
    qual : pd.DataFrame
        DataFrame of U* quality metrics by season and year. 
        0 = high quality estimate
        1 = medium quality estimate (indicates low sample size or temperature correlation issues)
        2 = low quality estimate (gap-filled)
    """
    flag: pd.Series
    thresholds: pd.DataFrame
    qual: pd.DataFrame


# possible speedups, from clause

"""
Reformulate to avoid 3-parameter non-smooth optimization (biggest win)
For a fixed x0, the model intercept + slope * min(x, x0) is linear in (intercept, slope). That means the MAE/MSE loss over (intercept, slope) is a standard (constrained) L1/L2 regression problem with a closed-form or cheap solution — you don't need a general-purpose optimizer for those two parameters at all.

This lets you collapse the search to a 1-D bounded scan over x0 only:

from scipy.optimize import minimize_scalar

def _fit_at_x0(x, y, x0, loss):
    z = np.minimum(x, x0)
    if loss == "mse":
        # closed-form OLS, then clip slope to >= 0
        A = np.column_stack([np.ones_like(z), z])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept, slope = coef
        if slope < 0:
            slope = 0.0
            intercept = y.mean()
    else:
        # L1 regression via linprog, or IRLS - still O(n), no multi-start needed
        ...
    resid = y - (intercept + slope * z)
    err = np.mean(np.abs(resid)) if loss == "mae" else np.mean(resid**2)
    return intercept, slope, err

def _fit_broken_stick_fast(x, y, ust_thresh_bounds, loss="mae"):
    def objective(x0):
        _, _, err = _fit_at_x0(x, y, x0, loss)
        return err
    res = minimize_scalar(objective, bounds=ust_thresh_bounds, method="bounded")
    intercept, slope, _ = _fit_at_x0(x, y, res.x, loss)
    return intercept, slope, res.x

This replaces a 3-parameter, non-smooth, multi-start, finite-difference L-BFGS-B search with a 1-D golden-section/Brent scan (method="bounded" in minimize_scalar) wrapping a cheap 2-parameter linear fit. It's dramatically faster per call and far more robust (no local minima issues in 1-D over a bounded interval), which likely lets you shrink or even drop the multi-start component of your current bootstrap loop entirely (see #2).

3. Parallelize the independent fits
The n_boot loop in _fit_broken_stick, and the outer TA-class/season/year loops in ustar_broken_stick, are all embarrassingly parallel — each fit is independent. Since minimize/minimize_scalar calls release the GIL relatively little (they're pure Python + numpy), a concurrent.futures.ProcessPoolExecutor or joblib.Parallel(n_jobs=-1) across bootstrap samples (or across TA classes) would let you use multiple cores instead of the current fully sequential loop.

"""
def _broken_stick(x, intercept, slope, x0):
    """Piecewise linear: rising line up to x0, then flat (slope=0) beyond it."""
    return np.where(x < x0, intercept + slope * x, intercept + slope * x0)
def _mae_loss(params, x, y):
    intercept, slope, x0 = params
    y_pred = _broken_stick(x, intercept, slope, x0)
    return np.mean(np.abs(y - y_pred))
def _mse_loss(params, x, y):
    intercept, slope, x0 = params
    y_pred = _broken_stick(x, intercept, slope, x0)
    return np.mean((y - y_pred)**2)
def _fit_broken_stick(x, y, p0=(0.0, 15.0, 0.2), n_boot=30, ust_thresh_bounds=(0, 1), loss:Literal["mse", "mae"]="mae") -> tuple[float, float, float]:
    rng = np.random.default_rng()
    bounds = [
        (None, None),  # intercept: unbounded
        (0.0, None),   # slope: must be positive
        ust_thresh_bounds,   # x0: bounded intersection point
    ]
    x0s = []
    intercepts = []
    slopes = []
    for ust_thresh_init in rng.random(n_boot)*(ust_thresh_bounds[1]-ust_thresh_bounds[0]) + ust_thresh_bounds[0]:
        # idx = rng.choice(len(x), size=size, replace=True)
        x_boot = x
        y_boot = y
        if loss == "mae":
            result = minimize(_mae_loss, (p0[0], p0[1], ust_thresh_init), args=(x_boot, y_boot), bounds=bounds, method="L-BFGS-B")
        else:
            result = minimize(_mse_loss, (p0[0], p0[1], ust_thresh_init), args=(x_boot, y_boot), bounds=bounds, method="L-BFGS-B")
        intercept, slope, x0 = result.x

        if (not result.success) or (x0 == ust_thresh_bounds[0]) or (x0 == ust_thresh_bounds[1]):
            x0 = np.nan
            intercept = np.nan
            slope = np.nan
        
        x0s.append(x0)
        intercepts.append(intercept)
        slopes.append(slope)
    if all(np.isnan(x0s)):
        return np.nan, np.nan, np.nan
    return np.nanmedian(intercepts), np.nanmedian(slopes), np.nanmean(x0s)

def _fit_ta_class(ta_group, ta_col, ustar_col, nee_col, ustar_ta_corr_cutoff, ustar_thresh_bounds, parallel_fits):
    # 4. correlation check: skip class if |R(TA<USTAR)| is too high
    corr = ta_group[ta_col].corr(ta_group[ustar_col])
    if abs(corr) >= ustar_ta_corr_cutoff:
        # warnings.warn(f"R(U*,TA) = {abs(corr):.2f} > {ustar_ta_corr_cutoff:.2f} for TA class [{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping")
        return np.nan
    _, _, plateau_ustar = _fit_broken_stick(ta_group[ustar_col].values, ta_group[nee_col].values, p0=(0.0, 15.0, max(0.2, ustar_thresh_bounds[1])), n_boot=parallel_fits, ust_thresh_bounds=ustar_thresh_bounds)
    return plateau_ustar

def ustar_broken_stick(
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str, ustar_col: str, nee_col: str,
    seasons: int|list[list[int]]=4,
    n_ta_classes: int=6,
    ustar_ta_corr_cutoff: float=0.4,
    ustar_thresh_bounds: tuple[float, float]=(0, 1),
    n_restarts: int=10,
    gapfill_quantile: float=0.75,
    default_ustar_thresh: float=0.2,
    pbar:bool=False,
)->UStarFilterResult:
    r"""
    Creates a U* filter for each season for each year in the dataset, using a broken stick model to find the U* threshold:

    ::math::
        \text{NEE} = \begin{cases}
            \text{intercept} + \text{slope} * U^* & \text{if } U^* < U^*_\text{thresh} \\
            \text{intercept} + \text{slope} * U^*_\text{thresh} & \text{if } U^* \geq U^*_\text{thresh}
        \end{cases}

    This method has advantages of the Papale et al (2006) method when nighttime NEE is highly heteroschedastic: 
    if the standard deviation of NEE is much larger for large U* (unrelated to temperature), then the Papale et al (2006) method often
    over-estimates the U* threshold. The broken stick method is more robust to this heteroschedasticity, and can provide a more believable estimate of the U* threshold in these cases.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
        Must have:
            * columns for air temperature, U*, and nee/fc/co2_flux (positive-upwards)
            * have a `pd.DatetimeIndex`. This index should be a meridian offset time, not a civil time. e.g. Anchorage, AK, USA has the civil time zone of UTC-8 in the Summer and UTC-9 in the Winter, but has a meridian offset time zone of UTC-10 (calculated from its longitude).
    isday : pd.Series
        Boolean series indicating daytime (`True`) and nighttime (`False`) for each timestamp in `df.index`. utils.compute_isday can be used to generate this series.
    ta_col : str
        The column name for air temperature in `df`. Unit-agnostic.
    ustar_col : str
        The column name for ustar in `df`. m s-1 preferred.
    nee_col : str
        The column name for NEE (storage-corrected CO2 flux is preferred over turbulent CO2 flux alone) in `df`. Unit-agnostic.
    seasons : int|list[int]
        The number of seasons in the year. Default 4. Increase this value if you want finer seasonal resolution for U* threshold estimation. Decrease this value if the algorithm has trouble converging on a solution.
    n_ta_classes : int
        The number of air temperature classes. Default 6. Increase this value to limit the impact of temperature variation on the final U* threshold. Decrease this value if the algorithm has trouble converging on a solution.
    ustar_ta_corr_cutoff : float
        The maximum acceptable correlation between U* and TA. Seasons where |R(U*, TA)| > `ustar_ta_corr_cutoff` will be skipped to avoid confounding the U* ~ NEE relationship. Default 0.4. Increase this value to allow more correlation between U* and TA before skipping a season. Decrease this value to be more strict about the correlation threshold, however, it is recommended to also increase `n_ta_classes` to reduce the impact of temperature variation when the correlation threshold is looser.
    ustar_thresh_bounds : tuple[float, float]
        Bounds for the U* threshold when fitting the broken stick model. Default (0, 1). Increase the upper bound if you expect U* thresholds to be higher than 1 m/s. Decrease this upper bound if your data has high scatter, or the method fails to converge often.
    n_restarts : int
        Number of restarts of the broken stick model to perform with randomized initial parameters. Higher values (>10) are more robust, but slower.
    gapfill_quantile : float
        If the algorithm fails for a given season/year, U* thresholds are filled in with this quantile first by season, then by year. For example, if in 2012, the ustar thresholds for Winter, Spring, Summer, and Fall were 0.19, nan, 0.21, 0.3, and `gapfill_quantile`=0.75, the nan would be filled with the 0.75 quantile of [0.19, 0.21, 0.3]. Default 0.75.
    default_ustar_thresh : float
        If the entire pipeline fails, this is the fallback U* threshold value that will be used in the filter. Default 0.2.
    pbar : bool
        Whether to show a progress bar. Requires tqdm. Default False
    
    Returns
    -------
    UStarFilterResult
        A dataclass containing:
            * `flag`: pd.Series of type `bool`, indexed by `df.index`. `False` indicates that the datapoint should be filtered out.
            * `thresholds`: pd.DataFrame with columns representing seasons and rows representing years. Values indicate the U* threshold for that season and year.
            * `qual`: pd.DataFrame for each year/season in ustar_thresh_df indicating the quality of the U* threshold estimation (0=best, 1=acceptable, 2=poor)
    """
    _check_common_args(df, isday)
    if isinstance(seasons, int):
        if (seasons > 12 or seasons < 0):
            msg = f"must have  0 < seasons <= 12. Got {seasons}."
            raise ValueError(msg)
        elif seasons == 1:
            msg = f"n_seasons should be greater than 1, as turbulence conditions can change throughout the year. Got {seasons}."
            warnings.warn(msg)
    elif isinstance(seasons, list):
        for s in seasons:
            if not isinstance(s, list):
                msg = f"seasons must be a list of lists. Got {seasons}."
                raise TypeError(msg)
            for m in s:
                if not isinstance(m, int):
                    msg = f"seasons must be a list of lists of integers. Got {seasons}."
                    raise TypeError(msg)
                if m < 1 or m > 12:
                    msg = f"seasons must be a list of lists of integers between 1 and 12. Got {seasons}."
                    raise ValueError(msg)
    else:
        raise TypeError(f"seasons must be an int or a list of lists. Got {type(seasons)}")
    
    if n_ta_classes == 0:
        msg = f"n_ta_classes must be > 0, got {n_ta_classes}"
        raise ValueError(msg)
    elif n_ta_classes < 3:
        msg = f"n_ta_classes should be >= 3, got {n_ta_classes}. TA classes that are too large can lead to confounding of the U* ~ NEE relationship by air temperature. Recommended value is ~6."
        warnings.warn(msg)
    if ustar_ta_corr_cutoff > 1:
        msg = f"ustar_ta_corr_cutoff should be <= 1 to avoid air temperature from confounding the U* ~ NEE relationship, got {ustar_ta_corr_cutoff}. Results may be unreliable. Recommended value is |R| ~ 0.4"
        warnings.warn(msg)
    if gapfill_quantile < 0:
        msg = f"gapfill_quantile must be >= 0, got {gapfill_quantile}."
        raise ValueError(msg)
    elif gapfill_quantile > 1:
        msg = f"gapfill_quantile must be <= 1, got {gapfill_quantile}."
        raise ValueError(msg)
    if ustar_thresh_bounds[0] < 0:
        msg = f"ustar_thresh_bounds[0] must be >= 0, got {ustar_thresh_bounds[0]}."
        raise ValueError(msg)
    if ustar_thresh_bounds[1] <= ustar_thresh_bounds[0]:
        msg = f"ustar_thresh_bounds[1] must be > ustar_thresh_bounds[0], got {ustar_thresh_bounds}."
        raise ValueError(msg)
    if n_restarts <= 0:
        msg = f"n_restarts must be > 0, got {n_restarts}."
        raise ValueError(msg)
    if n_restarts < 3:
        msg = f"n_restarts should be >= 3, got {n_restarts}. Using too few bootstrap samples can lead to unstable estimates. Recommended value is ~30."
        warnings.warn(msg)
    n_restarts = int(n_restarts)
    if pbar:
        try:
            from tqdm import tqdm
        except ImportError:
            msg = "tqdm is not installed. Install it with `pip install tqdm` to use the progress bar."
            warnings.warn(msg)



    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()

    if isinstance(seasons, int):
        n_seasons = seasons
        seasons = month_to_season(df.index.month, n_seasons)
    else:
        n_seasons = len(seasons)
        seasons = np.select(
            [df.index.month.isin(s) for s in seasons],
            list(range(n_seasons)),
        )

    df_old = df.copy()
    df_old["season"] = seasons
    df = df_old.query(f"{ustar_col} < {ustar_thresh_bounds[1]*1.5} and {ustar_col} > {ustar_thresh_bounds[0]*0.5}").copy()
    
    # need about ~50 days worth of data in each season to get a reasonable estimate.
    min_night_samples_per_season_per_year = n_ta_classes*2*5
    dt = df.index[1] - df.index[0]
    samples_per_year = 365.25*86400 / dt.total_seconds()
    samples_per_season = samples_per_year / n_seasons
    samples_per_ta_class = samples_per_season / n_ta_classes
    samples_per_ustar_class = samples_per_ta_class
    if samples_per_ustar_class < 2*5:
        msg = f"Too many data classes! Each ustar class will only have {samples_per_ustar_class} samples per ta class per season per year! Must have at least 10, and ideally more. Consider decreasing: n_seasons (recommended: 4), n_ta_classes (recommended: 6), n_ustar_classes (recommended: 20)"
        raise ValueError(msg)

    # 1. select only nighttime data
    night_df = df.loc[~isday, [ta_col, ustar_col, nee_col, "season"]].dropna()

    # 3. split by year
    all_thresholds = np.full((len(np.unique(night_df.index.year)), n_seasons), np.nan)
    all_ustar_qual = np.full((len(np.unique(night_df.index.year)), n_seasons), 0)
    yearly_thresholds = np.full(all_thresholds.shape[0], np.nan)

    if pbar:
        pbar = tqdm(total=len(np.unique(night_df.index.year))*n_seasons*n_ta_classes, desc="Fitting U* thresholds")
    for iyr, (yr, yr_group) in enumerate(night_df.groupby(night_df.index.year)):
        if yr_group.shape[0] < 1:
            continue
        
        # 2. split into seasons
        seasons = yr_group["season"].values
        season_thresholds = np.full(n_seasons, np.nan)
        for issn, (ssn, ssn_group) in enumerate(yr_group.groupby("season")):
            if ssn_group.shape[0] < min_night_samples_per_season_per_year:
                all_ustar_qual[iyr, issn] = 2  # failed, will need to gap-fill
                continue
            
            # 3. split into TA classes of equal sample size
            ta_bins = pd.qcut(ssn_group[ta_col], q=n_ta_classes, labels=False)
            plateau_ustars = []
            for _, ta_group in ssn_group.groupby(ta_bins):
                plateau_ustars.append(
                    _fit_ta_class(ta_group, ta_col, ustar_col, nee_col, ustar_ta_corr_cutoff, ustar_thresh_bounds, n_restarts)
                )
            if any(np.isnan(plateau_ustars)):
                all_ustar_qual[iyr, issn] = 1  # medium quality estimate
            if all(np.isnan(plateau_ustars)):
                all_ustar_qual[iyr, issn] = 2  # failed, will need to gap-fill

            # 7. Median across TA classes: one threshold per season
            season_thresholds[issn] = np.nanmedian(plateau_ustars)
            if pbar:
                pbar.update(n_ta_classes)
        all_thresholds[iyr] = season_thresholds
        yearly_thresholds[iyr] = max(season_thresholds)

    ustar_thresh_df = (
        pd.DataFrame(all_thresholds)
        .set_index(np.unique(night_df.index.year))
    )
    ustar_qual_df = pd.DataFrame(all_ustar_qual, index=np.unique(night_df.index.year), columns=range(n_seasons))
    ustar_qual_df.index = ustar_qual_df.index.rename("Year")
    ustar_qual_df.columns = ustar_qual_df.columns.rename("Season")

    num_na = ustar_thresh_df.isna().sum().sum()
    if num_na / ustar_thresh_df.size > 0.5:
        warnings.warn(f"Warning! I was unable to determine a USTAR threshold for {num_na/ustar_thresh_df.size*100:.0f}% of the study period!")

    ustar_thresh_df.index = ustar_thresh_df.index.rename("Year")
    ustar_thresh_df.columns = ustar_thresh_df.columns.rename("Season")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_by_season = np.nanquantile(ustar_thresh_df, gapfill_quantile, axis=0)
        for issn, ssn in enumerate(ustar_thresh_df.columns):
            ustar_thresh_df.loc[:, ssn] = ustar_thresh_df.loc[:, ssn].fillna(max_by_season[issn])
        max_by_year = np.nanquantile(ustar_thresh_df, gapfill_quantile, axis=1)
        for iyr, yr in enumerate(ustar_thresh_df.index):
            ustar_thresh_df.loc[yr] = ustar_thresh_df.loc[yr].fillna(max_by_year[iyr])
    ustar_thresh_df = ustar_thresh_df.fillna(default_ustar_thresh)

    # build up list of flags
    ustar_flag = pd.DataFrame(index=df_old.index, columns=["ustar_flag", "season", "year"])
    ustar_flag["season"] = df_old["season"]
    ustar_flag["year"] = df_old.index.year
    ustar_flag["ustar_flag"] = True
    for group, gdf in ustar_flag.groupby(["year", "season"]):
        yr, ssn = group
        ustar_thresh = ustar_thresh_df.loc[yr, ssn]
        ustar_flag.loc[gdf.index, "ustar_flag"] = np.where(df_old.loc[gdf.index, ustar_col] <= ustar_thresh, False, True)
    ustar_flag = ustar_flag["ustar_flag"].astype(bool)
    return UStarFilterResult(flag=ustar_flag, thresholds=ustar_thresh_df, qual=ustar_qual_df.astype(int))

def ustar_papale_2006(
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str, ustar_col: str, nee_col: str,
    n_seasons: int=4,
    n_ta_classes: int=6, n_ustar_classes: int=20,
    ustar_ta_corr_cutoff: float=0.4,
    plateau_pct: float=0.95,
    gapfill_quantile: float=0.75,
    default_ustar_thresh: float=0.2,
)->UStarFilterResult:
    """
    Creates a U* filter for each season for each year in the dataset, following Papale et al (2006), Biogeosciences.

    General description of method:
    1. Select nighttime data based on a theoretical insolation threshold (`nighttime_swin`).
    2. For each year, for each season, bin the data by air temperature into `n_ta_classes` classes of equal sample size.
    3. Check the correlation between U* and TA. If |R(U*, TA)| > `ustar_ta_corr_cutoff`, skip the season.
    4. For each temperature class, bin the data by U* into `n_ustar_classes` classes of equal sample size.
    5. Identify the U* threshold: when U* plateaus within a temperature class (detected as U* being greater than `plateau_pct`*mean(USTAR) for all U* greater than in the current bin).
    6. If the algorithm fails for any particular season, fill in the U* threshold using the `gapfill_quantile` first by the seasonal mean, then by year annual mean if that also fails. If the entire pipeline fails, use `default_ustar_thresh`.
    7. Return a boolean series indicating which data points pass the U* filter and a dataframe of U* thresholds by season and year.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
        Must have:
            * columns for air temperature, U*, and nee/fc/co2_flux (positive-upwards)
            * have a `pd.DatetimeIndex`. This index should be a meridian offset time, not a civil time. e.g. Anchorage, AK, USA has the civil time zone of UTC-8 in the Summer and UTC-9 in the Winter, but has a meridian offset time zone of UTC-10 (calculated from its longitude).
    isday : pd.Series
        Boolean series indicating daytime (`True`) and nighttime (`False`) for each timestamp in `df.index`. utils.compute_isday can be used to generate this series.
    ta_col : str
        The column name for air temperature in `df`. Unit-agnostic.
    ustar_col : str
        The column name for ustar in `df`. m s-1 preferred.
    nee_col : str
        The column name for NEE (storage-corrected CO2 flux is preferred over turbulent CO2 flux alone) in `df`. Unit-agnostic.
    n_seasons : int
        The number of seasons in the year. Default 4. Increase this value if you want finer seasonal resolution for U* threshold estimation. Decrease this value if the algorithm has trouble converging on a solution.
    n_ta_classes : int
        The number of air temperature classes. Default 6. Increase this value to limit the impact of temperature variation on the final U* threshold. Decrease this value if the algorithm has trouble converging on a solution.
    n_ustar_classes : int
        The number of U* classes. Default 20. Increase this value for more precise determination of the U* threshold. Decrease this value if the algorithm has trouble converging on a solution.
    ustar_ta_corr_cutoff : float
        The maximum acceptable correlation between U* and TA. Seasons where |R(U*, TA)| > `ustar_ta_corr_cutoff` will be skipped to avoid confounding the U* ~ NEE relationship. Default 0.4. Increase this value to allow more correlation between U* and TA before skipping a season. Decrease this value to be more strict about the correlation threshold, however, it is recommended to also increase `n_ta_classes` to reduce the impact of temperature variation when the correlation threshold is looser.
    plateau_pct : float
        Criterion used to identify a plateau in the U* ~ NEE relationship. After binning seasonal data by TA, the data is binned by U*. If U* for a given bin is greater than `plateau_pct`*mean(USTAR) for all USTAR greater than in the current bin, then a plateau has been reached. Default 0.95.
    gapfill_quantile : float
        If the algorithm fails for a given season/year, U* thresholds are filled in with this quantile first by season, then by year. For example, if in 2012, the ustar thresholds for Winter, Spring, Summer, and Fall were 0.19, nan, 0.21, 0.3, and `gapfill_quantile`=0.75, the nan would be filled with the 0.75 quantile of [0.19, 0.21, 0.3]. Default 0.75.
    default_ustar_thresh : float
        If the entire pipeline fails, this is the fallback U* threshold value that will be used in the filter. Default 0.2.

    Returns
    -------
    UStarFilterResult
        A dataclass containing:
            * `flag`: pd.Series of type `bool`, indexed by `df.index`. `False` indicates that the datapoint should be filtered out.
            * `thresholds`: pd.DataFrame with columns representing seasons and rows representing years. Values indicate the U* threshold for that season and year.
            * `qual`: pd.DataFrame for each year/season in ustar_thresh_df indicating the quality of the U* threshold estimation (0=best, 1=acceptable, 2=poor)
    
    Important
    ---------
    Storage-corrected NEE is preferred for estimating U* thresholds because it provides a more complete representation of nighttime ecosystem-atmosphere CO2 exchange.
    When using storage-corrected NEE to determine U* thresholds, you should perform all future analyses using storage-corrected fluxes (e.g. FC + SC, H + SC, LE + SLE instead of FC, H, and LE by themselves).
    However, if you do not have access to high-quality storage flux measurements (for example, if your gas analyzer is infrequently calibrated), then using storage-corrected NEE may introduce more uncertainty than it resolves, and you may need to rely on turbulent fluxes alone for U* threshold estimation.
    """
    _check_common_args(df, isday)
    if n_seasons <= 0:
        msg = f"n_seasons must be > 0, got {n_seasons}"
        raise ValueError(msg)
    if n_seasons == 1:
        msg = f"n_seasons should be greater than 1, as turbulence conditions can change throughout the year. Got {n_seasons}."
        warnings.warn(msg)
    if n_ta_classes == 0:
        msg = f"n_ta_classes must be > 0, got {n_ta_classes}"
        raise ValueError(msg)
    elif n_ta_classes < 3:
        msg = f"n_ta_classes should be >= 3, got {n_ta_classes}. TA classes that are too large can lead to confounding of the U* ~ NEE relationship by air temperature. Recommended value is ~6."
        warnings.warn(msg)
    if n_ustar_classes < 2:
        msg = f"n_ustar_classes must be >= 2, got {n_ustar_classes}."
        raise ValueError(msg)
    elif n_ustar_classes < 5:
        msg = f"n_ustar_classes should be >= 5, got {n_ustar_classes}. Use a higher value to increase precision. Recommended value is ~20."
        warnings.warn(msg)
    if ustar_ta_corr_cutoff > 1:
        msg = f"ustar_ta_corr_cutoff should be <= 1 to avoid air temperature from confounding the U* ~ NEE relationship, got {ustar_ta_corr_cutoff}. Results may be unreliable. Recommended value is |R| ~ 0.4"
        warnings.warn(msg)
    if plateau_pct < 0:
        msg = f"plateau_pct should be >= 0, got {plateau_pct}."
        raise ValueError(msg)
    elif plateau_pct > 1:
        msg = f"plateau_pct should be <= 1, got {plateau_pct}."
        raise ValueError(msg)
    elif plateau_pct < 0.75:
        msg = f"plateau_pct should be >= 0.75, got {plateau_pct}. Using too low a value can result in underestimation of the U* threshold. Recommended value is ~0.95"
        warnings.warn(msg)
    if gapfill_quantile < 0:
        msg = f"gapfill_quantile must be >= 0, got {gapfill_quantile}."
        raise ValueError(msg)
    elif gapfill_quantile > 1:
        msg = f"gapfill_quantile must be <= 1, got {gapfill_quantile}."
        raise ValueError(msg)


    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()
    # need about ~50 days worth of data in each season to get a reasonable estimate.
    min_night_samples_per_season_per_year = n_ta_classes*n_ustar_classes*2*5
    dt = df.index[1] - df.index[0]
    samples_per_year = 365.25*86400 / dt.total_seconds()
    samples_per_season = samples_per_year / n_seasons
    samples_per_ta_class = samples_per_season / n_ta_classes
    samples_per_ustar_class = samples_per_ta_class / n_ustar_classes
    if samples_per_ustar_class < 2*5:
        msg = f"Too many data classes! Each ustar class will only have {samples_per_ustar_class} samples per ta class per season per year! Must have at least 10, and ideally more. Consider decreasing: n_seasons (recommended: 4), n_ta_classes (recommended: 6), n_ustar_classes (recommended: 20)"
        raise ValueError(msg)

    # 1. select only nighttime data
    night_df = df.loc[~isday, [ta_col, ustar_col, nee_col]].dropna()

    # 3. split by year
    all_thresholds = np.full((len(np.unique(night_df.index.year)), n_seasons), np.nan)
    all_ustar_qual = np.full((len(np.unique(night_df.index.year)), n_seasons), 0)
    yearly_thresholds = np.full(all_thresholds.shape[0], np.nan)
    for iyr, (yr, yr_group) in enumerate(night_df.groupby(night_df.index.year)):
        if yr_group.shape[0] < 1:
            continue
        
        # 2. split into seasons
        seasons = month_to_season(yr_group.index.month, n_seasons)
        season_thresholds = np.full(n_seasons, np.nan)
        for issn, (ssn, ssn_group) in enumerate(yr_group.groupby(seasons)):
            if ssn_group.shape[0] < min_night_samples_per_season_per_year:
                all_ustar_qual[iyr, issn] = 2  # failed, will need to gap-fill
                continue
            
            # 3. split into TA classes of equal sample size
            ta_bins = pd.qcut(ssn_group[ta_col], q=n_ta_classes, labels=False)
            plateau_ustars = []
            for ta_class, ta_group in ssn_group.groupby(ta_bins):
                
                # 4. correlation check: skip class if |R(TA<USTAR)| is too high
                corr = ta_group[ta_col].corr(ta_group[ustar_col])
                if abs(corr) >= ustar_ta_corr_cutoff:
                    warnings.warn(f"R(U*,TA) = {abs(corr):.2f} > {ustar_ta_corr_cutoff:.2f} for TA class [{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping")
                    all_ustar_qual[iyr, issn] = 1  # medium quality estimate
                    continue

                # 5. split into U* classes, equal sample size
                ustar_bins = pd.qcut(ta_group[ustar_col], q=n_ustar_classes, labels=False)
                plateau_ustar_candidates = []
                for ustar_class, ustar_group in ta_group.groupby(ustar_bins):

                    # 6. find plateau: lowest U* class where NEE >= plateau_pct% of mean NEE across all bins above it
                    max_bin_ustar = ustar_group[ustar_col].max()
                    mean_nee_above = ta_group.loc[ta_group[ustar_col] > max_bin_ustar, nee_col].mean()
                    mean_nee_bin = ustar_group[nee_col].mean()

                    if mean_nee_bin >= plateau_pct * mean_nee_above:
                        plateau_ustar_candidates.append(max_bin_ustar)

                if plateau_ustar_candidates:
                    plateau_ustars.append(min(plateau_ustar_candidates))

            # 7. Median across TA classes: one threshold per season
            season_thresholds[issn] = np.nanmedian(plateau_ustars)
        all_thresholds[iyr] = season_thresholds
        yearly_thresholds[iyr] = max(season_thresholds)

    ustar_thresh_df = (
        pd.DataFrame(all_thresholds)
        .set_index(np.unique(night_df.index.year))
    )
    ustar_qual_df = pd.DataFrame(all_ustar_qual, index=np.unique(night_df.index.year), columns=range(n_seasons))
    ustar_qual_df.index = ustar_qual_df.index.rename("Year")
    ustar_qual_df.columns = ustar_qual_df.columns.rename("Season")

    num_na = ustar_thresh_df.isna().sum().sum()
    if num_na / ustar_thresh_df.size > 0.5:
        warnings.warn(f"Warning! I was unable to determine a USTAR threshold for {num_na/ustar_thresh_df.size*100:.0f}% of the study period!")

    ustar_thresh_df.index = ustar_thresh_df.index.rename("Year")
    ustar_thresh_df.columns = ustar_thresh_df.columns.rename("Season")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_by_season = np.nanquantile(ustar_thresh_df, gapfill_quantile, axis=0)
        for issn, ssn in enumerate(ustar_thresh_df.columns):
            ustar_thresh_df.loc[:, ssn] = ustar_thresh_df.loc[:, ssn].fillna(max_by_season[issn])
        max_by_year = np.nanquantile(ustar_thresh_df, gapfill_quantile, axis=1)
        for iyr, yr in enumerate(ustar_thresh_df.index):
            ustar_thresh_df.loc[yr] = ustar_thresh_df.loc[yr].fillna(max_by_year[iyr])
    ustar_thresh_df = ustar_thresh_df.fillna(default_ustar_thresh)

    # build up list of flags
    ustar_flag = pd.Series(np.full(df.shape[0], True, dtype=bool), name="ustar_flag", index=df.index)
    for yr in ustar_thresh_df.index:
        for ssn in ustar_thresh_df.columns:
            ustar_thresh = ustar_thresh_df.loc[yr, ssn]
            ustar_flag.loc[
                (ustar_flag.index.year == yr) 
                & (month_to_season(ustar_flag.index.month, n_seasons) == ssn)
                & ((df[ustar_col] <= ustar_thresh) | (df[ustar_col].isna()))
            ] = False
    
    return UStarFilterResult(flag=ustar_flag, thresholds=ustar_thresh_df, qual=ustar_qual_df.astype(int))

__all__ = ["ustar_papale_2006", "ustar_broken_stick", "UStarFilterResult"]