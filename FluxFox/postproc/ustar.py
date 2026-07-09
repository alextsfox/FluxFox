# Author: Alex Fox
# Created: 2026-06-17
"""
Creates a U* filter for a given dataset based on air temperature, U*, and NEE
"""


import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

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


def ustar_papale_2006(
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str, ustar_col: str, nee_col: str,
    nighttime_swin: float=20,
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
    6. If the algorithm fails for any particular season, fill in the U* threshold using the `gapfill_quantile` first by season, then by year. If the entire pipeline fails, use `default_ustar_thresh`.
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
        The column name for air temperature in `df`.
    ustar_col : str
        The column name for ustar in `df`.
    nee_col : str
        The column name for nee/fc/co2_flux in `df`.
    nighttime_swin : float
        The algorithm will simulate theoretical insolation on a flat plane, in W m-2. Whenever theoretical sw_in < `nighttime_swin`, the algorithm will assume it is nighttime and compute the ustar threshold on that data. Default 20 W m-2.
    n_seasons : int
        The number of seasons in the year. Default 4.
    n_ta_classes : int
        The number of air temperature classes. Default 6.
    n_ustar_classes : int
        The number of U* classes. Default 20.
    ustar_ta_corr_cutoff : float
        The maximum acceptable correlation between U* and TA. Seasons where |R(U*, TA)| > `ustar_ta_corr_cutoff` will be skipped to avoid confounding the U* ~ NEE relationship. Default 0.4
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
            * `ustar_flag`: pd.Series of type `bool`, indexed by `df.index`. `False` indicates that the datapoint should be filtered out.
            * `ustar_thresh_df`: pd.DataFrame with columns representing seasons and rows representing years. Values indicate the U* threshold for that season and year.
            * `qual`: pd.DataFrame for each year/season in ustar_thresh_df indicating the quality of the U* threshold estimation (0=best, 1=acceptable, 2=poor)
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
    if nighttime_swin < 0:
        msg = f"nighttime_swin must be >= 0, got {nighttime_swin}."
        raise ValueError(msg)
    elif nighttime_swin > 100:
        msg = f"nighttime_swin should be <= 100, got {nighttime_swin}. Recommended value is ~20 W m-2"
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
        for ssn, ssn_group in yr_group.groupby(seasons):
            if ssn_group.shape[0] < min_night_samples_per_season_per_year:
                all_ustar_qual[iyr, ssn-1] = 2  # failed, will need to gap-fill
                continue
            
            # 3. split into TA classes of equal sample size
            ta_bins = pd.qcut(ssn_group[ta_col], q=n_ta_classes, labels=False)
            plateau_ustars = []
            for ta_class, ta_group in ssn_group.groupby(ta_bins):
                
                # 4. correlation check: skip class if |R(TA<USTAR)| is too high
                corr = ta_group[ta_col].corr(ta_group[ustar_col])
                if abs(corr) >= ustar_ta_corr_cutoff:
                    warnings.warn(f"R(U*,TA) = {abs(corr):.2f} > {ustar_ta_corr_cutoff:.2f} for TA class [{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping")
                    all_ustar_qual[iyr, ssn-1] = 1  # medium quality estimate
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
            season_thresholds[ssn-1] = np.nanmedian(plateau_ustars)
        all_thresholds[iyr] = season_thresholds
        yearly_thresholds[iyr] = max(season_thresholds)

    ustar_thresh_df = (
        pd.DataFrame(all_thresholds)
        .set_index(np.unique(night_df.index.year))
        .rename(columns=lambda x: x+1)
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
        for i, ssn in enumerate(ustar_thresh_df.columns):
            ustar_thresh_df.loc[:, ssn] = ustar_thresh_df.loc[:, ssn].fillna(max_by_season[i])
        max_by_year = np.nanquantile(ustar_thresh_df, gapfill_quantile, axis=1)
        for i, yr in enumerate(ustar_thresh_df.index):
            ustar_thresh_df.loc[yr] = ustar_thresh_df.loc[yr].fillna(max_by_year[i])
    ustar_thresh_df = ustar_thresh_df.fillna(default_ustar_thresh)

    # build up list of flags
    ustar_flag = pd.Series(np.full(df.shape[0], True, dtype=bool), name="ustar_flag", index=df.index)
    for yr in ustar_thresh_df.index:
        for ssn in ustar_thresh_df.columns:
            ustar_thresh = ustar_thresh_df.loc[yr, ssn]
            ustar_flag.loc[
                (ustar_flag.index.year == yr) 
                & ((ustar_flag.index.month%12) // months_per_season == ssn) 
                & ((df[ustar_col] <= ustar_thresh) | (df[ustar_col].isna()))
            ] = False
    
    return UStarFilterResult(flag=ustar_flag, thresholds=ustar_thresh_df, qual=ustar_qual_df.astype(int))