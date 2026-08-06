# Author: Alex Fox
# Created: 2026-08-06
"""U* threshold estimation and filtering methods."""

import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils import _check_common_args, month_to_season


@dataclass
class UStarFilterResult:
    """Container for U* filter output and diagnostics."""

    flag: pd.Series
    thresholds: pd.DataFrame
    qual: pd.DataFrame


def _broken_stick(x, intercept, slope, x0):
    return np.where(x < x0, intercept + slope * x, intercept + slope * x0)


def _mae_loss(params, x, y):
    intercept, slope, x0 = params
    return np.mean(np.abs(y - _broken_stick(x, intercept, slope, x0)))


def _mse_loss(params, x, y):
    intercept, slope, x0 = params
    return np.mean((y - _broken_stick(x, intercept, slope, x0)) ** 2)


def _fit_broken_stick(
    x,
    y,
    p0=(0.0, 15.0, 0.2),
    n_restarts=30,
    ustar_thresh_bounds=(0, 1),
    loss: Literal["mse", "mae"] = "mse",
    random_state=None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    bounds = [(None, None), (0.0, None), ustar_thresh_bounds]
    x0s = []
    intercepts = []
    slopes = []
    # Vary the initial breakpoint to reduce sensitivity to local optimizer failures.
    for ustar_thresh_init in (
        rng.random(n_restarts)
        * (ustar_thresh_bounds[1] - ustar_thresh_bounds[0])
        + ustar_thresh_bounds[0]
    ):
        objective = _mae_loss if loss == "mae" else _mse_loss
        result = minimize(
            objective,
            (p0[0], p0[1], ustar_thresh_init),
            args=(x, y),
            bounds=bounds,
            method="L-BFGS-B",
        )
        intercept, slope, x0 = result.x
        if (
            not result.success
            or x0 == ustar_thresh_bounds[0]
            or x0 == ustar_thresh_bounds[1]
        ):
            intercept = slope = x0 = np.nan
        intercepts.append(intercept)
        slopes.append(slope)
        x0s.append(x0)
    if all(np.isnan(x0s)):
        return np.nan, np.nan, np.nan
    return np.nanmedian(intercepts), np.nanmedian(slopes), np.nanmean(x0s)


def _fit_ta_class_broken_stick(
    ta_group: pd.DataFrame,
    ta_col: str,
    ustar_col: str,
    nee_col: str,
    ustar_ta_corr_cutoff: float,
    ustar_thresh_bounds: tuple[float, float],
    n_restarts: int,
    random_state,
    loss: Literal["mse", "mae"],
) -> tuple[float, str | None]:
    # Exclude temperature classes where U* and temperature remain confounded.
    corr = ta_group[ta_col].corr(ta_group[ustar_col])
    if abs(corr) >= ustar_ta_corr_cutoff:
        message = (
            "[ustar_broken_stick] WARNING: "
            f"R(U*,TA) = {abs(corr):.2f} > {ustar_ta_corr_cutoff:.2f} for TA class "
            f"[{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping\n"
        )
        return np.nan, message
    _, _, threshold = _fit_broken_stick(
        ta_group[ustar_col].to_numpy(),
        ta_group[nee_col].to_numpy(),
        p0=(0.0, 15.0, max(0.2, ustar_thresh_bounds[1])),
        n_restarts=n_restarts,
        ustar_thresh_bounds=ustar_thresh_bounds,
        random_state=random_state,
        loss=loss,
    )
    if np.isnan(threshold):
        message = (
            "[ustar_broken_stick] WARNING: Failed to converge for TA class "
            f"[{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping\n"
        )
        return threshold, message
    return threshold, None


def _fit_ta_class_papale_2005(
    ta_group: pd.DataFrame,
    ta_col: str,
    ustar_col: str,
    nee_col: str,
    ustar_ta_corr_cutoff: float,
    n_ustar_classes: int,
    plateau_pct: float,
) -> tuple[float, str | None]:
    # Apply the same temperature-confounding check used by the broken-stick fit.
    corr = ta_group[ta_col].corr(ta_group[ustar_col])
    if abs(corr) >= ustar_ta_corr_cutoff:
        message = (
            "[ustar_papale_2006] WARNING: "
            f"R(U*,TA) = {abs(corr):.2f} > {ustar_ta_corr_cutoff:.2f} for TA class "
            f"[{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping\n"
        )
        return np.nan, message
    # Search U* quantiles for the first bin that reaches the upper-bin NEE plateau.
    ustar_bins = pd.qcut(ta_group[ustar_col], q=n_ustar_classes, labels=False)
    candidates = []
    for _, ustar_group in ta_group.groupby(ustar_bins, observed=True):
        max_bin_ustar = ustar_group[ustar_col].max()
        mean_nee_above = ta_group.loc[
            ta_group[ustar_col] > max_bin_ustar, nee_col
        ].mean()
        if ustar_group[nee_col].mean() >= plateau_pct * mean_nee_above:
            candidates.append(max_bin_ustar)
    if candidates:
        return min(candidates), None
    message = (
        "[ustar_papale_2006] WARNING: Failed to find a plateau for TA class "
        f"[{ta_group[ta_col].min():.2f}, {ta_group[ta_col].max():.2f}]. Skipping\n"
    )
    return np.nan, message


def _season_labels(
    index: pd.DatetimeIndex, seasons: int | list[list[int]]
) -> tuple[np.ndarray, int]:
    if isinstance(seasons, int):
        if not 0 < seasons <= 12:
            raise ValueError(f"must have 0 < seasons <= 12. Got {seasons}.")
        if 12 % seasons:
            raise ValueError(
                "seasons must evenly divide the 12-month calendar when specified "
                f"as an integer. Got {seasons}."
            )
        return month_to_season(index.month, seasons) - 1, seasons
    if not isinstance(seasons, list):
        raise TypeError(f"seasons must be an int or a list of lists. Got {type(seasons)}")
    for season in seasons:
        if not isinstance(season, list):
            raise TypeError(f"seasons must be a list of lists. Got {seasons}.")
        if any(not isinstance(month, int) for month in season):
            raise TypeError(f"seasons must be a list of lists of integers. Got {seasons}.")
        if any(month < 1 or month > 12 for month in season):
            raise ValueError(
                "seasons must be a list of lists of integers between 1 and 12. "
                f"Got {seasons}."
            )
    season_coverage = sorted([month for season in seasons for month in season])
    if set(season_coverage) != set(range(1, 13)):
        raise ValueError(
            f"seasons must cover all months from 1 to 12. Got {seasons} which covers {season_coverage}."
        )
    if len(season_coverage) != 12:
        raise ValueError(
            "seasons must not have overlapping months. "
            f"Got {seasons} which covers {season_coverage}."
        )
    return (
        np.select(
            [index.month.isin(season) for season in seasons], range(len(seasons))
        ),
        len(seasons),
    )


def _emit_warnings(warn_msg: str) -> None:
    if warn_msg:
        warnings.warn("\n" + warn_msg.rstrip("\n"), stacklevel=3)
        sys.stderr.flush()


def _ustar_filter(
    fit_ta_class: Callable[..., tuple[float, str | None]],
    fit_ta_class_kwargs: dict,
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str,
    ustar_col: str,
    nee_col: str,
    seasons: int | list[list[int]],
    n_ta_classes: int,
    gapfill_quantile: float,
    default_ustar_thresh: float,
    min_samples_per_ta_class: int,
    pbar: bool,
    fit_ustar_bounds: tuple[float, float] | None = None,
    initial_warnings: str = "",
) -> UStarFilterResult:
    _check_common_args(df, isday)
    if n_ta_classes <= 0:
        raise ValueError(f"n_ta_classes must be > 0, got {n_ta_classes}")
    if not 0 <= gapfill_quantile <= 1:
        raise ValueError(f"gapfill_quantile must be between 0 and 1, got {gapfill_quantile}.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        isday = isday.sort_index()

    season_labels, n_seasons = _season_labels(df.index, seasons)
    if n_seasons == 1:
        initial_warnings += (
            "[ustar] WARNING: n_seasons should be greater than 1, as turbulence "
            "conditions can change throughout the year. Got 1.\n"
        )
    if n_ta_classes < 3:
        initial_warnings += (
            "[ustar] WARNING: n_ta_classes should be >= 3. TA classes that are "
            "too large can confound the U* ~ NEE relationship. Recommended value is ~6.\n"
        )
    _emit_warnings(initial_warnings)

    # Preserve all observations for final flags; optionally restrict only the fit data.
    full_df = df.copy()
    full_df["season"] = season_labels
    if len(full_df) < 2:
        raise ValueError("df must contain at least two timestamps to infer its sampling interval.")
    dt = full_df.index[1] - full_df.index[0]
    if dt <= pd.Timedelta(0):
        raise ValueError("df timestamps must have a positive sampling interval.")
    fit_df = full_df
    if fit_ustar_bounds is not None:
        lower, upper = fit_ustar_bounds
        fit_df = full_df.loc[
            full_df[ustar_col].between(lower * 0.5, upper * 1.5, inclusive="neither")
        ].copy()
    if len(fit_df) < 2:
        raise ValueError(
            "Fewer than two observations remain after applying the U* fitting range."
        )
    # Estimate expected samples per class from the original sampling cadence.
    samples_per_year = 365.25 * 86400 / dt.total_seconds()
    if samples_per_year / n_seasons / n_ta_classes / min_samples_per_ta_class < 1:
        raise ValueError(
            "Too many data classes! Each U* class must have at least 10 samples. "
            "Consider decreasing n_seasons, n_ta_classes, or n_ustar_classes."
        )
    # Thresholds are estimated from complete nighttime TA/U*/NEE observations only.
    night_df = fit_df.loc[
        ~isday.loc[fit_df.index], [ta_col, ustar_col, nee_col, "season"]
    ].dropna()
    years = np.unique(night_df.index.year)
    thresholds = np.full((len(years), n_seasons), np.nan)
    # A cell remains poor quality until a seasonal fit provides evidence otherwise.
    quality = np.full(thresholds.shape, 2, dtype=int)
    warn_msg = ""

    progress = None
    if pbar:
        try:
            from tqdm import tqdm
        except ImportError:
            warn_msg += "[ustar] WARNING: tqdm is not installed; progress bar disabled.\n"
        else:
            progress = tqdm(total=len(years) * n_seasons * n_ta_classes, desc="Fitting U* thresholds")

    # Fit one seasonal threshold per year from the median of its TA-class estimates.
    for year_index, (year, year_group) in enumerate(night_df.groupby(night_df.index.year)):
        for season, season_group in year_group.groupby("season", observed=True):
            if len(season_group) < n_ta_classes * min_samples_per_ta_class:
                quality[year_index, season] = 2
                if progress:
                    progress.update(n_ta_classes)
                continue
            ta_bins = pd.qcut(season_group[ta_col], q=n_ta_classes, labels=False)
            class_thresholds = []
            for _, ta_group in season_group.groupby(ta_bins, observed=True):
                threshold, message = fit_ta_class(
                    ta_group, ta_col, ustar_col, nee_col, **fit_ta_class_kwargs
                )
                class_thresholds.append(threshold)
                if message:
                    warn_msg += message
            if any(np.isnan(class_thresholds)):
                quality[year_index, season] = 1
            if all(np.isnan(class_thresholds)):
                quality[year_index, season] = 2
                thresholds[year_index, season] = np.nan
            else:
                if not any(np.isnan(class_thresholds)):
                    quality[year_index, season] = 0
                thresholds[year_index, season] = np.nanmedian(class_thresholds)
            if progress:
                progress.update(n_ta_classes)
    if progress:
        progress.close()

    threshold_df = pd.DataFrame(thresholds, index=years, columns=range(n_seasons))
    threshold_df.index.name = "Year"
    threshold_df.columns.name = "Season"
    quality_df = pd.DataFrame(quality, index=years, columns=range(n_seasons))
    quality_df.index.name = "Year"
    quality_df.columns.name = "Season"
    num_missing = threshold_df.isna().sum().sum()
    if threshold_df.size and num_missing / threshold_df.size > 0.5:
        warn_msg += (
            "[ustar] WARNING: Unable to determine a USTAR threshold for "
            f"{num_missing / threshold_df.size * 100:.0f}% of the study period!\n"
        )
    # Fill missing fits conservatively by season, then by year, then by the fallback.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seasonal_fill = np.nanquantile(threshold_df, gapfill_quantile, axis=0)
        threshold_df = threshold_df.fillna(pd.Series(seasonal_fill, index=threshold_df.columns))
        annual_fill = np.nanquantile(threshold_df, gapfill_quantile, axis=1)
        for year, fill_value in zip(threshold_df.index, annual_fill):
            threshold_df.loc[year] = threshold_df.loc[year].fillna(fill_value)
    threshold_df = threshold_df.fillna(default_ustar_thresh)

    # Apply each year/season threshold back to the complete, unfiltered input.
    flags = pd.Series(True, index=full_df.index, name="ustar_flag")
    for (year, season), group in full_df.groupby([full_df.index.year, "season"], observed=True):
        if year in threshold_df.index:
            flags.loc[group.index] = group[ustar_col] > threshold_df.loc[year, season]
    _emit_warnings(warn_msg)
    return UStarFilterResult(flag=flags.astype(bool), thresholds=threshold_df, qual=quality_df)


def ustar_broken_stick(
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str,
    ustar_col: str,
    nee_col: str,
    seasons: int | list[list[int]] = 4,
    n_ta_classes: int = 6,
    ustar_ta_corr_cutoff: float = 0.4,
    ustar_thresh_bounds: tuple[float, float] = (0, 1),
    n_restarts: int = 10,
    loss: Literal["mse", "mae"] = "mse",
    gapfill_quantile: float = 0.75,
    default_ustar_thresh: float = 0.2,
    pbar: bool = False,
    random_state=None,
) -> UStarFilterResult:
    if ustar_thresh_bounds[0] < 0 or ustar_thresh_bounds[1] <= ustar_thresh_bounds[0]:
        raise ValueError(f"ustar_thresh_bounds must be increasing and non-negative, got {ustar_thresh_bounds}.")
    if not isinstance(n_restarts, (int, np.integer)) or isinstance(n_restarts, bool):
        raise TypeError(f"n_restarts must be an integer, got {n_restarts}.")
    if n_restarts <= 0:
        raise ValueError(f"n_restarts must be > 0, got {n_restarts}.")
    initial_warnings = ""
    if ustar_ta_corr_cutoff > 1:
        initial_warnings += "[ustar_broken_stick] WARNING: ustar_ta_corr_cutoff should be <= 1.\n"
    if n_restarts < 3:
        initial_warnings += "[ustar_broken_stick] WARNING: n_restarts should be >= 3.\n"
    return _ustar_filter(
        _fit_ta_class_broken_stick,
        {
            "ustar_ta_corr_cutoff": ustar_ta_corr_cutoff,
            "ustar_thresh_bounds": ustar_thresh_bounds,
            "n_restarts": n_restarts,
            "random_state": random_state,
            "loss": loss,
        },
        df, isday, ta_col, ustar_col, nee_col, seasons, n_ta_classes,
        gapfill_quantile, default_ustar_thresh, 10, pbar,
        fit_ustar_bounds=ustar_thresh_bounds,
        initial_warnings=initial_warnings,
    )


def ustar_papale_2006(
    df: pd.DataFrame,
    isday: pd.Series,
    ta_col: str,
    ustar_col: str,
    nee_col: str,
    seasons: int | list[list[int]] = 4,
    n_ta_classes: int = 6,
    n_ustar_classes: int = 20,
    ustar_ta_corr_cutoff: float = 0.4,
    plateau_pct: float = 0.95,
    gapfill_quantile: float = 0.75,
    default_ustar_thresh: float = 0.2,
    pbar: bool = False,
) -> UStarFilterResult:
    if n_ustar_classes < 2:
        raise ValueError(f"n_ustar_classes must be >= 2, got {n_ustar_classes}.")
    if not 0 <= plateau_pct <= 1:
        raise ValueError(f"plateau_pct must be between 0 and 1, got {plateau_pct}.")
    initial_warnings = ""
    if n_ustar_classes < 5:
        initial_warnings += "[ustar_papale_2006] WARNING: n_ustar_classes should be >= 5.\n"
    if plateau_pct < 0.75:
        initial_warnings += "[ustar_papale_2006] WARNING: plateau_pct should be >= 0.75.\n"
    if ustar_ta_corr_cutoff > 1:
        initial_warnings += "[ustar_papale_2006] WARNING: ustar_ta_corr_cutoff should be <= 1.\n"
    return _ustar_filter(
        _fit_ta_class_papale_2005,
        {
            "ustar_ta_corr_cutoff": ustar_ta_corr_cutoff,
            "n_ustar_classes": n_ustar_classes,
            "plateau_pct": plateau_pct,
        },
        df, isday, ta_col, ustar_col, nee_col, seasons, n_ta_classes,
        gapfill_quantile, default_ustar_thresh, n_ustar_classes * 10, pbar,
        initial_warnings=initial_warnings,
    )


__all__ = ["UStarFilterResult", "ustar_broken_stick", "ustar_papale_2006"]