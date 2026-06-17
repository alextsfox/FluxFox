# Author: Alex Fox
# Created: 2026-06-17
"""
Postprocessing gap-filling methods
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost_gapfill requires scikit-optimize. Install it with "
        "`pip install scikit-optimize`."
    ) from exc


__all__ = ["xgboost_gapfill", "GapfillResult"]


@dataclass
class GapfillResult:
    """Container for diagnostic info about a gap-filling run.

    Attributes
    ----------
    filled : pd.Series
        The gap-filled target column, indexed like the input ``df``.
    was_gapfilled : pd.Series (bool)
        True where the value was predicted/filled rather than observed.
    model : HistGradientBoostingRegressor
        The fitted model (with best hyperparameters from the search).
    best_params : dict
        Best hyperparameters found by BayesSearchCV.
    feature_names : list of str
        Names of the predictor columns used (including engineered
        time features), in the order passed to the model.
    train_scores : dict
        RMSE / R2 / bias on the main train/test split used to fit and
        sanity-check the final model.
    """

    filled: pd.Series
    was_gapfilled: pd.Series
    model: HistGradientBoostingRegressor
    best_params: dict
    feature_names: list = field(default_factory=list)
    train_scores: dict = field(default_factory=dict)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year sin/cos and a linear time index"""
    out = df.copy()
    doy = out.index.dayofyear.values.astype(float)
    days_in_year = np.where(out.index.is_leap_year, 366.0, 365.0)
    frac_year = doy / days_in_year
    out["_doy_sin"] = np.sin(2 * np.pi * frac_year)
    out["_doy_cos"] = np.cos(2 * np.pi * frac_year)
    # Linear timestamp feature, scaled to days since the first record,
    # so the magnitude is reasonable regardless of the absolute date.
    t0 = out.index[0]
    out["_timestamp"] = (out.index - t0).total_seconds() / 86400.0
    return out


def _default_search_space() -> dict:
    """A generic hyperparameter search space for HistGradientBoostingRegressor.

    NOTE: the paper's actual search space is reported only in
    supplementary Text S1, which was not available when writing this
    function. Adjust this space for your data size / compute budget.
    """
    return {
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "max_iter": Integer(100, 600),
        "max_depth": Integer(3, 12),
        "max_leaf_nodes": Integer(15, 127),
        "min_samples_leaf": Integer(5, 100),
        "l2_regularization": Real(1e-4, 10.0, prior="log-uniform"),
    }


def xgb_gapfill(
    df: pd.DataFrame,
    tgt_col: str,
    ta_col: Optional[str] = None,
    ppfd_col: Optional[str] = None,
    sw_in_col: Optional[str] = None,
    vpd_col: Optional[str] = None,
    tsoil_col: Optional[str] = None,
    swc_col: Optional[str] = None,
    ppfd_dir_col: Optional[str] = None,
    ppfd_dif_col: Optional[str] = None,
    gcc_col: Optional[str] = None,
    evi_col: Optional[str] = None,
    extra_predictor_cols: Optional[Sequence[str]] = None,
    n_missing_allowed: int = 0,
    hyper_train_frac: float = 0.3,
    hyper_test_frac: float = 0.1,
    train_frac: float = 0.8,
    n_bayes_iter: int = 50,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    search_space: Optional[dict] = None,
    verbose: bool = True,
) -> GapfillResult:
    """Gap-fill a flux time series using a gradient-boosted tree model.

    Based on Liu et al. (2025), "Robust filling of extra-long gaps in eddy covariance CO2 flux measurements from a temperate deciduous forest using eXtreme Gradient Boosting.", Agricultural and Forest Meteorology.

    Parameters
    ----------
    df : pd.DataFrame
        Must be indexed by a pandas DatetimeIndex
    tgt_col : str
        Name of the column in ``df`` to gap-fill (e.g. "FC" or "LE").
    ta_col, ppfd_col, sw_in_col, vpd_col, tsoil_col, swc_col,
    ppfd_dir_col, ppfd_dif_col, gcc_col, evi_col : str, optional
        Column names for air temperature, photosynthetic photon flux
        density, incoming shortwave radiation, vapor pressure deficit,
        soil temperature, soil water content, direct-beam PPFD,
        diffuse PPFD, PhenoCam GCC, and MODIS EVI, respectively. Any
        subset may be supplied; only the columns actually provided are
        used as predictors (plus engineered time features).
    extra_predictor_cols : sequence of str, optional
        Any additional columns in ``df`` to use as predictors, beyond
        the named meteorological/vegetation-index columns above (e.g., canopy temperature).
    n_missing_allowed : int, default 0
        Maximum number of missing predictor values tolerated in a row
        for that row to be used in training/fitting. Rows where the
        count of NaNs across the supplied predictor columns exceeds
        this value are dropped before training. (Missing *target*
        values are handled separately -- those rows define the gaps
        to be filled, and are never used for training.)
    hyper_train_frac, hyper_test_frac : float, default 0.3, 0.1
        Fraction of the available (non-gap) data used, respectively,
        for training and testing during the BayesSearchCV hyperparameter
        search. These need not sum to 1; any remainder is unused during
        the search (this keeps the search fast on large datasets). Must
        each be in (0, 1) and together sum to <= 1. Can be decreased to speed up the search,
        at the cost of potentially less optimal hyperparameters.
    train_frac : float, default 0.8
        Fraction of the available (non-gap) data used to fit the final
        model with the best hyperparameters found; the remainder is
        held out to report train_scores diagnostics.
    n_bayes_iter : int, default 50
        Number of parameter settings sampled by BayesSearchCV.
    cv_folds : int, default 5
        Number of cross-validation folds used internally by
        BayesSearchCV to score each candidate hyperparameter set on
        the hyperparameter-tuning subset.
    random_state : int, default 0
        Random seed for reproducibility of splits and model fitting.
    search_space : dict, optional
        Override the default skopt search space. Keys must be valid
        HistGradientBoostingRegressor parameter names; values must be
        skopt ``Dimension`` objects (e.g. ``Real``, ``Integer``).
        e.g. {"learning_rate": Real(0.01, 0.3, prior="log-uniform")}
    verbose : bool, default True
        Print basic progress information.

    Returns
    -------
    GapfillResult
        Dataclass containing the gap-filled series, a boolean
        "was gap-filled" indicator series, the fitted model, and
        diagnostic information. (See ``GapfillResult`` docstring.)
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        msg = f"df must be a pandas DataFrame. Got {type(df).__name__}"
        raise TypeError(msg)
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = f"df must be indexed by a pandas DatetimeIndex. Got index of type {type(df.index).__name__}"
        raise TypeError(msg)
    
    if tgt_col not in df.columns:
        raise ValueError(f"tgt_col '{tgt_col}' not found in df.columns")

    if not (0 < hyper_train_frac < 1):
        raise ValueError("hyper_train_frac must be in (0, 1)")
    if not (0 < hyper_test_frac < 1):
        raise ValueError("hyper_test_frac must be in (0, 1)")
    if hyper_train_frac + hyper_test_frac > 1:
        raise ValueError("hyper_train_frac + hyper_test_frac must be <= 1")
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1)")
    
    if random_state is None:
        rng = np.random.default_rng()
        random_state = rng.integers(0, 2**32 - 1)

    # build up predictor column set
    named_cols = {
        "ta": ta_col,
        "ppfd": ppfd_col,
        "sw_in": sw_in_col,
        "vpd": vpd_col,
        "tsoil": tsoil_col,
        "swc": swc_col,
        "ppfd_dir": ppfd_dir_col,
        "ppfd_dif": ppfd_dif_col,
        "gcc": gcc_col,
        "evi": evi_col,
    }
    predictor_cols = [c for c in named_cols.values() if c is not None]
    if extra_predictor_cols:
        predictor_cols.extend(extra_predictor_cols)
    if not predictor_cols:
        msg = "No predictor columns supplied. Provide at least one of ta_col, ppfd_col, sw_in_col, vpd_col, tsoil_col, swc_col, ppfd_dir_col, ppfd_dif_col, gcc_col, evi_col, or extra_predictor_cols."
        raise ValueError(msg)
    missing_requested = [c for c in predictor_cols if c not in df.columns]
    if missing_requested:
        msg = f"The following requested predictor columns are not in df: {missing_requested}"
        raise ValueError(msg)

    if not df.index.is_monotonic_increasing:
        msg = "df.index is not sorted in increasing order; sorting a copy before proceeding."
        warnings.warn(msg, stacklevel=2)
        df = df.sort_index()

    # ------------------------------------------------------------------
    # 2. Build the feature matrix (named predictors + engineered time
    #    features), and a boolean target-missing mask.
    # ------------------------------------------------------------------
    work = _add_time_features(df[predictor_cols + [tgt_col]])
    engineered_cols = ["_doy_sin", "_doy_cos", "_timestamp"]
    feature_cols = predictor_cols + engineered_cols

    # Rows usable for training: non-NaN target column and predictor NaN count is within tolerance
    target_missing = work[tgt_col].isna()
    predictor_nan_count = work[predictor_cols].isna().sum(axis=1)
    usable_for_training = (~target_missing) & (predictor_nan_count <= n_missing_allowed)
    n_total = len(work)
    n_gaps = int(target_missing.sum())
    n_usable = int(usable_for_training.sum())
    n_dropped = int((~target_missing).sum() - n_usable)

    if verbose:
        print(
            f"[xgb_gapfill] {n_total} rows total | "
            f"{n_gaps} target gaps to fill | "
            f"{n_usable} rows usable for training | "
            f"{n_dropped} non-gap rows dropped for excess missing predictors"
        )
    if n_usable < 10:
        raise ValueError(
            f"Only {n_usable} rows are usable for training after applying "
            "n_missing_allowed; cannot fit a model. Consider relaxing "
            "n_missing_allowed or checking predictor column coverage."
        )

    X_all = work.loc[usable_for_training, feature_cols]
    y_all = work.loc[usable_for_training, tgt_col]

    # ------------------------------------------------------------------
    # 3. Hyperparameter search via BayesSearchCV on a train/test subset
    #    carved out of the usable data (this deviates from Liu et al. 
    #    to reduce computation time)
    # ------------------------------------------------------------------
    space = search_space if search_space is not None else _default_search_space()

    hyper_combined_frac = hyper_train_frac + hyper_test_frac
    X_hyper_pool, _, y_hyper_pool, _ = train_test_split(
        X_all,
        y_all,
        train_size=hyper_combined_frac,
        random_state=random_state,
        shuffle=True,
    )
    # Split the pool into the requested train/test proportions.
    relative_train_frac = hyper_train_frac / hyper_combined_frac
    X_hyper_train, X_hyper_test, y_hyper_train, y_hyper_test = train_test_split(
        X_hyper_pool,
        y_hyper_pool,
        train_size=relative_train_frac,
        random_state=random_state,
        shuffle=True,
    )

    if verbose:
        print(
            f"[xgb_gapfill] hyperparameter search: "
            f"{len(X_hyper_train)} train / {len(X_hyper_test)} test rows, "
            f"{n_bayes_iter} Bayes iterations, {cv_folds}-fold CV scoring"
        )

    base_model = HistGradientBoostingRegressor(
        random_state=random_state,
        early_stopping=True,
    )

    opt = BayesSearchCV(
        base_model,
        search_spaces=space,
        n_iter=n_bayes_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )


    if verbose:
        # BayesSearchCV.fit gives internal CV scores on X_hyper_train
        # we separately evaluate the best best estimator on the held-out
        # X_hyper_test to confirm the chosen hyperparameters generalize
        opt.fit(X_hyper_train, y_hyper_train)
        best_params = dict(opt.best_params_)

        hyper_test_pred = opt.predict(X_hyper_test)
        hyper_test_rmse = float(
            np.sqrt(np.mean((hyper_test_pred - y_hyper_test.values) ** 2))
        )
        print(f"[xgb_gapfill] best hyperparameters: {best_params}")
        print(
            f"[xgb_gapfill] held-out hyper_test RMSE: {hyper_test_rmse:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Fit the final model with the best hyperparameters on a fresh
    #    train/test split (train_frac) drawn from the full usable pool.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        train_size=train_frac,
        random_state=random_state,
        shuffle=True,
    )

    final_model = HistGradientBoostingRegressor(
        random_state=random_state,
        early_stopping=True,
        **best_params,
    )
    final_model.fit(X_train, y_train)

    test_pred = final_model.predict(X_test)
    residuals = test_pred - y_test.values
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_test.values - y_test.values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bias = float(np.mean(residuals))

    train_scores = {"rmse": rmse, "r2": r2, "bias": bias, "n_test": len(y_test)}

    if verbose:
        print(
            f"[xgb_gapfill] final model test scores: "
            f"RMSE={rmse:.4f}, R2={r2:.4f}, bias={bias:.4f} "
            f"(n_test={len(y_test)})"
        )

    # Refit the final model on ALL usable rows (not just X_train) so
    # that gap predictions benefit from the maximum available training
    # data, now that hyperparameters and diagnostics are settled.
    if verbose:
        print(f"[xgb_gapfill] refitting final model on all usable rows")
    final_model_full = HistGradientBoostingRegressor(
        random_state=random_state,
        early_stopping=True,
        **best_params,
    )
    final_model_full.fit(X_all, y_all)

    # ------------------------------------------------------------------
    # 5. Predict for all gap rows and assemble the output series.
    # ------------------------------------------------------------------
    filled = work[tgt_col].copy()
    was_gapfilled = pd.Series(False, index=df.index)

    if n_gaps > 0:
        gap_idx = work.index[target_missing]
        predictor_nan_count_gaps = work.loc[gap_idx, predictor_cols].isna().sum(axis=1)
        predictable_gap_idx = gap_idx[predictor_nan_count_gaps <= n_missing_allowed]
        unpredictable_gap_idx = gap_idx[predictor_nan_count_gaps > n_missing_allowed]

        if len(unpredictable_gap_idx) > 0 and verbose:
            msg = f"[xgb_gapfill] WARNING: {len(unpredictable_gap_idx)} gap rows exceed the n_missing_allowed={n_missing_allowed} predictor NaN tolerance and cannot be filled; they remain NaN."
            warnings.warn(msg)
    
        if len(predictable_gap_idx) > 0:
            X_gaps = work.loc[predictable_gap_idx, feature_cols]
            gap_pred = final_model_full.predict(X_gaps).astype("float32")
            filled.loc[predictable_gap_idx] = gap_pred
            was_gapfilled.loc[predictable_gap_idx] = True

    filled.name = tgt_col
    was_gapfilled.name = f"{tgt_col}_F"

    return GapfillResult(
        filled=filled,
        was_gapfilled=was_gapfilled,
        model=final_model_full,
        best_params=best_params,
        feature_names=feature_cols,
        train_scores=train_scores,
    )

__all__ = [
    "xgb_gapfill"
]