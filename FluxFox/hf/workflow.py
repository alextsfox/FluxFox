"""
Functions and classes for stringing together high frequency processing steps into a workflow.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from .timelag import TimelagAutoOptFratini, TimelagAutoOptFratiniResult, timelag_covmax
from .rotate import RotatePlanarFit, PlanarFitResult
from .clean import despike_mad

if __name__ == "__main__":
    files: list[Path] = []
    avg_len = pd.Timedelta("30min")
    file_len = pd.Timedelta("1D")
    acq_freq = 20
    start_time = pd.Timestamp("2023-01-01")
    end_time = pd.Timestamp("2023-01-02")
    chi_c_col = "chi_c"
    chi_v_col = "chi_v"
    rho_c_col = "rho_c"
    rho_v_col = "rho_v"
    w_col = "w"
    u_col = "u"
    v_col = "v"
    t_sonic_col = "t_sonic"
    keep_sonic_col = "diag_sonic"
    keep_irga_col = "diag_irga"
    cell_temp_col = "cell_temp"
    cell_press_col = "cell_press"

    chi_c_lims = (0, 10_000)
    chi_v_lims = (0, 1_000)
    rho_c_lims = (0, 10_000)
    rho_v_lims = (0, 1_000)
    w_lims = (-100, 100)
    u_lims = (-100, 100)
    v_lims = (-100, 100)
    t_sonic_lims = (-100, 100)

    irga_data_cols = [chi_c_col, chi_v_col, rho_c_col, rho_v_col, cell_temp_col, cell_press_col]
    irga_cols = irga_data_cols + [keep_irga_col]
    sonic_data_cols = [w_col, u_col, v_col, t_sonic_col]
    sonic_cols = sonic_data_cols + [keep_sonic_col]
    data_cols = irga_data_cols + sonic_data_cols
    all_cols = irga_cols + sonic_cols

    # Despiker arguments
    z_thresh = 3
    chunksize = 20*300
    interp_limit = 3

    # Planar fir arguments
    max_w = 1.0
    min_speed = 0.1
    assume_unbiased_w = False

    # Summarization Step
    summary_df: pd.DataFrame = None
    lag_df = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=avg_len))
    for f in files:
        file_df = pd.read_feather(f)

        # compute raw file summary before anything else happens
        file_summary_raw = file_df.resample(avg_len).agg({
            # need to rename the kurtosis column:
            x: ["mean", "std", "skew", ("kurt", lambda x: x.kurtosis()), ("med", "median"), "min", "max", "count"]
            for x in file_df.columns
        })
        file_summary_raw = file_summary_raw.rename(columns={c:f"{c}_raw" for c in file_summary_raw})

        num_na_raw = file_df[data_cols].resample(avg_len).isna().sum().to_frame()
        num_na_raw = num_na_raw.rename(columns={c: f"{c}_raw_count" for c in num_na_raw})

        # filter for diagnostics
        file_df[irga_cols] = file_df[irga_cols].where(file_df[keep_irga_col])
        file_df[sonic_cols] = file_df[sonic_cols].where(file_df[keep_sonic_col])

        num_na_post_diag = file_df[data_cols].resample(avg_len).isna().sum().to_frame()
        num_diag_removed = num_na_post_diag - num_na_raw
        num_diag_removed = num_diag_removed.rename(columns={c: f"{c}_n_diag_removed" for c in num_diag_removed})

        # despike
        file_df[data_cols] = despike_mad(file_df[data_cols], z_thresh=z_thresh, chunksize=chunksize)

        num_na_post_despike = file_df[data_cols].resample(avg_len).isna().sum().to_frame()
        num_despike_removed = num_na_post_despike - num_na_post_diag
        num_despike_removed = num_despike_removed.rename(columns={c: f"{c}_n_despike_removed" for c in num_despike_removed})

        # *light* interpolation
        file_df[data_cols] = file_df[data_cols].interpolate(method="pchip", limit=interp_limit, limit_direction="both", limit_area="inside")

        num_removed = num_diag_removed - num_na_raw
        num_removed = num_removed.rename(columns={c: f"{c}_n_removed" for c in num_despike_removed})

        # statistical summaries
        file_summary = file_df.resample(avg_len).agg({
            # need to rename the kurtosis column:
            x: ["mean", "std", "skew", ("kurt", lambda x: x.kurtosis()), ("med", "median"), "min", "max", "count"]
            for x in file_df.columns
        })
        file_summary.columns = file_summary.columns.map("_".join)
        na_count_df = pd.concat([num_na_raw, num_diag_removed, num_despike_removed, num_removed], axis=1)


        # compute timelags for individual files
        file_lags = file_df.resample(avg_len).apply(lambda x: timelag_covmax(x, w_col)[1])
        lag_df.loc[file_lags.index] = file_lags.values
        lag_df["irga"] = file_lags[irga_data_cols].median(axis=1)  # TODO: what if we have multiple instruments? we should group by instrument
        file_lags[sonic_cols] = 0  # time lags for reference instruments is 0
        file_lags = file_lags.rename({c:f"{c}_lag_idx" for c in file_lags})

        # now that we have the full file summary dataframe, create the overall summary dataframe if we haven't yet.
        file_summary = pd.concat([file_summary, file_summary_raw, file_lags, na_count_df])
        file_summary = file_summary.rename({c:f"{c}_unrot" for c in [u_col, v_col, w_col]})
        if summary_df is None:
            summary_df = pd.DataFrame(index=pd.date_range(start_time, end_time, freq=avg_len), columns=file_summary.columns)
        summary_df.loc[file_summary.index] = file_summary.values

    # compute rotation
    pf = RotatePlanarFit()
    pf_result = pf.compute(
        summary_df[[u_col + "_mean_unrot", v_col + "_mean_unrot", w_col + "_mean_unrot"]], u_bar_col=u_col + "_mean_unrot", v_bar_col=v_col + "_mean_unrot", w_bar_col=w_col + "_mean_unrot",
        max_w=max_w, min_speed=min_speed, assume_unbiased_w=assume_unbiased_w
    )

    # compute optimal time lag
    tau_optimizer = TimelagAutoOptFratini()
    tau_opt_result = tau_optimizer.compute(lag_series = lag_df[rho_c_col])

    for f in files:
        file_df = pd.read_feather(f)

        for start, df in file_df.resample(avg_len):
            if df.empty:
                continue

            # apply rotation
            df[[u_col, v_col, w_col]] = pf.rotate(
                df[[u_col, v_col, w_col]], 
                u_bar_col=u_col, v_bar_col=v_col, w_bar_col=w_col
            )

            # apply time lag
            df[irga_data_cols] = tau_opt_result.apply(df[irga_data_cols])

            # compute all covariances and variances
            # cov(w, x) for x in data columns
            # TODO: MAKING A one-row dataframe is very inefficient.
            # compute cov(w,x), var(x), and cov(u, v).
            cov_df = pd.DataFrame(
                {
                    f'cov_w_{c}': np.cov(df[[w_col, c]].dropna().to_numpy(), rowvar=False, ddof=0)[0, 1] 
                    for c in [dc for dc in data_cols if dc != w_col]
                }.update({
                    f'var_{c}': np.var(df[c].dropna().to_numpy(), ddof=0) for c in data_cols
                }),
                index=[start]
            )
            cov_df[f"cov_{u_col}_{v_col}"] = np.cov(df[[u_col, v_col]].dropna().to_numpy(), rowvar=False, ddof=0)[0, 1]

            # spectral corrections
            # compute the spectra identically to cov...just as a full spectrum. Perhaps use xarray for this now?
            
            # compute spectral stuff

    # apply spectral stuff

            




            
            
        

