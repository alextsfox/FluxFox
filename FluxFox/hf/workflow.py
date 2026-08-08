"""
Functions and classes for stringing together high frequency processing steps into a workflow.
"""

from pathlib import Path

import pandas as pd

from .timelag import *
from .rotate import RotatePlanarFit, PlanarFitResult

if __name__ == "__main__":
    files: list[Path] = []
    avg_len = pd.Timedelta("30min")
    file_len = pd.Timedelta("1D")
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

    irga_data_cols = [chi_c_col, chi_v_col, rho_c_col, rho_v_col, cell_temp_col, cell_press_col]
    irga_cols = irga_data_cols + [keep_irga_col]
    sonic_data_cols = [w_col, u_col, v_col, t_sonic_col]
    sonic_cols = sonic_data_cols + [keep_sonic_col]
    all_cols = irga_cols + sonic_cols


    # Planar fir arguments
    max_w = 1.0
    min_speed = 0.1
    assume_unbiased_w = False

    # Summarization Step
    summary_df = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=avg_len))
    lag_df = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=avg_len))
    for f in files:
        file_df = pd.read_feather(f)

        # despike/clean the data
        file_df[irga_cols] = file_df[irga_cols].where(file_df[keep_irga_col])
        file_df[sonic_cols] = file_df[sonic_cols].where(file_df[keep_sonic_col])

        # statistical summaries
        file_summary = file_df.resample(avg_len).agg({
            # need to rename the kurtosis column:
            x: ["mean", "std", "skew", ("kurt", lambda x: x.kurtosis()), ("med", "median"), "min", "max", "count"]
            for x in file_df.columns
        })
        file_summary.columns = file_summary.columns.map("_".join)
        summary_df.loc[file_summary.index] = file_summary.values

        # compute timelags for individual files
        file_lags = file_df.resample(avg_len).apply(lambda x: timelag_covmax(x, w_col)[1])
        lag_df.loc[file_lags.index] = file_lags.values
        lag_df["irga"] = file_lags[irga_data_cols].median(axis=1)
        file_lags[sonic_cols] = 0  # time lags for reference instruments is 0

    # compute rotation
    pf = RotatePlanarFit()
    pf_result = pf.compute(
        summary_df[[u_col, v_col, w_col]], u_bar_col=u_col + "_mean", v_bar_col=v_col + "_mean", w_bar_col=w_col + "_mean",
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

            # spectral corrections
            # compute spectral stuff

    # apply spectral stuff

            




            
            
        

