import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from FluxFox import postproc

import numpy as np
import pandas as pd

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wd = Path(__file__).parent
    
    cpk = pd.read_parquet(wd / "data" / "cpk.parquet")
    float_cols = cpk.select_dtypes(include=["float"]).columns
    cpk.loc[:, float_cols] = cpk.loc[:, float_cols].astype("float32")
    lat, lon, elev = 41.3966, -106.8024, 2069.0
    
    
    
    
    start, end = "2008-12-31", "2009-12-31"
    cpk.loc[start:end, "FC"].plot(label="Original FC")

    # despike
    spike_flag = postproc.despike_mad_papale_2006(cpk, list(cpk), lat, lon, elev, z=4)
    cpk = cpk.where(spike_flag)
    cpk.loc[start:end, "FC"].plot(label="Despiked FC")

    # ustar filter
    ustar_flag = postproc.ustar_papale_2006(cpk, "TA", "USTAR", "FC", lat, lon, elev).flag
    cpk.loc[~ustar_flag, ["H", "LE", "FC"]] = np.nan
    cpk.loc[start:end, "FC"].plot(label="Ustar Filtered FC")

    # gapfill
    gapfill_result = postproc.gapfill_xgb_liu_2025(
        cpk, "FC",
        ta_col="TA",
        ppfd_col="PPFD_IN",
        vpd_col="VPD_PI",
        tsoil_col="TS_1_1_1",
        swc_col="SWC_1_1_1",
        extra_predictor_cols=["WD"],
        n_missing_allowed=1,
        random_state=8472,
        verbose=True,
        
        hyper_train_frac=0.1,  # 0.3
        hyper_test_frac=0.033,  # 0.1
        n_bayes_iter=10,  #50
        cv_folds=3,  # 5
    )
    cpk.loc[:, "FC"] = gapfill_result.filled
    cpk["FC"] = cpk["FC"] + cpk["FC"].quantile(0.5)
    partition, res = postproc.gpp_night_reichstein_2005(cpk, "FC", "TA", lat=lat, lon=lon, elev=elev, sw_thresh=0)
    partition.loc[start:end, ["GPP", "Reco"]].plot(label=["GPP", "Reco"])
    cpk.loc[start:end, "FC"].plot(label="NEE")
    plt.legend()
    plt.show()

    # gapfill_result = postproc.mds_gapfill_reichstein_2005(
    #     cpk, "FC",
    #     ta_col="TA",
    #     sw_in_col=None,
    #     vpd_col="VPD_PI",
    #     tol_sw=50,
    #     tol_ta=2.5,
    #     tol_vpd=0.5,
    #     max_window_days=28,
    #     min_samples=5,
    #     verbose=True
    # )

    
    # cpk.loc[start:end, "FC"].plot(label="Gapfilled FC", style='o', markersize=3)
    
    # from matplotlib import colors as mcolors
    # minmax = max(np.abs(np.nanquantile(cpk["FC"], 0.05)), np.abs(np.nanquantile(cpk["FC"], 0.95)))
    # norm = mcolors.TwoSlopeNorm(vmin=-minmax, vcenter=0, vmax=minmax)
    # ax = postproc.fingerprint_plot(cpk["FC"], norm=norm, cmap="RdBu", smear_days=7)
    plt.show()

    plt.scatter(cpk.loc[start:end, "TA"], partition.loc[start:end, "Reco"])
    plt.show()