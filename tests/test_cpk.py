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
    
    start, end = "2009-12-01", "2009-12-31"
    cpk.loc[start:end, "FC"].plot(label="Original FC")

    # despike
    spike_flag = postproc.mad_despike_papale_2006(cpk, list(cpk), lat, lon, elev, z=4)
    cpk = cpk.where(spike_flag)
    cpk.loc[start:end, "FC"].plot(label="Despiked FC")

    # ustar filter
    ustar_flag, ustar_thresh = postproc.ustar_filter_papale_2006(cpk, "TA", "USTAR", "FC", lat, lon, elev)
    cpk.loc[~ustar_flag, ["H", "LE", "FC"]] = np.nan
    cpk.loc[start:end, "FC"].plot(label="Ustar Filtered FC")

    # gapfill
    # gapfill_result = postproc.xgb_gapfill_liu_2025(
    #     cpk, "FC",
    #     ta_col="TA",
    #     ppfd_col="PPFD_IN",
    #     vpd_col="VPD_PI",
    #     tsoil_col="TS_1_1_1",
    #     swc_col="SWC_1_1_1",
    #     extra_predictor_cols=["WD"],
    #     n_missing_allowed=1,
    #     random_state=8472,
    #     verbose=True,
        
    #     hyper_train_frac=0.3,
    #     hyper_test_frac=0.3/3,
    #     n_bayes_iter=50,
    #     cv_folds=5,
    # )
    gapfill_result = postproc.mds_gapfill_reichstein_2005(
        cpk, "FC",
        ta_col="TA",
        sw_in_col="PPFD_IN",
        vpd_col="VPD_PI",
        tol_sw=50,
        tol_ta=2.5,
        tol_vpd=0.5,
        max_window_days=28,
        min_samples=5,
        verbose=True
    )
    print(gapfill_result)
    cpk.loc[:, "FC"] = gapfill_result.filled
    cpk.loc[start:end, "FC"].plot(label="Gapfilled FC", style='o', markersize=3)
    plt.legend()

    plt.show()

    # import matplotlib.pyplot as plt
    # gapfill_result.filled.loc["2009-12-01":"2009-12-31"].plot()
    # cpk_ustar.loc["2009-12-01":"2009-12-31", "FC"].plot()
    # plt.show()
