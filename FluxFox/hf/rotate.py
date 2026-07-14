"""
Functions for applying tilt corrections
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

@dataclass
class PlanarFitResult:
    r"""
    Result of the planar fit rotation computation, containing the regression coefficients and the rotation matrix.
    Apply M_pf @ U to get the rotated velocity vector. U should be a 3 x N array with U[0, :] being the x-component, U[1, :] being the y-component, and U[2, :] being the z-component.
    """
    b0: float
    b1: float
    b2: float
    M_pf: np.ndarray

class PlanarFitRotation:
    def __init__(self):
        r"""
        Class for performing planar fit rotation on 3D velocity data.
        
        1. First, a multilinear regression is performed between the mean components of the measured wind velocity to obtain :math:`b_0`, :math:`b_1`, and :math:`b_2`, such that
        
        .. math::
            |\vec{U}_z| \sim b_0 + b_1*|\vec{U}_x| + b_2*|\vec{U}_y|
        
        2. The coefficients :math:`b_1` and :math:`b_2` are then used to compute the rotation angles :math:`\alpha` and :math:`\beta` for the planar fit rotation.
        .. math::
            \sin\alpha = -\frac{b_1}{\sqrt{b_1^2 + b_2^2 + 1}}
        
        .. math::
            \cos\alpha = \frac{\sqrt{b_2^2 + 1}}{\sqrt{b_1^2 + b_2^2 + 1}}
        
        .. math::
            \sin\beta = \frac{b_2}{\sqrt{b_2^2 + 1}}
        
        .. math::
            \cos\beta = \frac{1}{\sqrt{b_2^2 + 1}}

        .. math::
            M_pf = 
                \begin{pmatrix}
                    \cos\alpha & 0 & -\sin\alpha\\
                    0&1&0\\
                    \sin\alpha&0&\cos\alpha
                \end{pmatrix}
                \begin{pmatrix}
                    1&0&0\\
                    0&\cos\beta&\sin\beta\\
                    0&\sin\beta&\cos\beta
                \end{pmatrix}

        3. The resulting rotation matrix :math:`M_{pf}` is then applied to the high-frequency velocity data to align the mean wind vector with the horizontal plane.
        
        .. math::
            \vec{U}^{\prime} = M_{pf} \vec{U}

        4. A final rotation aligns the mean wind vector with the x-axis, ensuring that the mean wind direction is along the streamwise direction.
        
        .. math::
            \vec{U}^{\prime\prime} = 
                \begin{pmatrix}
                    \cos\theta & \sin\theta & 0\\
                    -\sin\theta & \cos\theta & 0\\
                    0 & 0 & 1
                \end{pmatrix}
                \vec{U}^{\prime}
        
        where :math:`\theta=\arctan2(|\vec{U}^{\prime}_y|, |\vec{U}^{\prime}_x|)` is the mean wind direction.

        See Wilczak et al. (2001) for details.

        Steps 1 and 2 are performed by the `compute` method of this class, which calculates the rotation matrix based on the mean velocities.

        Steps 3 and 4 are performed by the `rotate` method of this class, which applies the previously computed rotation matrix to the high-frequency velocity data, and can only be called once the `compute` method has been called to calculate the rotation matrix.

        Usage
        -----
        First, create an instance of the `PlanarFitRotation` class and call the `compute` method with the mean velocity data over the desired time period (at least 2 weeks, ideally more) to calculate the rotation matrix. Then, use the `rotate` method to apply the rotation to the high-frequency velocity data.

        Examples
        --------
        .. code-block:: python
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd
            from FluxFox.hf.rotate import PlanarFitRotation

            
            rng = np.random.default_rng(8472)
            # simulating a planar fit on 2 weeks of 0.1Hz data with a 30 minute averaging period
            raw_dfs = []
            for _ in range(48*14):
                u = rng.normal(1, 0.5, 180)
                v = rng.normal(0.5, 0.2, 180)
                w = rng.normal(u*0.03 + v*0.06, 0.05, 180)
                raw_dfs.append(pd.DataFrame({
                    "u": u,
                    "v": v,
                    "w": w
                }))

            # compute the mean wind components
            mean_df = pd.DataFrame({
                "u": [df["u"].mean() for df in raw_dfs],
                "v": [df["v"].mean() for df in raw_dfs],
                "w": [df["w"].mean() for df in raw_dfs]
            })

            # compute planar fit matrix
            pf = PlanarFitRotation()
            pf_res = pf.compute(
                mean_df=mean_df,
                u_bar_col="u",
                v_bar_col="v",
                w_bar_col="w",
                assume_perfect_w=True
            )
            print(pf_res)

            # apply the rotation to the high-frequency data
            rot_dfs = []
            for df in raw_dfs:
                rot_dfs.append(pf.rotate(df, "u", "v", "w"))

            # proper way:
            rot_mean_df = pd.DataFrame({
                "u": [df["u"].mean() for df in rot_dfs],
                "v": [df["v"].mean() for df in rot_dfs],
                "w": [df["w"].mean() for df in rot_dfs]
            })
            
            # visualize the rotated data
            fig = plt.figure(figsize=(14, 7))

            fig.add_subplot(231)
            plt.scatter(mean_df["u"], mean_df["w"], s=4, alpha=0.5, label="Unrotated")
            plt.scatter(rot_mean_df["u"], rot_mean_df["w"], s=4, alpha=0.5, label="Rotated")
            plt.axline([mean_df["u"].mean(), mean_df["w"].mean()], slope=pf_res.b1)
            plt.title("Mean Wind Component\nScatterplots")
            plt.xlabel("Mean Ux")
            plt.ylabel("Mean Uz")
            plt.grid(True)

            fig.add_subplot(234)
            plt.scatter(mean_df["v"], mean_df["w"], s=4, alpha=0.5)
            plt.scatter(rot_mean_df["v"], rot_mean_df["w"], s=4, alpha=0.5)
            plt.axline([mean_df["v"].mean(), mean_df["w"].mean()], slope=pf_res.b2)
            plt.xlabel("Mean Uv")
            plt.ylabel("Mean Uz")
            plt.grid(True)

            fig.add_subplot(332)
            plt.plot(raw_dfs[5]["u"], 'o', markersize=1, label="Unrotated")
            plt.plot(rot_dfs[5]["u"], 'o', markersize=1, label="Rotated")
            plt.axhline(raw_dfs[5]["u"].mean(), color="C0", ls="--")
            plt.axhline(rot_dfs[5]["u"].mean(), color="C1", ls="--")
            plt.ylabel("Ux")
            plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
            plt.title("Example High Frequency\nTimeseries for\naveraging window #5")
            plt.legend()
            
            fig.add_subplot(335)
            plt.plot(raw_dfs[5]["v"], 'o', markersize=1, )
            plt.plot(rot_dfs[5]["v"], 'o', markersize=1, )
            plt.axhline(raw_dfs[5]["v"].mean(), color="C0", ls="--")
            plt.axhline(rot_dfs[5]["v"].mean(), color="C1", ls="--")
            plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
            plt.ylabel("Uy")
            
            fig.add_subplot(338)
            plt.plot(raw_dfs[5]["w"], 'o', markersize=1, )
            plt.plot(rot_dfs[5]["w"], 'o', markersize=1, )
            plt.axhline(raw_dfs[5]["w"].mean(), color="C0", ls="--")
            plt.axhline(rot_dfs[5]["w"].mean(), color="C1", ls="--")
            plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
            plt.xlabel("Seconds")
            plt.ylabel("Uz")
            
            fig.add_subplot(333)
            plt.plot(mean_df["u"], 'o', markersize=1, label="Unrotated")
            plt.plot(rot_mean_df["u"], 'o', markersize=1, label="Rotated")
            plt.axhline(mean_df["u"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
            plt.axhline(rot_mean_df["u"].mean(), color="C1", ls="--", label="Rotated Global Mean")
            plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
            plt.ylabel("Mean Ux")
            plt.title("Mean Wind Velocities")
            
            fig.add_subplot(336)
            plt.plot(mean_df["v"], 'o', markersize=1, label="Unrotated")
            plt.plot(rot_mean_df["v"], 'o', markersize=1, label="Rotated")
            plt.axhline(mean_df["v"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
            plt.axhline(rot_mean_df["v"].mean(), color="C1", ls="--", label="Rotated Global Mean")
            plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
            plt.ylabel("Mean Uy")
            
            fig.add_subplot(339)
            plt.plot(mean_df["w"], 'o', markersize=1, label="Unrotated")
            plt.plot(rot_mean_df["w"], 'o', markersize=1, label="Rotated")
            plt.axhline(mean_df["w"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
            plt.axhline(rot_mean_df["w"].mean(), color="C1", ls="--", label="Rotated Global Mean")
            plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
            plt.xlabel("Hours")
            plt.ylabel("Mean Uz")

            fig.tight_layout(h_pad=1.5, w_pad=50)
            plt.show()
        """
        self._rotation: Callable | None = None
        pd.DataFrame.resample
    
    def compute(
        self, 
        mean_df:pd.DataFrame,
        u_bar_col: str,
        v_bar_col: str,
        w_bar_col: str,
        max_w: float = 10,
        min_speed: float = 0,
        assume_unbiased_w: bool = False
        ) -> None:
        r"""
        Compute the rotation matrix from the mean velocities.

        Parameters
        ----------
        mean_df : pd.DataFrame
            DataFrame containing mean velocities with columns specified by `u_bar_col`, `v_bar_col`, and `w_bar_col`.
        u_bar_col : str
            Name of the column in `mean_df` containing the mean velocity in the x-direction.
        v_bar_col : str
            Name of the column in `mean_df` containing the mean velocity in the y-direction.
        w_bar_col : str
            Name of the column in `mean_df` containing the mean velocity in the z-direction.
        max_w : float
            Maximum allowable mean vertical velocity magnitude (measured in the un-rotated frame). Default is 10 m/s, which only filters out non-physical vertical velocities.
        min_speed : float
            Minimum allowable mean wind velocity magnitude. Default is 0 m/s.
        assume_unbiased_w: bool = False
            Whether to assume there is no instrument-induced bias in the vertical velocity measurement (w). If True, the regression :math:`|\vec{U}_z| \sim 0 + b_1*|\vec{U}_x| + b_2*|\vec{U}_y|` is performed with b0 set to 0. If False, the planar fit computation estimates b0 from the data as :math:`|\vec{U}_z| \sim b_0 + b_1*|\vec{U}_x| + b_2*|\vec{U}_y|`. Default False.
        Returns
        -------
        PlanarFitResult
            Object containing the computed rotation matrix and any other relevant information from the planar fit computation.
            The PlanarFitRotation object stores the computed rotation function internally -- this returned value is only for the user's records.
        """
        
        # compute the plane of the local streamlines to yield a rotation matrix to apply to all the data
        ubar = mean_df[u_bar_col].to_numpy()
        vbar = mean_df[v_bar_col].to_numpy()
        wbar = mean_df[w_bar_col].to_numpy()

        # mask out bad values
        mask = (
            (np.abs(wbar) <= max_w) 
            & (np.sqrt(ubar**2 + vbar**2 + wbar**2) >= min_speed)
            & np.isfinite(ubar)
            & np.isfinite(vbar)
            & np.isfinite(wbar)
        )
        if assume_unbiased_w:
            b0 = 0.0
            b1, b2 = sm.OLS(wbar[mask], np.stack((ubar[mask], vbar[mask]), axis=1)).fit().params
        else:
            b0, b1, b2 = sm.OLS(wbar[mask], sm.add_constant(np.stack((ubar[mask], vbar[mask]), axis=1))).fit().params
        
        sinbeta = b2 / np.sqrt(b2**2 + 1)
        cosbeta = 1 / np.sqrt(b2**2 + 1)
        sinalpha = -b1 / np.sqrt(b1**2 + b2**2 + 1)
        cosalpha = np.sqrt(b2**2 + 1) / np.sqrt(b1**2 + b2**2 + 1)

        M1 = np.array([
            [cosalpha, 0.0, -sinalpha],
            [0.0, 1.0, 0.0],
            [sinalpha, 0.0, cosalpha]
        ])
        M2 = np.array([
            [1.0, 0.0, 0.0],
            [0, cosbeta, sinbeta],
            [0, -sinbeta, cosbeta]
        ]) 
        M_pf = M1 @ M2

        def _rotation(U:np.ndarray) -> np.ndarray:
            r"""
            Function to apply the planar fit rotation to high-frequency data.

            Parameters
            ----------
            U : np.ndarray
                High-frequency velocity data array of shape (3, N), where N is the number of samples. The rows correspond to the u, v, and w velocity components, respectively.

            Returns
            -------
            np.ndarray
                Rotated high-frequency velocity data array of shape (3, N), where N is the number of samples. The rows correspond to the u, v, and w velocity components, respectively.
            """
            # Apply the planar fit matrix
            Uprime = M_pf @ U
            
            # align wind velocity with the x-axis with a rotation about z-axis
            Uprimebar = np.nanmean(Uprime, axis=1)
            theta = np.arctan2(Uprimebar[1], Uprimebar[0])
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            Rtheta = np.array([
                [costheta, sintheta, 0.0],
                [-sintheta, costheta, 0.0],
                [0.0, 0.0, 1.0]
            ])
            return Rtheta @ Uprime
        self._rotation = _rotation

        return PlanarFitResult(b0, b1, b2, M_pf)

    def rotate(
            self, 
            hf_df:pd.DataFrame,
            u_col: str,
            v_col: str,
            w_col: str,
        ) -> pd.DataFrame:
        """
        Apply the planar fit rotation to high-frequency data.

        Parameters
        ----------
        hf_df : pd.DataFrame
            High-frequency data containing the velocity components.
        u_col : str
            Column name for the u-component of velocity.
        v_col : str
            Column name for the v-component of velocity.
        w_col : str
            Column name for the w-component of velocity.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the rotated velocity components with the same column names and index as the input.
        """
        if self._rotation is None:
            raise ValueError("Rotation not computed yet. Call compute() first.")

        # 3 x n
        U_unrot = hf_df[[u_col, v_col, w_col]].to_numpy().T
        U_rot = self._rotation(U_unrot)

        out_df = pd.DataFrame(data=U_rot.T, columns=[u_col, v_col, w_col], index=hf_df.index)

        return out_df

class DoubleRotation:
    def __init__(self):
        r"""
        Initialize the DoubleRotation object.
        This class provides an interface for performing a double rotation on high-frequency wind data.
        The first rotation aligns the mean horizontal wind direction with the x-axis, and the second rotation aligns the mean vertical wind component with the z-axis.

        Steps:
        1. Compute the mean horizontal wind direction and rotate the data so that this direction aligns with the x-axis:
        
        .. math::
            \theta = \arctan2(|\vec{U}|_y, |\vec{U}|_x)
        
        .. math::
            R_\theta = \begin{pmatrix}
                \cos\theta & \sin\theta & 0 \\
                -\sin\theta & \cos\theta & 0 \\
                0 & 0 & 1
            \end{pmatrix}
        
        .. math::
            \vec{U}' = R_\theta \vec{U}
        
        2. Compute the mean vertical wind component and rotate the data so that this component aligns with the z-axis:
        
        .. math::
            \phi = \arctan2(|\vec{U}'|_z, |\vec{U}'|_x)
        
        .. math::
            R_\phi = \begin{pmatrix}
                \cos\phi & 0 & \sin\phi \\
                0 & 1 & 0 \\
                -\sin\phi & 0 & \cos\phi
            \end{pmatrix}
        
        .. math::
            \vec{U}'' = R_\phi \vec{U}'

        Usage
        -----
        To use this class, call the `rotate` method with high-frequency data to apply the double rotation.
        The `compute` dummy method is included to maintain a consistent interface with other rotation classes, but it does not perform any computation.

        Example
        -------
        .. code-block:: python
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from FluxFox.hf import rotate

            rng = np.random.default_rng(8472)
            u = rng.normal(1, 0.5, 100)
            v = rng.normal(0.5, 0.2, 100)
            w = rng.normal(u*0.05 + v*0.06, 0.1, 100)
            raw = pd.DataFrame({
                'u': u,
                'v': v,
                'w': w
            })
            dr = rotate.DoubleRotation()
            rotated = dr.rotate(raw, u_col='u', v_col='v', w_col='w')

            fig, axs = plt.subplots(3, 1, figsize=(8, 8))
            for ax, col in zip(axs, ['u', 'v', 'w']):
                ax.plot(raw[col], label='Unrotated')
                ax.plot(rotated[col], label='Rotated')
                ax.axhline(raw[col].mean(), color='C0', linestyle='--', label='Unrotated Mean')
                ax.axhline(rotated[col].mean(), color='C1', linestyle='--', label='Rotated Mean')
                ax.set_title(col)
            axs[0].legend()
            plt.tight_layout()
            plt.show()
        """
        pass
    def compute(
        self, 
        mean_df:pd.DataFrame,
        u_bar_col: str,
        v_bar_col: str,
        w_bar_col: str
    )->None:
        r"""Dummy function to maintain the interface. Does nothing."""
        return
    
    def rotate(
        self, 
        hf_df:pd.DataFrame, 
        u_col: str, 
        v_col: str, 
        w_col: str
    ) -> pd.DataFrame:
        r"""
        Apply a double rotation to high-frequency data, first aligning the mean horizontal wind direction with the x-axis, and then aligning the mean vertical wind component with the z-axis.

        Parameters
        ----------
        hf_df : pd.DataFrame
            High-frequency data containing the velocity components.
        u_col : str
            Column name for the u-component of velocity.
        v_col : str
            Column name for the v-component of velocity.
        w_col : str
            Column name for the w-component of velocity.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the rotated velocity components with the same column names and index as the input.
        """

        Ubar = hf_df[[u_col, v_col, w_col]].mean().to_numpy()
        theta = np.arctan2(Ubar[1], Ubar[0])
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        Rtheta = np.array([
            [costheta, sintheta, 0.0],
            [-sintheta, costheta, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        Uprime = Rtheta @ hf_df[[u_col, v_col, w_col]].to_numpy().T

        Uprimebar = np.nanmean(Uprime, axis=1)
        phi = np.arctan2(Uprimebar[2], Uprimebar[0])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        Rphi = np.array([
            [cosphi, 0.0, sinphi],
            [0.0, 1.0, 0.0],
            [-sinphi, 0.0, cosphi]
        ])
        U_rot = Rphi @ Uprime

        df_out = pd.DataFrame(
            data=U_rot.T,
            columns=[u_col, v_col, w_col],
            index=hf_df.index
        )

        return df_out

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # rng = np.random.default_rng(8472)
    # u = rng.normal(1, 0.5, 100)
    # v = rng.normal(0.5, 0.2, 100)
    # w = rng.normal(u*0.05 + v*0.06, 0.1, 100)
    # raw = pd.DataFrame({
    #     'u': u,
    #     'v': v,
    #     'w': w
    # })
    # dr = DoubleRotation()
    # rotated = dr.rotate(raw, u_col='u', v_col='v', w_col='w')

    # fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    # for ax, col in zip(axs, ['u', 'v', 'w']):
    #     ax.plot(raw[col], 'o', label='Unrotated')
    #     ax.plot(rotated[col], 'o', label='Rotated')
    #     ax.axhline(raw[col].mean(), color='C0', linestyle='--', label='Unrotated Mean')
    #     ax.axhline(rotated[col].mean(), color='C1', linestyle='--', label='Rotated Mean')
    #     ax.set_title(col)
    # axs[0].legend()
    # plt.tight_layout()
    # plt.show()
    
    rng = np.random.default_rng(8472)
    # simulating a planar fit on 2 weeks of 0.1Hz data with a 30 minute averaging period
    raw_dfs = []
    for _ in range(48*14):
        u = rng.normal(1, 0.5, 180)
        v = rng.normal(0.5, 0.2, 180)
        w = rng.normal(u*0.03 + v*0.06, 0.05, 180)
        raw_dfs.append(pd.DataFrame({
            "u": u,
            "v": v,
            "w": w
        }))

    # compute the mean wind components
    mean_df = pd.DataFrame({
        "u": [df["u"].mean() for df in raw_dfs],
        "v": [df["v"].mean() for df in raw_dfs],
        "w": [df["w"].mean() for df in raw_dfs]
    })

    # compute planar fit matrix
    pf = PlanarFitRotation()
    pf_res = pf.compute(
        mean_df=mean_df,
        u_bar_col="u",
        v_bar_col="v",
        w_bar_col="w",
        assume_unbiased_w=False
    )
    print(pf_res)

    # apply the rotation to the high-frequency data
    rot_dfs = []
    for df in raw_dfs:
        rot_dfs.append(pf.rotate(df, "u", "v", "w"))

    # proper way:
    rot_mean_df = pd.DataFrame({
        "u": [df["u"].mean() for df in rot_dfs],
        "v": [df["v"].mean() for df in rot_dfs],
        "w": [df["w"].mean() for df in rot_dfs]
    })
    
    # visualize the rotated data
    fig = plt.figure(figsize=(14, 7))

    fig.add_subplot(231)
    plt.scatter(mean_df["u"], mean_df["w"], s=4, alpha=0.5, label="Unrotated")
    plt.scatter(rot_mean_df["u"], rot_mean_df["w"], s=4, alpha=0.5, label="Rotated")
    plt.axline([mean_df["u"].mean(), mean_df["w"].mean()], slope=pf_res.b1)
    plt.title("Mean Wind Component\nScatterplots")
    plt.xlabel("Mean Ux")
    plt.ylabel("Mean Uz")
    plt.grid(True)

    fig.add_subplot(234)
    plt.scatter(mean_df["v"], mean_df["w"], s=4, alpha=0.5)
    plt.scatter(rot_mean_df["v"], rot_mean_df["w"], s=4, alpha=0.5)
    plt.axline([mean_df["v"].mean(), mean_df["w"].mean()], slope=pf_res.b2)
    plt.xlabel("Mean Uv")
    plt.ylabel("Mean Uz")
    plt.grid(True)

    fig.add_subplot(332)
    plt.plot(raw_dfs[5]["u"], 'o', markersize=1, label="Unrotated")
    plt.plot(rot_dfs[5]["u"], 'o', markersize=1, label="Rotated")
    plt.axhline(raw_dfs[5]["u"].mean(), color="C0", ls="--")
    plt.axhline(rot_dfs[5]["u"].mean(), color="C1", ls="--")
    plt.ylabel("Ux")
    plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
    plt.title("Example High Frequency\nTimeseries for\naveraging window #5")
    plt.legend()
    
    fig.add_subplot(335)
    plt.plot(raw_dfs[5]["v"], 'o', markersize=1, )
    plt.plot(rot_dfs[5]["v"], 'o', markersize=1, )
    plt.axhline(raw_dfs[5]["v"].mean(), color="C0", ls="--")
    plt.axhline(rot_dfs[5]["v"].mean(), color="C1", ls="--")
    plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
    plt.ylabel("Uy")
    
    fig.add_subplot(338)
    plt.plot(raw_dfs[5]["w"], 'o', markersize=1, )
    plt.plot(rot_dfs[5]["w"], 'o', markersize=1, )
    plt.axhline(raw_dfs[5]["w"].mean(), color="C0", ls="--")
    plt.axhline(rot_dfs[5]["w"].mean(), color="C1", ls="--")
    plt.xticks(np.arange(raw_dfs[5].shape[0], step=30), np.arange(0, 1800, step=300))
    plt.xlabel("Seconds")
    plt.ylabel("Uz")
    
    fig.add_subplot(333)
    plt.plot(mean_df["u"], 'o', markersize=1, label="Unrotated")
    plt.plot(rot_mean_df["u"], 'o', markersize=1, label="Rotated")
    plt.axhline(mean_df["u"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
    plt.axhline(rot_mean_df["u"].mean(), color="C1", ls="--", label="Rotated Global Mean")
    plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
    plt.ylabel("Mean Ux")
    plt.title("Mean Wind Velocities")
    
    fig.add_subplot(336)
    plt.plot(mean_df["v"], 'o', markersize=1, label="Unrotated")
    plt.plot(rot_mean_df["v"], 'o', markersize=1, label="Rotated")
    plt.axhline(mean_df["v"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
    plt.axhline(rot_mean_df["v"].mean(), color="C1", ls="--", label="Rotated Global Mean")
    plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
    plt.ylabel("Mean Uy")
    
    fig.add_subplot(339)
    plt.plot(mean_df["w"], 'o', markersize=1, label="Unrotated")
    plt.plot(rot_mean_df["w"], 'o', markersize=1, label="Rotated")
    plt.axhline(mean_df["w"].mean(), color="C0", ls="--", label="Unrotated Global Mean")
    plt.axhline(rot_mean_df["w"].mean(), color="C1", ls="--", label="Rotated Global Mean")
    plt.xticks(np.arange(mean_df.shape[0], step=96), np.arange(0, 48*14/2, step=48).astype(int))
    plt.xlabel("Hours")
    plt.ylabel("Mean Uz")

    fig.tight_layout(h_pad=1.5, w_pad=50)
    plt.show()