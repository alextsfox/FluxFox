"""
Functions for computing fluxes
"""

import numpy as np

from .thermo import (
    EPS, 
    compute_R_m, 
    compute_ta_from_tson, 
    compute_Cp_m, compute_Cv_m
)

def compute_H_tsonic_crosswind_correction(
    cov_w_tson: np.ndarray, cov_w_rhov: np.ndarray, cov_w_u: np.ndarray, 
    rho_v: np.ndarray, rho_d: np.ndarray, 
    tson_c: np.ndarray, 
    u: np.ndarray, 
    cov_w_q: np.ndarray=None
) -> np.ndarray:
    r"""
    Compute the sonic temperature and cross-wind correction or sensible heat flux according to

    P. Schotanus, F.T.M. Nieuwstadt & H.A.R. De Bruin. "Temperature measurement with a sonic anemometer and its application to heat and moisture fluxes." Boundary-Layer Meteorology 26, pp81-93 (1983).

    .. math::
        \begin{align}
            H &= \rho_m c_{p,m}\text{Cov}(w,T_{\text{son}}) + H_{T_{\text{son}}} + H_{u\star} \\
            H_{T_{\text{son}}} &= - 0.51 \rho_m c_{p, m} \text{Cov}(w, q) \approx - 0.51 c_{p, m} \text{Cov}(w, \rho_v) \\
            H_{u\star} &= \rho_m c_{p, m} \frac{2 u}{c^2}  \text{Cov}(w,u)
        \end{align}


    where ::math::`c^2 \approx 1.4(1 + 0.51q)R_m` is the (squared) speed of sound in moist air and ::math::`q=\rho_v/rho_m` is specific humidity.
    Parameters
    ----------
    cov_w_tson : float
        Covariance of vertical wind speed and sonic temperature.
    cov_w_rhov : float
        Covariance of vertical wind speed and water vapor density.
    cov_w_u : float
        Covariance of vertical wind speed and horizontal wind speed.
    rho_m : float
        Mean air density (kg/m^3).
    cp_m : float
        Mean specific heat capacity of air at constant pressure (J/(kg*K)).
    tson_c : float
        Mean sonic temperature (C).
    u : float
        Mean horizontal wind speed (m/s).
    cov_w_q : float | None
        Covariance of vertical wind speed and specific humidity, if available. Will be used instead of cov(w, rho_v)/(rho_v + rho_d) if provided.
    
    Returns
    -------
    float
        Sensible heat flux (H) in W/m^2.

    # TODO: this whole expression seems like black magic, specifically the conversions between cov(w, q) and cov(w, rho_v)/rho_m and between Ta and Tson. Do more research.
    """

    rho_m = rho_v + rho_d
    w = rho_v / rho_d
    temp_c = compute_ta_from_tson(tson_c, w)
    t_k = temp_c + 273.15

    # adiabatic index gamma
    c_p = compute_Cp_m(w=w, temp_c=temp_c)
    c_v = compute_Cv_m(w=w, temp_c=temp_c)
    gamma = c_p/c_v
    R_m = compute_R_m(w)  # specific heat capacity of moist air
    # TODO: is this different for a closed path system?
    c_sq = gamma*R_m*t_k

    # 0.51 often appears instead of 0.61...why?
    cov_w_q = cov_w_q if cov_w_q is not None else cov_w_rhov/rho_m

    return rho_m*c_p * (
        - 0.51*t_k*cov_w_rhov/rho_m
        + 2*t_k*u/c_sq * cov_w_u
    )

def compute_wpl_co2_term(cov_w_rhov, cov_w_t, rho_c, rho_v, rho_d, temp_c, cov_w_p=0, p=101300):
    r"""
    Compute the WPL term to be applied to carbon dioxide (or other trace gas) flux.

    .. math::
        \begin{align}
        F_c &= \text{Cov}(w, \rho_c) + \text{WPL}_c \\
        \text{WPL_c} &= + \rho_c\left[1.61r\frac{\text{Cov}(w, \rho_v)}{\rho_v} + (1 + 1.61r)\frac{\text{Cov}(w, T)}{T} + (1 + 1.61r)\frac{\text{Cov}(w, p)}{p}\right]
        \end{align}

    where ::math::`r` is the mixing ratio (::math::`\rho_v/\rho_d`), ::math::`T` is temperature in K, ::math::`p` is ambient pressure. Note that the last term (pressure term) is often ignored.
    Parameters
    ----------
    cov_w_rhov : float
        Covariance of vertical wind speed and water vapor density.
    cov_w_t : float
        Covariance of vertical wind speed and actual (not sonic) temperature.
    rho_c : float
        Mean CO2 density (kg/m^3) (or any other trace gas).
    rho_v : float
        Mean water vapor density (kg/m^3).
    rho_d : float
        Mean dry air density (kg/m^3).
    temp_c : float
        Mean temperature (C).
    cov_w_p : float, optional
        Covariance of vertical wind speed and pressure. Default is 0, as this term is often negligible and unavailable in many datasets.
    p : float, optional
        Mean pressure (Pa). Default is 101300. Required only if cov_w_p is provided.

    Returns
    -------
    float
        WPL-corrected CO2 flux in kg/(m^2*s).
    """
    t_k = temp_c + 273.15
    musig_term =  1 + rho_v/rho_d * 1/EPS

    return 1/EPS*rho_c/rho_d*cov_w_rhov + musig_term*rho_c*cov_w_t/t_k + rho_c*musig_term*cov_w_p/p

def compute_wpl_h2o_term(cov_w_rhov, cov_w_t, rho_v, rho_d, temp_c):
    r"""
    Compute the WPL term to be applied to water vapor flux

    .. math::
        \begin{align}
            F_v &= \text{Cov}(w, \rho_v) + \text{WPL}_v \\ 
            \text{WPL}_v &= 1.61r\  \text{Cov}(w, \rho_v) + \rho_v(1 + 1.61r)\frac{\text{Cov}(w, T)}{T}
        \end{align}

    where ::math::`r` is the mixing ratio (::math::`\rho_v/\rho_d`) and ::math::`T` is temperature in K.

    Parameters
    ----------
    cov_w_rhov : float
        Covariance of vertical wind speed and water vapor density.
    cov_w_t : float
        Covariance of vertical wind speed and actual (not sonic) temperature.
    rho_v : float
        Mean water vapor density (kg/m^3).
    rho_d : float
        Mean dry air density (kg/m^3).
    temp_c : float
        Mean temperature (C).

    Returns
    -------
    float
        WPL-corrected H2O flux in kg/(m^2*s).
    """
    musig_term =  1 + rho_v/rho_d * 1/EPS
    t_k = temp_c + 273.15
    return rho_v/rho_d*1/EPS*cov_w_rhov + musig_term*rho_v/t_k*cov_w_t