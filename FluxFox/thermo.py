"""
Functions for performing generic thermodynamic calculations
"""
import warnings
from typing import Any

import numpy as np
import pandas as pd

MV = 0.018015  
"""kg/mol, molar mass of water"""

MC = 0.044009
"""kg/mol, molar mass of CO2"""

MD = 0.0289647
"""kg/mol, molar mass of dry air"""

EPS = MV / MD
"""ratio of molar masses of water and dry air"""

R = 8.314463
"""J/mol/K, universal gas constant"""
RV = R / MV 
"""J/kg/K, specific gas constant for water vapor"""

RC = R / MC 
"""J/kg/K, specific gas constant for CO2"""

RD = R / MD 
"""J/kg/K, specific gas constant for dry air"""

def _check_temp_c(temp_c: Any) -> None:
    if np.nanmean(temp_c) > 200:
        warnings.warn(f"[thermo] WARNING: average temperature is extremely high ({np.nanmean(temp_c):.1f}°C). Did you input Kelvin by mistake?")

def _check_w_kgkg(w: Any) -> None:
    if np.nanmean(w) > 1:
        warnings.warn(f"[thermo] WARNING: average mixing ratio is extremely high ({np.nanmean(w):.3f} kg/kg). Did you input g/kg by mistake?")

def _check_rh_frac(rh_frac: Any) -> None:
    if np.nanmean(rh_frac) > 2:
        warnings.warn(f"[thermo] WARNING: average relative humidity is extremely high ({np.nanmean(rh_frac):.3f}). Did you input % by mistake?")

def _check_pa_pa(pa: Any) -> None:
    if np.nanmean(pa) < 10_000:
        warnings.warn(f"[thermo] WARNING: average air pressure is extremely low ({np.nanmean(pa):.1f} Pa). Did you input hPa or kPa by mistake?")

def _check_e_pa(e: Any) -> None:
    if np.nanmean(e) < 10:
        warnings.warn(f"[thermo] WARNING: average vapor pressure is suspiciously low ({np.nanmean(e):.1f} Pa). Did you input hPa or kPa by mistake?")

def _check_chi_c(chi_c: Any) -> None:
    if np.nanmean(chi_c) > 0.1:
        warnings.warn(f"[thermo] WARNING: average mole fraction of CO2 is extremely high ({np.nanmean(chi_c):.3f}). Did you input ppm by mistake?")

def _return_any(input, output) -> Any:
    if isinstance(input, float):
        return output
    elif isinstance(input, np.ndarray):
        return np.asarray(output)
    elif isinstance(input, pd.Series):
        return pd.Series(output, index=input.index)
    elif isinstance(input, pd.DataFrame):
        return pd.DataFrame(output, index=input.index, columns=input.columns)
    else:
        return output

def compute_Cp_d(temp_c: Any | None=None, check_inputs:bool=True)->Any:
    """"
    Computed the specific heat capacity of dry air (J/kg/K) as a function of temperature in degrees C.
    If temperature is not provided (or where temperature is NaN), a constant value of 1004.7 J/kg/K is returned (Cp_d at 10°C).

    ::math::
        C_{p,d} = 1003 + 0.05(T_C + 23.15)

    Parameters
    ----------
    temp_c
        Temperature, in degrees C.  Can be a float, np.ndarray, pd.Series, or pd.DataFrame.
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.
    
    Returns
    -------
    Cp_d
        Specific heat capacity of dry air in J/kg/K
    """
    
    cp_d = 1004.7
    if temp_c is not None:
        if check_inputs:
            _check_temp_c(temp_c)
        cp_d = np.where(
            np.isfinite(temp_c), 
            1003 + 0.05*(temp_c + 23.15),
            cp_d
        )
    return _return_any(temp_c, cp_d)

def compute_Cp_v(temp_c: Any | None=None, check_inputs: bool=True)->Any:
    """
    Compute the specific heat capacity of water vapor (J/kg/K) as a function of temperature in degrees C.

    ::math::
        C_{p,v} = 1855 + 0.2133(T_C + 23.15)

    If temperature is not provided (or where temperature is NaN), a constant value of 1862 J/kg/K (Cp_v at 10°C) is returned.
    
    Parameters
    ----------
    temp_c
        Temperature, in degrees C. Can be a float, np.ndarray, pd.Series, or pd.DataFrame.
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.
    
    Returns
    -------
    Cp_v
        Specific heat capacity of water vapor in J/kg/K
    """
    cp_v = 1862
    if temp_c is not None:
        if check_inputs:
            _check_temp_c(temp_c)
        cp_v = np.where(
            np.isfinite(temp_c), 
            1855 + 0.2133*(temp_c + 23.15),
            cp_v
        )
    return _return_any(temp_c, cp_v)

def compute_Cp_m(
    rh_frac: Any | None=None,
    w: Any | None=None,
    pa: Any | None=None,
    e: Any | None=None,
    temp_c: Any | None=None,
    check_inputs: bool=True
)->Any:
    """
    Compute the specific heat capacity of moist air (J/kg/K) as a function of temperature in degrees C and water vapor content.

    ::math::
        C_{p,m} = C_{p,d} + w C_{p,v}

    This function works with many combinations of input parameters. The following combinations are allowed, and will be used in the order of preference listed below:

    * w, temp_c (best)
    * e, pa, temp_c (best)
    * rh_frac, temp_c, pa (less accurate, requires computing e_sat from temp_c)
    * w (less accurate, requires assuming a constant value of Cp_d and Cp_v at 10°C)
    * e, pa (less accurate, requires assuming a constant value of Cp_d and Cp_v at 10°C)

    Additionally, all of the above can be provided, in which case the first valid combination will be used for each time point. (e.g. if w and temp_c are not nan for index 0, then method 1 will be used for index 0. If w is nan for index 1 but e, pa, and temp_c are not nan, then method 2 will be used for index 1, etc.)

    Parameters
    ----------
    rh_frac : array-like, optional
        Relative humidity (0-1)
    w : array-like, optional
        Mixing ratio of water vapor (kg/kg)
    pa : array-like, optional
        Air pressure (Pa)
    e : array-like, optional
        Vapor pressure (Pa)
    temp_c : array-like, optional
        Temperature (°C)
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.
    
    Returns
    -------
    Cp_m
        Specific heat capacity of moist air in J/kg/K.
    """

    if w is None:
        w = np.nan
    else:
        if check_inputs:
            _check_w_kgkg(w)
    if e is None:
        e = np.nan
    else:
        if check_inputs:
            _check_e_pa(e)
    if pa is None:
        pa = np.nan
    else:
        if check_inputs:
            _check_pa_pa(pa)
    if temp_c is None:
        temp_c = np.nan
    else:
        if check_inputs:
            _check_temp_c(temp_c)
    if rh_frac is None:
        rh_frac = np.nan
    else:
        if check_inputs:
            _check_rh_frac(rh_frac)

    cp_m = np.select(
        [
            np.isfinite(w) & np.isfinite(temp_c),
            np.isfinite(e) & np.isfinite(pa) & np.isfinite(temp_c),
            np.isfinite(rh_frac) & np.isfinite(temp_c) & np.isfinite(pa),
            np.isfinite(w),
        ],
        [
            compute_Cp_d(temp_c) + w * compute_Cp_v(temp_c),
            compute_Cp_d(temp_c) + (EPS * e / (pa - e)) * compute_Cp_v(temp_c),
            compute_Cp_d(temp_c) + (EPS * rh_frac * compute_e_sat_water(temp_c) / (pa - (rh_frac * compute_e_sat_water(temp_c)))) * compute_Cp_v(temp_c),
            compute_Cp_d(None) + w * compute_Cp_v(None),
            compute_Cp_d(None) + (EPS * e / (pa - e)) * compute_Cp_v(None),
        ],
        default=np.nan
    )
    return _return_any(w, cp_m)

def compute_Lv(temp_c: Any | None=None, check_inputs: bool=True)->Any:
    r"""
    Henderson-Sellers (1984), QJRMS 110:1186-1190
    Matches Smithsonian Meteorological Tables to within ~0.1%
    Valid 0-100°C

    ::math::
        L_v = 1.91846 \times 10^6 \left(\frac{T_K}{T_K - 33.91}\right)^2

    If temperature is not provided (or where temperature is NaN), a constant value of 2475998 J/kg (L_v at 10°C)is returned.

    Parameters
    ----------
    temp_c
        Temperature, in degrees C. Can be a float, np.ndarray, pd.Series, or pd.DataFrame.
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    Lv
        Latent heat of vaporization in J/kg.
    """
    lv = 2475998
    if temp_c is not None:
        if check_inputs:
            _check_temp_c(temp_c)
        temp_k = temp_c + 273.15
        lv = np.where(
            np.isfinite(temp_c), 
            1.91846e6 * (temp_k / (temp_k - 33.91))**2, 
            lv
        )
    return _return_any(temp_c, lv)

def compute_Lf()->Any:
    r"""
    Returns a constant value of 334000 J/kg
    """
    return 334000

def compute_Ls(temp_c: Any, check_inputs: bool=True)->Any:
    r"""
    Latent heat of sublimation of ice (J/kg)
    Approximately constant-ish but has slight T dependence
    Valid roughly -40 to 0°C

    Murphy-Koop

    ::math::
        L_s = \frac{1}{0.018015}(45782.5 + 32.192*T_K - 0.0706*T_K^2)

    If temperature is not provided (or where temperature is NaN), a constant value of 2835472 J/kg (L_s at -5°C) is returned.

    Parameters
    ----------
    temp_c
        Temperature, in degrees C
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    Ls
        Latent heat of sublimation in J/kg

    """
    ls =  2835472
    if temp_c is not None:
        if check_inputs:
            _check_temp_c(temp_c)
        temp_k = temp_c + 273.15
        ls = np.where(
            np.isfinite(temp_c),
            (46782.5 + 35.8925*temp_k - 0.07414*temp_k**2 + 541.5 * np.exp(-(temp_k /123.75)**2)) / 0.018015,
            ls
        )
    return _return_any(temp_c, ls)

def compute_e_sat_water(temp_c: Any, check_inputs: bool=True)->Any:
    r"""
    Compute the saturation vapor pressure of water vapor over liquid water (Pa) as a function of temperature in degrees C.

    Uses the Buck (1996) formulation.

    ::math::
        e_{sat} = 611.21 \exp\left(\left(18.678 - \frac{T_C}{234.5}\right) \frac{T_C}{257.14 + T_C}\right)

    Parameters
    ----------
    temp_c
        Temperature, in degrees C
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    e_sat
        Saturation vapor pressure in Pa

    """
    if check_inputs:
        _check_temp_c(temp_c)
    e_sat = 611.21 * np.exp((18.678 - temp_c/234.5) * (temp_c / (257.14 + temp_c)))
    return _return_any(temp_c, e_sat)

def compute_e_sat_ice(temp_c: Any, check_inputs: bool=True)->Any:
    r"""
    Compute the saturation vapor pressure of water vapor over ice (Pa) as a function of temperature in degrees C.

    Uses the Buck (1996) formulation.

    ::math::
        e_{sat} = 611.15 \exp\left(\left(23.036 - \frac{T_C}{333.7}\right) \frac{T_C}{279.82 + T_C}\right)

    Parameters
    ----------
    temp_c
        Temperature, in degrees C
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    e_sat
        Saturation vapor pressure in Pa

    """
    if check_inputs:
        _check_temp_c(temp_c)
    e_sat = 611.15 * np.exp((23.036 - temp_c/333.7) * (temp_c / (279.82 + temp_c)))
    return _return_any(temp_c, e_sat)

def compute_rho_d(temp_c: Any, pa: Any, check_inputs: bool=True)->Any:
    """
    Compute the density of dry air (kg/m^3) as a function of temperature in degrees C and air pressure in Pa using the ideal gas law.
    
    Parameters
    ----------
    temp_c : array-like
        Temperature, in degrees C
    pa : array-like
        Air pressure (Pa)
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.
    
    Returns
    -------
    rho_d : array-like
        Density of dry air (kg/m^3)
    """
    if check_inputs:
        _check_temp_c(temp_c)
        _check_pa_pa(pa)
    temp_k = temp_c + 273.15
    rho_d = pa / (RD * temp_k)
    return _return_any(temp_c, rho_d)

def compute_rho_c(temp_c: Any, chi_c: Any, pa: Any, check_inputs: bool=True)->Any:
    """
    Compute the density of CO2 (kg/m^3) as a function of temperature in degrees C, mole fraction of CO2 (mol/mol), and air pressure in Pa using the ideal gas law.

    Parameters
    ----------
    temp_c : array-like
        Temperature, in degrees C
    chi_c : array-like
        Mole fraction of CO2 (mol/mol)
    pa : array-like
        Air pressure (Pa)
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    rho_c : array-like
        Density of CO2 (kg/m^3)
    """
    if check_inputs:
        _check_temp_c(temp_c)
        _check_chi_c(chi_c)
        _check_pa_pa(pa)
    temp_k = temp_c + 273.15
    rho_c = (chi_c * pa) / (RC * temp_k)
    return _return_any(temp_c, rho_c)

def compute_rho_v(temp_c: Any, e: Any | None=None, w: Any | None=None, pa: Any | None=None, rh_frac: Any | None=None, chi_v: Any | None=None, check_inputs: bool=True)->Any:
    """
    Compute the density of water vapor (kg m-3).

    Ambient temperature is required as input, but many combinations of the other parameters can be used to compute partial of water vapor if it is not directly provided.
    

    (w/eps * pa)/(1 + w/eps) = e

    Allowed combinations:
    * temp_c, e (best)
    * temp_c, w, pa (best)
    * temp_c, chi_v, pa (best)
    * temp_c, rh_frac (less accurate, requires computing e_sat from temp_c)
    
    Additionally, all of the above can be provided, in which case the first valid combination will be used for each time point. (e.g. if w is not nan for index 0, then method 1 will be used for index 0. If w is nan for index 1 but vp and pa are not nan not nan, then method 2 will be used for index 1, etc.)
    
    # TODO: add this caveat to all "multi-input" functions
    Note
    ----
    Note that the combination priority reflects how accurate the algorithm is expected to be, not how accurate the final result will be.
    This matters if, for example, your measurement of chi_v comes from a less accurate instrument (such as a fast response infrared gas analyzer), but your rh measurement comes from a more accurate instrument (such as an aspirated, well-shielded, high-quality hygrometer).
    In this case, the rh_frac measurement may be more accurate than the chi_v measurement, and you may want to use the rh_frac method instead of the chi_v method:
    ```python
        # in this specific case, method A is more accurate than method B

        # Method A: compute rho_v first from rh then from chi_v
        rho_v_1 = compute_rho_v(temp_c, rh_frac=rh_frac)
        rho_v_1 = np.where(np.isnan(rho_v_1), compute_rho_v(temp_c, chi_v=chi_v, pa=pa), rho_v_1)

        # Method B: compute rho_v first from chi_v then from rh, outsourcing the priotization to the function
        rho_v_2 = compute_rho_v(temp_c, chi_v=chi_v, pa=pa, rh_frac=rh_frac)
    ```

    Parameters
    ----------
    temp_c : array-like
        Temperature, in degrees C
    e : array-like, optional
        Vapor pressure (Pa)
    w : array-like, optional
        Mixing ratio of water vapor (kg/kg)
    pa : array-like, optional
        Ambient air pressure (Pa)
    rh_frac : array-like, optional
        Relative humidity (0-1)
    chi_v : array-like, optional
        Mole fraction of water vapor (mol/mol)
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.
    
    Returns
    -------
    rho_v : array-like
        Density of water vapor (kg/m^3)
    """
    if check_inputs:
        _check_temp_c(temp_c)
    temp_k = temp_c + 273.15

    if e is None:
        e = np.nan
    else:
        if check_inputs:
            _check_e_pa(e)
    if w is None:
        w = np.nan
    else:
        if check_inputs:
            _check_w_kgkg(w)
    if pa is None:
        pa = np.nan
    else:
        if check_inputs:
            _check_pa_pa(pa)
    if rh_frac is None:
        rh_frac = np.nan
    else:
        if check_inputs:
            _check_rh_frac(rh_frac)
    if chi_v is None:
        chi_v = np.nan

    rho_v = np.select(
        [
            np.isfinite(e),
            np.isfinite(w) & np.isfinite(pa),
            np.isfinite(chi_v) & np.isfinite(pa),
            np.isfinite(rh_frac)
        ],
        [
            e / (RV * temp_k),
            (w/EPS * pa) / (1 + w/EPS) / (RV * temp_k),
            (chi_v*pa) / (RV*temp_k),
            (rh_frac*compute_e_sat_water(temp_c)) / (RV * temp_k)
        ],
        default=np.nan
    )

    return _return_any(temp_c, rho_v)

def compute_rho_m(temp_c: Any, e: Any | None=None, w: Any | None=None, pa: Any | None=None, rh_frac: Any | None=None, chi_v: Any | None=None)->Any:
    """
    Compute the density of moist air (kg m-3).

    Ambient temperature is required as input, but many combinations of the other parameters can be used to compute partial of water vapor if it is not directly provided.

    Allowed combinations:
    * temp_c, e, pa (best)
    * temp_c, w, pa (best)
    * temp_c, chi_v, pa (best)
    * temp_c, rh_frac (less accurate, requires computing e_sat from temp_c)
    
    Additionally, all of the above can be provided, in which case the first valid combination will be used for each time point. (e.g. if w is not nan for index 0, then method 1 will be used for index 0. If w is nan for index 1 but vp and pa are not nan not nan, then method 2 will be used for index 1, etc.)

    Note
    ----
    Note that the combination priority reflects how accurate the algorithm is expected to be, not how accurate the final result will be.
    This matters if, for example, your measurement of chi_v comes from a less accurate instrument (such as a fast response infrared gas analyzer), but your rh measurement comes from a more accurate instrument (such as an aspirated, well-shielded, high-quality hygrometer).
    In this case, the rh_frac measurement may be more accurate than the chi_v measurement, and you may want to use the rh_frac method instead of the chi_v method:
    ```python
        # in this specific case, method A is more accurate than method B
        # Method A: compute rho_m first from rh then from chi_v
        rho_m_1 = compute_rho_m(temp_c, rh_frac=rh_frac)
        rho_m_1 = np.where(np.isnan(rho_m_1), compute_rho_m(temp_c, chi_v=chi_v, pa=pa), rho_m_1)

        # Method B: compute rho_m first from chi_v then from rh, outsourcing the priotization to the function
        rho_m_2 = compute_rho_m(temp_c, chi_v=chi_v, pa=pa, rh_frac=rh_frac)
    ```

    Parameters
    ----------
    temp_c : array-like
        Temperature, in degrees C
    e : array-like, optional
        Vapor pressure (Pa)
    w : array-like, optional
        Mixing ratio of water vapor (kg/kg)
    pa : array-like, optional
        Ambient air pressure (Pa)
    rh_frac : array-like, optional
        Relative humidity (0-1)
    chi_v : array-like, optional
        Mole fraction of water vapor (mol/mol)

    Returns
    -------
    rho_m : array-like
        Density of moist air (kg/m^3)
    """
    rho_d = compute_rho_d(temp_c, pa)
    rho_v = compute_rho_v(temp_c, e=e, w=w, pa=pa, rh_frac=rh_frac, chi_v=chi_v)
    rho_m = rho_d + rho_v
    return _return_any(temp_c, rho_m)

def compute_ta_from_tv(tv_c:Any, w:Any|None=None, rh_frac:Any|None=None, ta_c:Any|None=None, pa:Any|None=None, e:Any|None=None, check_inputs:bool=True)->Any:
    """
    Compute ambient temperature from virtual temperature

    .. math::
        T_v = T \frac{1 + w/0.622}{1+w}
    
    where :math:`T_v` is the virtual temperature, :math:`T` is the ambient temperature, and :math:`w` is the mixing ratio of water vapor.

    Virtual temperature is required as input, but many combinations of the other parameters can be used to compute the mixing ratio of water vapor if it is not directly provided.

    Allowed combinations:
    * tv, w (best)
    * tv, e, pa (best)
    * tv, rh, ta, pa (less accurate, requires computing e_sat from ta)
    * tv, rh, pa (least accurate, requires iteration and computing e_sat from ta)

    Additionally, all of the above can be provided, in which case the first valid combination will be used for each time point. (e.g. if w is not nan for index 0, then method 1 will be used for index 0. If w is nan for index 1 but vp and pa are not nan not nan, then method 2 will be used for index 1, etc.)
    
    Parameters
    ----------
    tv_c : array-like
        Virtual temperature (°C)
    w : array-like, optional
        Mixing ratio of water vapor (kg/kg)
    rh_frac : array-like, optional
        Relative humidity (0-1)
    ta_c : array-like, optional
        Ambient temperature (if already known from another source) (°C)
    pa : array-like, optional
        Air pressure (Pa)
    e : array-like, optional
        Vapor pressure (Pa)
    check_inputs : bool, optional
        Whether to perform a sanity check on the input values to ensure they are within reasonable ranges for the expected units. 
        Setting this to False will skip the checks, which may be useful for performance reasons if you are confident in the input data units. Default is True.

    Returns
    -------
    ta : array-like
        Ambient temperature (°C)
    """

    def ta_eqn(tv, w):
        return tv / ((1 + w/0.622)/(1 + w))

    def w_eqn(e, pa):
        return EPS * e / (pa - e)

    if w is None:
        w = np.nan
    else:
        if check_inputs:
            _check_w_kgkg(w)
    if e is None:
        e = np.nan
    else:
        if check_inputs:
            _check_e_pa(e)
    if pa is None:
        pa = np.nan
    else:
        if check_inputs:
            _check_pa_pa(pa)
    if ta_c is None:
        ta = np.nan
        ta_c = np.nan
    else:
        if check_inputs:
            _check_temp_c(ta_c)
        ta = ta_c + 273.15
    if check_inputs:
        _check_temp_c(tv_c)
    tv = tv_c + 273.15

    ta = np.select(
        [
            np.isfinite(w),
            np.isfinite(e) & np.isfinite(pa),
            np.isfinite(rh_frac) & np.isfinite(ta) & np.isfinite(pa),
            np.isfinite(rh_frac) & np.isfinite(pa)
        ],
        [
            ta_eqn(tv, w),
            ta_eqn(tv, w_eqn(e, pa)),
            ta_eqn(tv, w_eqn(rh_frac * compute_e_sat_water(ta_c), pa)),
            ta_eqn(tv, w_eqn(rh_frac * compute_e_sat_water(ta_eqn(tv, w_eqn(rh_frac * compute_e_sat_water(tv_c), pa))-273.15), pa))
        ],
        default=np.nan
    )

    return _return_any(tv, ta - 273.15)

__all__ = [  # noqa: RUF022
    "MV", "MC", "MD", "EPS", "R", "RV", "RC", "RD",
    "compute_Cp_d", "compute_Cp_v", "compute_Cp_m", 
    "compute_Lv", "compute_Lf", "compute_Ls", 
    "compute_e_sat_water", "compute_e_sat_ice", 
    "compute_rho_d", "compute_rho_c", "compute_rho_v", "compute_rho_m", 
    "compute_ta_from_tv", 
]