"""
Spectral analysis functions for flux data.

1. First iteration using spectral model M0:
a. Use raw H and U* to construct cospectral model M0
b. Use M0 to apply freq response corrections (FR) to raw LE, obtain LE0
c. Apply Webb-Pearman-Leuning (WPL) to LE0, obtain LE1
d. Use LE1 to apply sonic correction to H, obtain H0
e. Apply FR correction to H0, obtain H1 and U*0

2. Second iteration using spectral model M1:
a. Use H1 and U*0 to construct cospectral model M1
b. Use M1 to apply FR corrections to LE and FC, obtain LE2 and FC0
c. Use H1 to apply WPL correction to LE2, obtain LE3
d. Apply Oxygen and Band-Broadening (BB) corrections to LE3 if needed to obtain LE4
e. Use H1 and LE4 to apply WPL correction to FC0, obtain FC1
f. Apply BB correction to FC1 if needed, obtain FC2

3. Final outputs: H1, LE4, FC2, U*0
"""

# TODO Reference Section 4.2 of burba textbook

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal, stats, special, optimize, integrate

def _interp(x):
    idx = np.arange(x.shape[0])
    nans = ~np.isfinite(x)
    idx_nans = idx[nans]
    idx_notna = idx[~nans]
    interp = np.interp(x=idx_nans, xp=idx_notna, fp=x[idx_notna])
    x_interp = x.copy()
    x_interp[nans] = interp
    return x_interp

def compute_spectrum(
    x1: np.ndarray, x2: np.ndarray | None = None,
    acq_freq: float = 10.0,
)->tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the one-sided Co- or Power spectrum of given time series using FFT.

    Any missing values in the input time series are linearly interpolated before computing the spectrum.

    ::math::
        \text{Co}_{x_1, x_2}(f) = \frac{2}{N f} \Re(\text{FFT}(x_1 - \bar{x}_1) \cdot \text{FFT}(x_2 - \bar{x}_2)^*)

    Parameters
    ----------
    x1 : np.ndarray
        The input time series data.
    x2 : np.ndarray | None
        The second input time series data (for co-spectrum). If None, compute power spectrum of x1.
    acq_freq : float
        The acquisition frequency of the time series data (Hz)

    Returns
    -------
    freqs : np.ndarray
        The one-sided frequency bins corresponding to the computed spectrum (Hz)
    spectrum : np.ndarray
        The computed one-sided co- or power spectrum values corresponding to the frequency bins
    """
    n = x1.shape[0]

    x1 = _interp(x1)
    if x2 is not None:
        x2 = _interp(x2)

    fft1 = np.fft.fft(x1 - np.mean(x1))
    if x2 is None:
        fft2 = fft1
    else:
        fft2 = np.fft.fft(x2 - np.mean(x2))

    # compute the 1 sided spectrum
    spectrum = np.real(fft1 * np.conj(fft2))[:n//2] * 2 / (acq_freq * n)
    freqs = np.fft.fftfreq(n, d=1/acq_freq)[:n//2]
    return freqs, spectrum

def bin_spectrum(freqs: np.ndarray, spectrum: np.ndarray, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin the spectrum into logarithmically spaced frequency bins.

    Parameters
    ----------
    freqs : np.ndarray
        The frequency bins corresponding to the computed spectrum (Hz)
    spectrum : np.ndarray
        The computed one-sided co- or power spectrum values corresponding to the frequency bins
    n_bins : int
        The number of logarithmically spaced bins to use for binning

    Returns
    -------
    binned_freqs : np.ndarray
        The binned frequency values (geometric mean of bin edges for each bin)
    binned_spectrum : np.ndarray
        The binned spectrum values (mean of each bin)
    """
    # create logarithmically spaced bins
    log_bins = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_bins + 1)
    binned_spectrum, bin_edges, _ = stats.binned_statistic(
        freqs, spectrum, statistic='mean', bins=log_bins
    )
    binned_freqs = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean of bin edges
    return binned_freqs, binned_spectrum

def scale_spectrum(freqs: np.ndarray, spectrum: np.ndarray, cov: float):
    r"""
    Normalize the (co)spectrum by the (co)variance of the two time series and scale to preserve area under log-plots.

    ::math::
        \text{CoScaled}_{x_1, x_2}(f) = \frac{f \cdot \text{Co}_{x_1, x_2}(f)}{\text{Cov}(x_1, x_2)}

    Parameters
    ----------
    freqs : np.ndarray
        The frequency bins corresponding to the computed spectrum (Hz)
    spectrum : np.ndarray
        The computed one-sided co- or power spectrum values corresponding to the frequency bins
    cov : float
        The covariance of the two time series (or variance if x2 is None)
    
    Returns
    -------
    scaled_spectrum : np.ndarray
        The scaled co- or power spectrum values corresponding to the frequency bins
    """
    return spectrum * freqs/cov

def nondimensionalize_freq(freqs: np.ndarray, z_m: float, z_d: float, u: float) -> np.ndarray:
    r"""
    Compute the non-dimensional frequency for a given acquisition frequency, measurement height, displacement height, and wind speed.

    ::math::
        f_{nd} = \frac{f \cdot (z_m - z_d)}{u}

    Parameters
    ----------
    freqs: np.ndarray
        The natural frequency bins (Hz) for which to compute the non-dimensional frequency.
    z_m : float
        The measurement height (m)
    z_d : float
        The displacement height (m)
    u : float
        The wind speed (m/s)

    Returns
    -------
    f_nd : np.ndarray
        The non-dimensional frequency bins
    """
    return (freqs * (z_m - z_d)) / u

# ######################
# Transfer function system
# \frac{\Delta F_x}{F_x} = 1 - \frac{\int_0^\infty T_{w\rho_x}\text{Co}_{w\rho_x}(f_{\text{nat}})\text{d}f_{\text{nat}}}{\int_0^\infty \text{Co}_{w\rho_x}(f_{\text{nat}})\text{d}f_{\text{nat}}}
# Where T_{w\rho_x} is the convolution of all transfer functions applicable to the measurement.
# So T_{w\rho_x} = T_1 \star T_2 \star ... \star T_n. Note that the transfer functions are defined in frequency domain,
# So we simply define the transfer functions in frequency domain and multiply them together to get the total transfer function, by the convolution theorem.
# ######################

# J.B. Moncrieff et aL/Journal of Hydrology 188-189 (1997) 589-611
class TransferFunction:
    """
    Base class for transfer functions.

    This class allows for the composition of multiple transfer functions into a single transfer function.

    Each transfer function should be a callable that takes a natural frequency array as the first argument, followed by any additional positional and keyword arguments. The transfer functions should return an array of the same shape as the input natural frequency array.

    Example
    -------
    .. code-block:: python
        import numpy as np
        import matplotlib.pyplot as plt

        def my_transfer_function_1(f_nat):  # sinc function
            return np.sin(f_nat)/f_nat
        def my_transfer_function_2(f_nat, a):  # exponential decay
            return np.exp(-a*f_nat)
        
        tf_system = TransferFunction()
        tf_system.add(my_transfer_function_1)
        tf_system.add(my_transfer_function_2, a=0.1)
        total_transfer_function = tf_system.T

        f_nat = np.linspace(0.01, 10, 500)
        plt.plot(f_nat, my_transfer_function_1(f_nat), label='TF1')
        plt.plot(f_nat, my_transfer_function_2(f_nat, a=0.1), label='TF2')
        plt.plot(f_nat, total_transfer_function(f_nat), label='Total TF')
        plt.legend()
        plt.xlabel('Natural Frequency (Hz)')
        plt.ylabel('Transfer Function Magnitude')
        plt.title('Transfer Function System')
        plt.show()
    """
    def __init__(self):
        self._functions: list[tuple[Callable[..., np.ndarray], dict]] = []
        self._T: Callable[..., np.ndarray] = lambda f: np.ones_like(f)

    @staticmethod
    def _compose(prev_T: Callable[..., np.ndarray], func: Callable[..., np.ndarray], kwargs: dict, f: np.ndarray) -> np.ndarray:
        return prev_T(f) * func(f, **kwargs)

    def add(self, func, **kwargs) -> None:
        """
        Add a transfer function to the system.
        """
        self._functions.append((func, kwargs))
        self._T = partial(self._compose, self._T, func, kwargs)

    @property
    def T(self) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Return the total transfer function as a callable function of frequency.

        .. math::
            T_\text{total}(f) = \prod_{i=1}^{n} T_i(f)
        """
        return self._T

    def compute_scf(self, co_wx_ref: Callable[[np.ndarray], np.ndarray]) -> float:
        r"""
        Compute the spectral correction factor (SCF) for the given natural frequency array and cospectrum.

        .. math::
            \begin{align}
                F_\text{ref} &= \text{SCF} \cdot F_\text{meas} \\
                \text{SCF} &= \int_0^\infty\text{Co}_\text{ref}(f_\text{nat})\text{d}f_\text{nat}\Bigg/\int_0^\infty\text{Co}_\text{ref}(f_\text{nat})\prod_i T_i(f_\text{nat})\text{d}f_\text{nat}
            \end{align}

        Parameters
        ----------
        co_wx_ref : Callable[[np.ndarray], np.ndarray]
            Reference (theoretical) cospectrum corresponding to the natural frequency array

        Returns
        -------
        float
            Spectral correction factor.
        """
        # integrate from 1e-4 Hz (~3hr) to 100Hz to approach the theoretical limits of the integral
        freqs = np.logspace(-4, 2, 1000)
        num = np.trapezoid(y=co_wx_ref(freqs), x=freqs)
        denom = np.trapezoid(y=self.T(freqs) * co_wx_ref(freqs), x=freqs)
        SCF = num / denom
        return SCF

def T_time_response(f_nat: np.ndarray, tau: float) -> np.ndarray:
    r"""
    Time response corrections compensate for the loss of flux due to inability of sensors to respond fast enough to small fluctuations that contribute to the flux.

    Applies to gas fluxes.

    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Burba, G, Eddy Covariance Method. Section 4.2

    .. math::
        T(f_\text{nat}) = \frac{1}{\sqrt{1 + (2 \pi f_\text{nat} \tau)^2}}
    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    tau : float
        Sensor dynamic time response (s)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    return 1 / np.sqrt(1 + (2 * np.pi * f_nat * tau) ** 2)

def T_tube_attenuation(f_nat: np.ndarray, r: float, L: float, D: float, u_tube: float) -> np.ndarray:
    r"""
    Simple analytic solution for tube attenuation transfer function. More advanced models for tube attenuation are found below.

    Required for closed-path gas fluxes.

    Burba, G, Eddy Covariance Method. Section 4.2

    .. math::
        T(f_\text{nat}) = \exp\left(-\frac{\pi^2 r^2 f_\text{nat}^2 L}{D}\right)
    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    r : float
        Tube radius (m)
    L : float
        Tube length (m)
    D : float
        Diffusion coefficient (m^2/s)
    u_tube : float
        Mean flow velocity in the tube (m/s)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    return np.exp(-4*np.pi**2 * D * r * L * f_nat**2 / u_tube**2)

def T_scalar_path_averaging(f_nat: np.ndarray, L_path: float, u: float) -> np.ndarray:
    r"""
    Path or volume averaging corrections compensate for the loss of flux due to averaging of very small eddies.

    Applies to all scalar fluxes (not momentum fluxes), which require vector path averaging.

    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Burba, G, Eddy Covariance Method, Section 4.2

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    L_path : float
        Path length (m). Note that this is different from the physical distance between the two sensors. For example, some optical instruments may have enourmous optical path lengths, since the light beam may reflect multiple times within the instrument before reaching the detector. Consult the manufacturer determining the path length.
    u : float
        Mean wind speed (m/s)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    denom = 2*np.pi*f_nat*L_path/u
    num_1 = 3
    num_2 = np.exp(-2*np.pi*f_nat*L_path/u)
    num_3 = 4/(2*np.pi*f_nat*L_path/u) * (1 - np.exp(-2*np.pi*f_nat*L_path/u))
    return np.sqrt(np.sqrt((num_1 + num_2 + num_3)/denom))  # sqrt twice because this function is developed for the power spectrum (loss in signal squared)

def T_vector_w_path_averaging(f_nat: np.ndarray, L_path: float, u: float) -> np.ndarray:
    r"""
    Path or volume averaging corrections compensate for the loss of flux due to averaging of very small eddies.

    Applies to vertical wind speeds, which require vector path averaging.

    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    L_path : float
        Path length (m). Note that this is different from the physical distance between the two sensors. For example, some optical instruments may have enourmous optical path lengths, since the light beam may reflect multiple times within the instrument before reaching the detector. Consult the manufacturer determining the path length.
    u : float
        Mean wind speed (m/s)
    
    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """

    f_p = f_nat*L_path/u
    # return 2/(np.pi*f_p) * (1 + np.exp(-2*np.pi*f_p)/2 - 3*(1 - np.exp(-2*np.pi*f_p))/(4*np.pi*f_p))
    return np.sqrt(2/(np.pi*f_p) * (1 + np.exp(-2*np.pi*f_p)/2 - 3*(1 - np.exp(-2*np.pi*f_p))/(4*np.pi*f_p)))  # TODO: do we return sqrt here? or do we return the full transfer function. Unclear.

def T_sensor_separation(f_nat: np.ndarray, L_sep: float, u: float) -> np.ndarray:
    r"""
    Compensates for the loss of flux due to the inability of the vertical wind speed and scalar sensors to sample in the same volume.

    Applies to fluxes across instruments: ie LE ad FC.

    Moncrieff et al. 1997. A refined method for the determination of scalar fluxes from the profile technique. Journal of Hydrology, 188-189, 589-611. https://doi.org/10.1016/S0022-1694(96)03128-1

    Burba, G, Eddy Covariance Method, Section 4.2

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    L_sep : float
        Sensor separation distance (m)
    u : float
        Mean wind speed (m/s)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    f_s = f_nat * L_sep / u
    return np.exp(-9.9 * f_s**1.5)

def T_sensor_dynamic_f_resp_mismatch(
    f_nat: np.ndarray, tau_1: float, tau_2: float
) -> np.ndarray:
    r"""
    Sensor dynamic frequency response mismatch transfer function from Moncrieff et al. (1997).

    Sometimes used in data processing programs to compensate for differences when slower-response and faster-response instruments are used together.

    Burba, G, Eddy Covariance Method, Section 4.2

    .. math::
        T(f_\text{nat}) = \frac{
            1 + (2\pi f_\text{nat})^2 \tau_{1} \tau_{2}
        }{
            \sqrt{\left[1 + (2\pi f_\text{nat} \tau_{1})^2\right]+\left[1 + (2\pi f_\text{nat} \tau_{2})^2\right]}
        }

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    tau_1 : float
        Sensor response time 1 (s)
    tau_2 : float
        Sensor response time 2 (s)
    
    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    num = 1 + (2*np.pi*f_nat)**2 * tau_1 * tau_2
    denom_f1 = 1 + (2*np.pi*f_nat*tau_1)**2
    denom_f2 = 1 + (2*np.pi*f_nat*tau_2)**2
    denom = np.sqrt(denom_f1 + denom_f2)
    return num / denom

def T_lowpass_filter(f_nat: np.ndarray, tau_f: float, f_nyquist: float) -> np.ndarray:
    r"""
    Compensates for loss of flux in the low frequency part of cospectrum due to time averaging, linear de-trend- ing, or non-linear filtering/

    Applies to all fluxes, example is a recursive low-pass filter.

    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Burba, G, Eddy Covariance Method, Section 4.2

    # TODO: what is tau_f??
    # TODO: is n_cutoff the same as f_nyquist? If not, what is it?

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    tau_f : float
        Filter time constant (s)
    f_nyquist : float
        Nyquist frequency (Hz) (1/2 of the acquisition frequency)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """

    num = 2*np.pi*f_nat*tau_f

    root_num = 1 + (2*np.pi*tau_f)**2
    root_denom = 1 + 1/(tau_f*f_nyquist)

    return 1 - num / np.sqrt(root_num/root_denom)


def T_highpass_filter(f_nat: np.ndarray, tau_f: float, f_nyquist: float) -> np.ndarray:
    r"""
    Compensates for loss of flux in the low frequency part of cospectrum due to time averaging, linear de-trending, or non-linear filtering

    Applies to all fluxes.


    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Burba, G, Eddy Covariance Method, Section 4.2

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    tau_f : float
        Filter time constant (s)
    f_nyquist : float
        Nyquist frequency (Hz) (1/2 of the acquisition frequency)
    
    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    return 1 - T_lowpass_filter(f_nat, tau_f, f_nyquist)

def T_digital_sampling(f_nat: np.ndarray, f_acq: float) -> np.ndarray:
    r"""
    Compensates for aliasing during digital sampling. Applies to all fluxes.

    Moore, C.J. 1986. Frequency response corrections for eddy correla- tion systems. Boundary-Layer Meteorology, 37, 17-35. https://doi. org/10.1007/bf00122754

    Burba, G, Eddy Covariance Method, Section 4.2

    Warning
    -------
    Digital sampling corrections are often assumed negligible for modern instruments. However, caution should be exercised when experimenting with novel or custom-made instruments, or non-standard settings and conditions.

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    f_acq : float
        Acquisition frequency (Hz)

    Returns
    -------
    np.ndarray
        Transfer function values corresponding to the input frequency array.
    """
    return np.where(f_nat <= f_acq/2, 1 + (f_nat/(f_acq - f_nat))**3, 1.0)


# Tall = Tt1 * Tt1 * Ttube * Tds * Ts * Thi * Thi * Tsp * Tvp
# TODO Questions remain.... why do some of these terms need to be there for each signal (Tt, Tp, Thi), but others (Ttube, Tds) are only applied once? Both signals are digitally sampled, so it would seem that Tds should appear twice. Ttube makes sense because we often only measure one of the components with a tube.
# Better:
# T_signal_1 = T_time_1 * T_tube * T_highpass_filter * T_scalar_path_averaging  # transfer functions affecting the first signal (in this case a scalar in a closed path system)
# T_signal_2 = T_time_2 * T_highpass_filter * T_vector_w_path_averaging  # transfer functions affecting the second signal (in this case vertical wind speed)
# T_cross_signal = T_sensor_separation * T_sensor_dynamic_f_resp_mismatch  # transfer functions resulting from the combining of two signals
# T_all = T_signal_1 * T_signal_2 * T_cross_signal  # final transfer function is the product of all transfer functions affecting the two signals

def spectrum_ref_kaimal_72(f_nat: np.ndarray, spectrum_type: Literal['T', 'w', 'u', 'x', 'wT', 'wu', 'wx'], u: float, z_m: float, z_d: float, L: float) -> np.ndarray:
    """
    Spectral model from Moncrieff et al (1997) based on Kaimal et al. (1972) and Moore et al (1986).

    Moncrieff et al. "A system to measure surface fluxes of momentum, sensible heat, water vapour and carbon dioxide." Journal of Hydrology 188-189 (1997) 589-611.

    Kaimal, J.C., Wyngaard, J.C., lzumi, Y. and Cote, O.R., 1972. Spectral characteristics of surface-layer turbulence. Q. J. R. Meteorol. Soc., 98: 563-589. 

    Moore, C.J., 1986. Frequency response corrections for eddy correlation systems. Boundary-Layer Meteorol., 37:17-35. 

    Parameters
    ----------
    f_nat : np.ndarray
        Natural frequency array (Hz)
    spectrum_type: Literal['T', 'w', 'u', 'x', 'wT', 'wu', 'wx']
        Type of spectrum to compute. Options are
        * 'T' for temperature power spectrum
        * 'w' for vertical wind power spectrum
        * 'u' for horizontal wind power spectrum
        * 'x' for a gas scalar power spectrum (e.g. CO2, H2O)
        * 'wT' for the co-spectrum of vertical wind and temperature
        * 'wu' for the co-spectrum of vertical wind and horizontal wind (u*, Reynolds stress)
        * 'wx' for the co-spectrum of vertical wind and a gas scalar (e.g. CO2, H2O)
    u : float
        Mean wind speed (m/s)
    z_m : float
        Measurement height (m)
    z_d : float
        Displacement height (m)
    L : float
        Monin-Obukhov length (m)

    Returns
    -------
    np.ndarray
        Cospectrum values corresponding to the input frequency array.
    """

    f_norm = nondimensionalize_freq(f_nat, z_m, z_d, u)
    z = z_m - z_d
    zL = z / L

    # stable case
    if L > 0:
        match spectrum_type:
            case 'T':
                A = 0.0961 + 0.644*zL**0.6  # Moore
            case 'w':
                A = 0.838 + 1.172*zL  # Moore
            case 'u':
                A = 0.2 * 0.838 + 1.172*zL  # Moore
            case 'x':
                raise NotImplementedError("Gas scalar spectrum not implemented for stable conditions.")
            case 'wT':
                A = 0.284*(1 + 6.4*zL)**0.75  # Moore
            case 'wu':
                A = 0.124*(1 + 7.9*zL)**0.75  # Moore
            case 'wx':
                A = 0.284*(1 + 6.4*zL)**0.75  # Moncrieff
            case _:
                raise ValueError(f"Invalid spectrum_type: {spectrum_type}. Must be one of 'T', 'w', 'u', 'x', 'wT', 'wu', 'wx'.")

        if spectrum_type in ['T', 'w', 'u', 'x']:
            B = 3.124*A**(-2/3)  # Moore
            out = f_norm / (A + B*f_norm**(5/3)) / f_nat  # Moore
        elif spectrum_type in ['wT', 'wu']:
            B = 2.34*A**(-1.1)  # Moore
            out = f_norm / (A + B*f_norm**2.1) / f_nat  # Moore
        elif spectrum_type == 'wx':
            B = 2.34*(0.284*(1 + 6.4*zL)**0.75)**(-1.1)  # 2.34 A_wt^-1.1  # TODO: is this a typo? should it not be the same as the wT and wu case? Moncrieff highlights a typo from Moore, but then maybe introduces a typo of their own?
            out = f_norm / (A + B*f_norm**2.1) / f_nat  # Moncrieff

    # unstable case
    elif L < 0:
        match spectrum_type:
            case 'T':
                out = np.where(
                    f_norm < 0.15,
                    f_norm * 14.94 / (1 + 24.0*f_norm)**(5/3),
                    f_norm * 6.827 / (1 + 12.5*f_norm)**(5/3)
                ) / f_nat  # Moore
            case 'w':
                raise NotImplementedError("Vertical wind spectrum not implemented for unstable conditions.")
            case 'u':
                raise NotImplementedError("Horizontal wind spectrum not implemented for unstable conditions.")
            case 'x':
                raise NotImplementedError("Gas scalar spectrum not implemented for unstable conditions.")
            case 'wT':
                out = np.where(
                    f_norm < 0.54,
                    f_norm * 12.92 / (1 + 26.7*f_norm)**1.375,
                    f_norm * 4.378 / (1 + 3.8*f_norm)**2.4
                ) / f_nat  # Moore
            case 'wu':
                out = np.where(
                    f_norm < 0.24,
                    f_norm * 20.78 / (1 + 31.0*f_norm)**1.575,
                    f_norm * 12.66 / (1 + 9.6*f_norm)**2.4
                ) / f_nat  # Moncrieff
            case 'wx':
                out = np.where(
                    f_norm < 0.54,
                    f_norm * 12.92 / (1 + 26.7*f_norm)**1.375,
                    f_norm * 4.378 / (1 + 3.8*f_norm)**2.4
                ) / f_nat  # Moncrieff

    return out

if __name__ == "__main__":
    from camp2ascii import camp2ascii
    import pandas as pd
    import matplotlib.pyplot as plt

    day_df: pd.DataFrame = next(camp2ascii("/Users/alex/Desktop/2992.CPk_BBSF4m_10Hz216.dat", output_dir="/tmp", output_format=4)).set_index("TIMESTAMP")

    for tstart, df in day_df.resample("30min"):
        if df.shape[0] < 10_000:
            continue
        ubar = np.sqrt(df["Ux"]**2 + df["Uy"]**2).mean()
        ts = df["Ts"]
        w = df["Uz"]

        freqs, co_w_ts = compute_spectrum(w, ts, acq_freq=10.0)
        cov = np.cov(df[["Uz", "Ts"]].dropna().to_numpy(), rowvar=False, ddof=0)[0, 1]

        co_w_ts_kaimal = spectrum_ref_kaimal_72(freqs, spectrum_type='wT', u=ubar, z_m=17.0, z_d=15.0, L=-10.0)
        cov_kaimal = 1.0
        
        transfer = TransferFunction()
        transfer.add(T_sensor_dynamic_f_resp_moncrieff_97, t_resp=1/60)
        transfer.add(T_sonic_path_avg_moncrieff_97, u=ubar, l_sonic=0.1)
        transfer.add(T_sensor_separation_moncrieff_97, u=ubar, l_sep=0.25)
        T_total = transfer.T(freqs)
        co_w_ts_kaimal_with_transfer = co_w_ts_kaimal * T_total

        co_w_ts_corrected = co_w_ts/T_total

        scf = transfer.compute_scf(co_wx_ref=lambda f: spectrum_ref_kaimal_72(f, spectrum_type='wT', u=ubar, z_m=17.0, z_d=15.0, L=-10.0))

        binned_freqs, binned_co_w_ts = bin_spectrum(freqs, co_w_ts, n_bins=50)
        _, binned_T_total = bin_spectrum(freqs, T_total, n_bins=50)
        _, binned_co_w_ts_kaimal = bin_spectrum(freqs, co_w_ts_kaimal, n_bins=50)
        _, binned_co_w_ts_kaimal_with_transfer = bin_spectrum(freqs, co_w_ts_kaimal_with_transfer, n_bins=50)
        _, binned_co_w_ts_corrected = bin_spectrum(freqs, co_w_ts_corrected, n_bins=50)

        binned_nondim_freqs = nondimensionalize_freq(binned_freqs, z_m=17.0, z_d=15.0, u=ubar)

        plt.plot(binned_nondim_freqs, np.abs(scale_spectrum(binned_nondim_freqs, binned_co_w_ts, cov)), color="C0", lw=0.3, alpha=0.5)#, label="Measured wT spectrum")
        plt.plot(binned_nondim_freqs, np.abs(scale_spectrum(binned_nondim_freqs, binned_co_w_ts_kaimal, cov_kaimal)), color="C1", lw=0.3, alpha=0.5)#, label="Kaimal 1972 wT spectrum")
        plt.plot(binned_nondim_freqs, np.abs(scale_spectrum(binned_nondim_freqs, binned_co_w_ts_kaimal_with_transfer, 1.0)), color="C2", lw=0.3, alpha=0.5)#, label="Kaimal 1972 wT spectrum with transfer functions")
        plt.plot(binned_nondim_freqs, np.abs(scale_spectrum(binned_nondim_freqs, binned_co_w_ts_corrected, cov)), color="C3", lw=0.3, alpha=0.5)#, label="Corrected wT spectrum")

    # # plt.plot(nondim_freqs, np.abs(scaled_co_w_ts), lw=0.1, alpha=0.5)
    # plt.plot(binned_nondim_freqs, np.abs(binned_scaled_co_w_ts), color="C0")
    # plt.plot(binned_nondim_freqs, np.abs(binned_co_w_ts*co_theory_with_transfer/co_w_ts_theory), color="C5", label="Measured wT spectrum with SCF")

    # plt.plot(freqs, np.abs(co_w_ts_theory)*nondim_freqs/cov_theory, color="C3", label="Kaimal 1972 wT spectrum")
    # plt.plot(freqs, np.abs(co_theory_with_transfer)*nondim_freqs/cov_theory, color="C4", label="Kaimal 1972 wT spectrum with transfer functions")
    # plt.plot(binned_nondim_freqs, np.abs(binned_T_total)*binned_nondim_freqs/cov_theory, color="C2", label="Total Transfer Function")

    plt.plot(np.logspace(-1, 1, 101), 0.2*np.logspace(-1, 1, 101)**(-4/3), color="k", ls=':', label="-4/3 slope")


    plt.xlabel("Non-dimensional Frequency")
    plt.ylabel("Co-spectrum")
    plt.title("Co-spectrum of Uz and Ts")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.001, 10.0)
    plt.ylim(0.0001, 10.0)
    plt.legend()
    plt.show()
    
    # # test the function
    # acq_freq = 10.0
    # t = np.arange(0, 100, 1/acq_freq)
    # x1 = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t)) + np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(len(t))
    # x2 = np.cos(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t)) + np.cos(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(len(t))

    # freqs, spectrum = compute_spectrum(x1, x2, acq_freq=acq_freq)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(freqs, spectrum)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Spectrum")
    # plt.title("Co-spectrum of x1 and x2")
    # plt.show()

