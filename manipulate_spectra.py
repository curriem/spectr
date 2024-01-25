# manipulate_spectra.py
import numpy as np
import astropy.units as unit
import astropy.constants as const
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d


def doppler_shift(lam, flux, velocity):
    doppler_factor = np.sqrt((1-velocity*unit.km/unit.s/const.c)/(1 + velocity*unit.km/unit.s/const.c))
    new_lam = lam * doppler_factor
    f = interp1d(lam, flux, fill_value = "extrapolate")
    new_flux = f(new_lam)
    return new_flux

def broadening_kernel(delta_lambdas, delta_lambda_l, v_rot, epsilon):
    r"""Rotation kernel for a given line center.

    Parameters
    ----------
    delta_lambdas: array
        Wavelength difference from the line center lambda.
    delta_lambda_l: float
        Maximum line shift of line center by vsini.
    vsini: float
        Projected rotational velocity in km/s.
    epsilon: float
        Linear limb-darkening coefficient, between 0 and 1.

    Returns
    -------
    kernel: array
        Rotational kernel.

    Notes
    -----
    Gray, D. F. (2005). The Observation and Analysis of Stellar Photospheres. 3rd ed. Cambridge University Press.

    """
    denominator = np.pi * v_rot * (1.0 - epsilon / 3.0)
    lambda_ratio_sqr = (delta_lambdas / delta_lambda_l) ** 2.0

    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator
    kernel = c1 * np.sqrt(1.0 - lambda_ratio_sqr) + c2 * (1.0 - lambda_ratio_sqr)

    return kernel

def rotational_broadening(lam, flux, v_rot, epsilon, normalize=True):


    broadened_flux = []
    for single_lam in lam:
        delta_lambda_l = single_lam * v_rot / const.c.to("km/s").value
        rotational_profile = broadening_kernel(lam - single_lam, delta_lambda_l, v_rot, epsilon)
        convolved_flux = np.nansum(rotational_profile * flux)
        if normalize:
            # Correct for the effect of non-equidistant sampling
            unitary_rot_val = np.nansum(rotational_profile)
            convolved_flux /= unitary_rot_val

        broadened_flux.append(convolved_flux)

    broadened_flux = np.array(broadened_flux)

    return broadened_flux

def instrumental_broadening(specHR, lamHR, res):
    # gaussian broadening
    # adapted from https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/broad.py
    """
    Broadens a high res spectrum according to instrumental resolution

    Parameters
    ----------
    specHR : array-like
        Spectrum to be degraded
    lamHR : array-like
        High-res wavelength grid
    res : int
        Resolution of instrument
    """

    fwhm = 1.0 / float(res) * np.mean(lamHR)
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    len_lam = len(lamHR)
    dxs = lamHR[1:] - lamHR[0:-1]
    x_ker = (np.arange(len_lam, dtype=np.int) - np.sum(np.divmod(len_lam, 2)) + 1) * dxs[0]
    y_ker = 1 / np.sqrt(2.*np.pi*sigma**2) * np.exp(-(0 - x_ker)**2/(2.0 * sigma**2))

    y_ker /= np.sum(y_ker)
    result = np.convolve(specHR, y_ker, mode="same")

    return result

def construct_instrument_lam(wlmin, wlmax, instrument_R):
    dlam0 = wlmin/instrument_R
    dlam1 = wlmax/instrument_R
    lam  = wlmin #in [um]
    Nlam = 1
    while (lam < wlmax + dlam1):
        lam  = lam + lam/instrument_R
        Nlam = Nlam +1
    lam    = np.zeros(Nlam)
    lam[0] = wlmin
    for j in range(1,Nlam):
        lam[j] = lam[j-1] + lam[j-1]/instrument_R
    Nlam = len(lam)
    dlam = np.zeros(Nlam) #grid widths (um)

    # Set wavelength widths
    for j in range(1,Nlam-1):
        dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])

    #Set edges to be same as neighbor
    dlam[0] = dlam0#dlam[1]
    dlam[Nlam-1] = dlam1#dlam[Nlam-2]

    lam = lam[:-1]
    dlam = dlam[:-1]
    return lam, dlam

def bin_to_instrument_lam(flux, lam, instrument_lam, instrument_dlam):
        # Reverse ordering if wl vector is decreasing with index
    if len(instrument_lam) > 1:
        if lam[0] > lam[1]:
            lamHI = np.array(lam[::-1])
            spec = np.array(flux[::-1])
        if instrument_lam[0] > instrument_lam[1]:
            lamLO = np.array(instrument_lam[::-1])
            dlamLO = np.array(instrument_dlam[::-1])

    # Calculate bin edges
    LRedges = np.hstack([instrument_lam - 0.5*instrument_dlam, instrument_lam[-1]+0.5*instrument_dlam[-1]])

    # Call scipy.stats.binned_statistic()

    instrument_spec = binned_statistic(lam, flux, statistic="mean", bins=LRedges)[0]
    return instrument_spec
