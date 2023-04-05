import numpy as np
import astropy.units as unit
import astropy.constants as const

def cstar(q, fpa, T, lam, dlam, Fstar, D):
    """
    Stellar photon count rate (not used with coronagraph)

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    fpa : float
        Fraction of planetary light that falls within photometric aperture
    T : float
        Telescope and system throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    Fstar : float or array-like
        Stellar flux [W/m**2/um]
    D : float
        Telescope diameter [m]

    Returns
    -------
    cs : float or array-like
        Stellar photon count rate [1/s]
    """
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fstar*(D/2.)**2.

def cplan(q, fpa, T, lam, dlam, Fplan, D):
    """
    Exoplanetary photon count rate (Equation 12 from Robinson et al. 2016)

    :math:`{c}_{{\\rm{p}}}=\\pi {{qf}}_{\\mathrm{pa}}{ \\mathcal T }\\displaystyle \\frac{\\lambda }{{hc}}{F}_{{\\rm{p}},\\lambda }(d)\\Delta \\lambda {\\left(\\displaystyle \\frac{D}{2}\\right)}^{2}`

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    fpa : float
        Fraction of planetary light that falls within photometric aperture
    T : float
        Telescope and system throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    Fplan : float or array-like
        Planetary flux [W/m**2/um]
    D : float
        Telescope diameter [m]

    Returns
    -------
    cplan : float or array-like
        Exoplanetary photon count rate [1/s]
    """
    hc  = 1.986446e-25 # h*c (kg*m**3/s**2)
    return np.pi*q*fpa*T*(lam*1.e-6/hc)*dlam*Fplan*(D/2.)**2.

def ctherm_earth(q, X, T, lam, dlam, D, Itherm, CIRC=False):
    """
    Earth atmosphere thermal count rate

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    T : float
        System throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    Itherm : float or array-like
        Earth thermal intensity [W/m**2/um/sr]
    CIRC : bool, optional
        Set to use a circular aperture

    Returns
    -------
    cthe : float or array-like
        Earth atmosphere thermal photon count rate [1/s]
    """
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Omega = Omega/(206265.**2.) # aperture size (sr**2)
    return np.pi*q*T*dlam*Itherm*Omega*(lam*1.e-6/hc)*(D/2)**2.

def ctherm(q, X, T, lam, dlam, D, Tsys, emis, CIRC=False):
    """
    Telescope thermal count rate

    Parameters
    ----------
    q : float or array-like
        Quantum efficiency
    X : float, optional
        Width of photometric aperture ( * lambda / diam)
    T : float
        System throughput
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    D : float
        Telescope diameter [m]
    Tsys  : float
        Telescope mirror temperature [K]
    emis : float
        Effective emissivity for the observing system (of order unity)
    CIRC : bool, optional
        Set to use a circular aperture


    Returns
    -------
    ctherm : float or array-like
        Telescope thermal photon count rate [1/s]
    """
    hc    = 1.986446e-25  # h*c (kg*m**3/s**2)
    c1    = 3.7417715e-16 # 2*pi*h*c*c (kg m**4 / s**3)
    c2    = 1.4387769e-2  # h*c/k (m K)
    lambd= 1.e-6*lam      # wavelength (m)
    power   = c2/lambd/Tsys
    Bsys  = c1/( (lambd**5.)*(np.exp(power)-1.) )*1.e-6/np.pi # system Planck function (W/m**2/um/sr)
    if CIRC:
        # circular aperture diameter (arcsec**2)
        Omega = np.pi*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    else:
        # square aperture diameter (arcsec**2)
        Omega = 4.*(X*lam*1e-6/D*180.*3600./np.pi)**2.
    Omega = Omega/(206265.**2.) # aperture size (sr**2)
    return np.pi*q*T*dlam*emis*Bsys*Omega*(lam*1.e-6/hc)*(D/2)**2.

def Fstar(fstar, R_star, a_p, dist):
    """
    

    Parameters
    ----------
    fstar : float or array-like
        Stellar flux [W/m**2/um]
    R_star : float
        radius of star in km
    a_p : float
        semi-major axis of planet [AU]
    dist : float
        distance to system [pc]

    Returns
    -------
    array
        stellar flux at earth

    """

    Bstar = fstar / ( np.pi*(R_star/\
                   (a_p.to(unit.km)))**2. )
    omega_star = np.pi*(R_star/\
                       (dist.to(unit.km)))**2.
    Fs = Bstar * omega_star # stellar flux at earth

    return Fs * (unit.W / unit.m**2 / unit.um)

def Fplan(fplan, R_plan, dist, Phi=1):
    """
    Parameters
    ----------
    fplan : float or array-like
        Planet flux [W/m**2/um]
    R_plan : float
        radius of planet in km

    dist : float
        distance to system [pc]

    Returns
    -------
    array
        planet flux at earth

    """

    Fp = fplan * Phi * (R_plan / dist.to(unit.km))**2
    return Fp * (unit.W / unit.m**2 / unit.um)
