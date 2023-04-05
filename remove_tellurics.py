# this is where telluric removal code goes
import numpy as np
from scipy.interpolate import splrep, splev


def standard(spec):
    """
    Removes tellurics using airmass detrending

    Parameters
    ----------
    spec : numpy array of shape (n_orders, n_spectra, n_wavelengths)
        observed spectrum 


    Returns
    -------
    detrended_spec : numpy array of shape (n_orders, n_spectra, n_wavelengths)
        The spectrum with NPC principal components removed

    """
    #sp_wave = spec[0,1,:].copy()
    no, nf, nx = spec.shape  # Getting dimensions
    yy = np.arange(nf)       # Used to detrend in time (see loop below)
    for io in range(no):
        #spec[io,:,[0,1,1022,1023]] = 1.0
        spc = spec[io,].copy()     # Slicing detector
        # Creating the time average of the spectra - robust against NaNs
        mspc = np.nanmedian(spc,axis=0)
        #mspc = np.max(spc,axis=0)
        # looping over phases
        for j in range(nf):
            spc = spec[io,j,].copy()

            # Excluding NaNs
            iok = np.isfinite(spc)

            # Computing the 2nd order fit on the finite elements only
            coef = np.polyfit(mspc[iok],spc[iok],2)
            fit = coef[0]*mspc**2 + coef[1]*mspc + coef[2]

            # Divide by the fit
            spec[io,j,] /= fit

        # Loop over wavelfdengths to detrend in time
        for i in range(nx):
            sp_chann = spec[io,:,i].copy()
            iok = np.isfinite(sp_chann)
            if np.sum(iok) > 0:
                coef = np.polyfit(yy[iok],sp_chann[iok],2)
                fit = coef[0]*yy**2 + coef[1]*yy + coef[2]
            else:
                fit = np.ones_like(yy)
            spec[io,:,i] /= fit
        # Masking the edges (almost always bad)
        
    detrended_spec = spec - 1.0
    return detrended_spec

def PCA(spec, NPC):
    """
    Runs a principal component analysis to remove telluric and stellar lines 
    from a spectrum

    Parameters
    ----------
    spec : numpy array of shape (n_orders, n_spectra, n_wavelengths)
        observed spectrum 
    NPC : int
        number of principal components to remove

    Returns
    -------
    detrended_spec : numpy array of shape (n_orders, n_spectra, n_wavelengths)
        The spectrum with NPC principal components removed

    """

    detrended_spec = np.empty_like(spec)

    no, nph, nlam = spec.shape

    for order in range(no):
        ######## perform PCA on "data" to separate star from planet ########
        u, s, vh = np.linalg.svd(spec[order], full_matrices=False)  #decompose

        # pull out star and tellurics
        s_first_PCs = np.copy(s)
        s_first_PCs[NPC:] = 0. # set first PCs to zero
        W_first_PCs = np.diag(s_first_PCs)
        A_first_PCs = np.dot(u, np.dot(W_first_PCs,vh))

        # pull out planet and noise: the rest of the eigenvectors
        s_last_PCs = np.copy(s)
        s_last_PCs[:NPC] = 0. # set last PCs to zero
        W_last_PCs = np.diag(s_last_PCs)
        A_last_PCs = np.dot(u, np.dot(W_last_PCs,vh))

        #sigma clipping the noise
        sig = np.std(A_last_PCs)
        med = np.median(A_last_PCs)
        loc = np.where(A_last_PCs > 3.*sig+med)
        A_last_PCs[loc] = 0 #*0.+20*sig
        loc = np.where(A_last_PCs < -3.*sig+med)
        A_last_PCs[loc] = 0 #*0.+20*sig
        ######################################################################
        detrended_spec[order] = A_last_PCs

    return detrended_spec


# ADD SPORK IN

# ADD SYSREM IN
