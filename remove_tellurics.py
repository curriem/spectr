# this is where telluric removal code goes
import numpy as np
from scipy.interpolate import splrep, splev


def standard(spec, **kwargs):
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
    return spec-1.0

def PCA(spec, **kwargs):
    NPC = kwargs["NPC"]

    detrended_spec = np.empty_like(spec)

    no, nph, nlam = spec.shape

    for order in range(no):
        ######## perform PCA on "data" to separate star from planet ########
        u, s, vh = np.linalg.svd(spec[order], full_matrices=False)  #decompose

        # pull out star and tellurics
        s_first_PCs = np.copy(s)
        s_first_PCs[NPC:] = 0. # one eigenvector is good for this simple calculation
        W_first_PCs = np.diag(s_first_PCs)
        A_first_PCs = np.dot(u, np.dot(W_first_PCs,vh))

        # pull out planet and noise: the rest of the eigenvectors
        s_last_PCs = np.copy(s)
        s_last_PCs[:NPC] = 0. # one eigenvector is good for this simple calculation
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

#### PREFECTLY REMOVED TELLURICS


def perfect(spec, **kwargs):
    
    def mask_outliers_phase_dependent(arr):
        
        norders, nph, nlam = arr.shape
        
        masked_arr = np.empty_like(arr)
        
        for no in range(norders):
            for phase in range(nph):
                arr_temp = outlier_mask(arr[no, phase, :])
                arr_filled, outlier_inds = fill_nans(arr_temp)
                hipass_arr = hipass_single(arr_filled)
                hipass_arr[outlier_inds] = 0
                masked_arr[no, phase, :] = hipass_arr
                
        return masked_arr
        
    def outlier_mask(data):
        std_cut = 3
        """mask outliers with NaN"""
        data_copy = np.copy(data)
        bin_width = 100
        temp = np.where((data_copy > 1) | (data_copy < -1))
        data_copy[temp] = np.nan
        for i in range(len(data) - bin_width + 1):
            map_inds = np.arange(i, i+bin_width, 1, dtype=int)
            current_bin = data_copy[i:i+bin_width]
            current_std = np.nanstd(current_bin)
            current_med = np.nanmedian(current_bin)

            current_outliers = (current_bin > current_med + std_cut*current_std) | (current_bin < current_med - std_cut*current_std)

            outlier_inds = map_inds[current_outliers]
            data_copy[outlier_inds] = np.nan
        return data_copy

    def fill_nans(data):
        all_outliers = np.where(~np.isfinite(data))
        data_filled = np.copy(data)
        for outlier in all_outliers[0]:
            width = 2
            while True:
                if np.isnan(np.nanmedian(data[outlier-width:outlier+width])):
                    width += 1
                else:
                    data_filled[outlier] = np.nanmedian(data[outlier-width:outlier+width])
                    break
        assert np.isfinite(np.sum(data_filled))

        return data_filled, all_outliers

    def hipass_single(arr):
        nbin = 100
        nlam = len(arr)
        bins = int(nlam / nbin)
        xb = np.arange(nbin)*bins + bins/2.0
        yb = np.zeros(nbin+2)
        xb = np.append(np.append(0,xb),nlam-1)
        for ib in range(nbin):
            imin = int(ib*bins)
            imax = int(imin + bins)
            yb[ib+1] = np.nanmean(arr[imin:imax])
            yb[0] = arr[0]
            yb[-1] = arr[-1]
        cs_bin = splrep(xb,yb,s=0.0)
        fit = splev(np.arange(nlam),cs_bin,der=0)
        arr -= fit
        return arr
    
    
    spec_oot = kwargs["spec_oot"] # out of transit observation
    #saturated_telluric_inds = kwargs["saturated_telluric_inds"]
    
    detrended_spec = spec / spec_oot
    
    # when perfectly removing tellurics you must use an aggressive 
    # outlier masking technique
    
    detrended_spec = mask_outliers_phase_dependent(detrended_spec)
    
    detrended_spec = 1 - detrended_spec
    
    return detrended_spec

def perfect_simple(spec, **kwargs):
    def sigma_clip(arr, sig=3.):
        arr_copy = np.copy(arr)
        arr_med = np.nanmedian(arr)
        arr_std = np.nanstd(arr)
        
        loc = np.where(arr > arr_std*sig+arr_med)
        arr_copy[loc] = np.nan
        loc = np.where(arr <  -1*arr_std*sig+arr_med)
        arr_copy[loc] = np.nan
        return arr_copy
        
    spec_oot = kwargs["spec_oot"] # out of transit observation
    tellurics = kwargs["tellurics"]
    saturation_metric = kwargs["saturation_metric"]
    
    saturated_telluric_inds = (tellurics < saturation_metric)
    
    detrended_spec = spec / spec_oot
    
    detrended_spec[saturated_telluric_inds[0]] = np.nan
    
    detrended_spec = 1 - detrended_spec
    
    detrended_spec = sigma_clip(detrended_spec)
    
    return detrended_spec
    

# ADD SPORK IN

# ADD SYSREM IN
