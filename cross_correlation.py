# cross_correlation.py
import numpy as np
from .manipulate_spectra import doppler_shift
from scipy.interpolate import splrep, splev
from scipy import stats
from .orbital_properties import calc_RVp
def cross_correlate(data, model, lam, v_shift_grid, hipass=True):
    N = len(model) # == len(data) == len(lam)
    s_f2 = 1/N * np.nansum(data * data) # data variance

    ccf = np.empty_like(v_shift_grid, dtype=np.float64)
    for i, v_shift in enumerate(v_shift_grid):
        model_shifted = doppler_shift(lam, model, v_shift)

        s_g2 = 1/N * np.nansum(model * model) # model variance

        R = 1/N * np.nansum(data * model_shifted) # cross covariance

        cc_coeff = R / np.sqrt(s_f2 * s_g2) # cross-correlation coefficient

        ccf[i] = cc_coeff

    if hipass:
        nbin = 10
        bins = int(len(v_shift_grid) / nbin)
        xb = np.arange(nbin)*bins + bins/2.0
        yb = np.zeros(nbin+2)

        xb = np.append(np.append(0,xb),len(v_shift_grid)-1)

        for ib in range(nbin):
            imin = int(ib*bins)
            imax = int(imin + bins)
            yb[ib+1] = np.mean(ccf[imin:imax])
        cs_bin = splrep(xb,yb,s=0.0)
        fit = splev(np.arange(len(v_shift_grid)),cs_bin,der=0)
        ccf -= fit
    return ccf

def cross_correlate_logL_Matteo(data, model, lam, v_shift_grid):
    N = len(model) # == len(data) == len(lam)
    s_f2 = 1/N * np.nansum(data * data) # data variance

    ccf = np.empty_like(v_shift_grid, dtype=np.float64)
    for i, v_shift in enumerate(v_shift_grid):
        model_shifted = doppler_shift(lam, model, v_shift)

        s_g2 = 1/N * np.nansum(model * model) # model variance

        R = 1/N * np.nansum(data * model_shifted) # cross covariance

        #logL = -N/2 * (np.log(s_f*s_g) + np.log(s_f/s_g + s_g/s_f - 2*R / (np.sqrt(s_f**2 * s_g**2))))
        logL = -N/2 * np.log(s_f2 + s_g2 - 2*R)
        ccf[i] = logL
    return ccf


def calc_significance_chi2(ccf, rv_grid, trail_bound=25):
    iout = np.abs(rv_grid) >= trail_bound         # boolean
    nout = iout.sum()               # how many 'trues'
    iin = np.abs(rv_grid) < trail_bound           # boolean
    nin = iin.sum()                 # how many 'falses'

    ccf_variance = np.nanstd(ccf[iout])**2    # CCF variance (away from peak)
    ccf_chi2 = (ccf[iin]**2).sum() / ccf_variance   # Chi-square of peak

    ccf_sigma = stats.norm.isf(stats.chi2.sf(ccf_chi2,nin-1)/2)   # Chi-square -> sigma
    return ccf_sigma

def calc_significance_simple(ccf, rv_grid, trail_bound=25):
    iout = np.abs(rv_grid) >= trail_bound         # boolean
    nout = iout.sum()               # how many 'trues'
    iin = np.abs(rv_grid) < trail_bound           # boolean
    nin = iin.sum()                 # how many 'falses'

    ccf_std = np.nanstd(ccf[iout], ddof=1)    # CCF variance (away from peak)
    
    ccf_signal_ind = np.argmin(np.abs(rv_grid))

    ccf_signal = ccf[ccf_signal_ind]
    

    ccf_sigma = ccf_signal / ccf_std
    return ccf_sigma

def calc_significance_ttest(ccf, rv_grid, trail_bound=25):
    iout = np.abs(rv_grid) >= trail_bound         # boolean
    nout = iout.sum()               # how many 'trues'
    iin = np.abs(rv_grid) < trail_bound           # boolean
    nin = iin.sum()                 # how many 'falses'


    mean_out = np.mean(ccf[iout])
    mean_in = np.mean(ccf[iin])

    std_out = ((ccf[iout] - mean_out)**2.).sum() / (nout - 1.) # kaitlin's std_out
    std_in  = ((ccf[iin] - mean_in  )**2.).sum() / (nin - 1.) # kaitlin's std_in

    # computing t statistic
    t = (mean_in - mean_out) / np.sqrt(std_out/nout + std_in/nin)
    nu = (std_out/nout + std_in/nin)**2. / (((std_out/nout)**2. / (nout-1)) # Kaitlin's nu
                                            + ((std_in/nin)**2. / (nin-1)))

    pvalue = stats.t.sf(t, nu)

    sigma = stats.norm.isf(pvalue)
    return sigma

def cc_vshift_grid(v_shift_range, v_shift_width):
    return np.arange(-v_shift_range, v_shift_range+v_shift_width, v_shift_width)



def mattcc(fVec, gVec):
    ''' Fast cross correlation function with full formula (mean subtraction and
    normalisation by variance). Takes in input the two vectors to cross correlate,
    the data (fVec) and the model (gVec). '''
    N, = fVec.shape
    Id = np.ones(N)
# =============================================================================
#     fVec -= (fVec @ Id) / N
#     gVec -= (gVec @ Id) / N
#     sf2 = (fVec @ fVec)
#     sg2 = (gVec @ gVec)
#     return (fVec @ gVec) / np.sqrt(sf2*sg2)
# =============================================================================

    fVec -= np.nansum(fVec * Id) / N
    gVec -= np.nansum(gVec * Id) / N
    sf2 = np.nansum(fVec * fVec)
    sg2 = np.nansum(gVec * gVec)
    return np.nansum(fVec * gVec) / np.sqrt(sf2*sg2)




def cc_at_vrest(wl_data, spec_data, wl_model, spec_model, kp, ph, rvtot, ncc, cc_lim=150, hipass=True):

    rv_grid = np.linspace(-1*cc_lim,cc_lim,ncc)
    norders, nph, nlam = spec_data.shape
    ccf = np.zeros((norders, nph, ncc))

    cs = splrep(wl_model, spec_model, s=0)

    for j in range(nph):
        #print(cs)
        RV_planet = calc_RVp(kp, ph[j], ecc=0, w_arg_peri=90)
        lagTemp =  rv_grid  + rvtot[j] + RV_planet # kp * np.sin(2*np.pi*ph[j])

        for cc_ind in range(ncc):

            wl_model_shift = wl_data * (1.0 - lagTemp[cc_ind]/2.998E5) # doppler shift
            spec_model_shift = splev(wl_model_shift, cs, der=0, ext=1) # the model interpolated onto data wl grid
            
            for order in range(norders):
                fVec = spec_data[order,j,].copy()
                gVec = spec_model_shift[order,].copy()

                ccf[order, j, cc_ind] = mattcc(fVec,gVec)
    # Removing gradients and trends in the CCF
    if hipass:
        nbin = 10
        bins = int(ncc / nbin)
        xb = np.arange(nbin)*bins + bins/2.0
        yb = np.zeros(nbin+2)
        xb = np.append(np.append(0,xb),ncc-1)
        for io in range(norders):
            for j in range(nph):
                for ib in range(nbin):
                    imin = int(ib*bins)
                    imax = int(imin + bins)
                    yb[ib+1] = np.nanmean(ccf[io,j,imin:imax])
                    yb[0] = ccf[io, j, 0]
                    yb[-1] = ccf[io, j, -1]
                cs_bin = splrep(xb,yb,s=0.0)
                fit = splev(np.arange(ncc),cs_bin,der=0)
                ccf[io,j,] -= fit
    return ccf, rv_grid

def inject_model(wl_data, spec_data, wl_model, spec_model, rvtot, kp, ph, scale, instrument_R):

    norders, nph, nlam = spec_data.shape

    dlam_lam_model = np.mean(2 * (wl_model[1:]-wl_model[0:-1]) / (wl_model[1:]+wl_model[0:-1]))
    dlam_lam_instrument = 1./instrument_R

    fwhm_pix = dlam_lam_instrument / dlam_lam_model

    # Conversion FWHM -> Gaussian sigma
    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Computing the convolution kernel
    xker = np.arange(61) - 30
    yker = np.exp(-0.5 * (xker/sigma_pix)**2)
    yker /= yker.sum() #Normalisation

    # Convolution of model with CRIRES Gaussian IP
    spec_model_conv = np.convolve(spec_model, yker, mode='same')

    cs = splrep(wl_model, spec_model_conv, s=0)

    spec_data_injected = np.copy(spec_data)
    for j in range(nph):
        RV_planet = calc_RVp(kp, ph[j], ecc=0, w_arg_peri=90)
        model_RV = RV_planet + rvtot[j]

        wl_model_shift = wl_data * (1.0 - model_RV/2.998E5) # doppler shift
        spec_model_shift = splev(wl_model_shift, cs, der=0) # the model interpolated onto data wl grid

        spec_model_shift *= scale # scale the model spectrum

        # injecting by multiplying the data by (1 + Fp/Fs)
        for order in range(norders):
            spec_data_injected[order, j, ] *= (1 + spec_model_shift[order])
    return spec_data_injected
