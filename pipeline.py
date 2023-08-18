# pipeline.py

import smart
import numpy as np
from .manipulate_spectra import *
from .orbital_properties import *
from .photon_counts import *
from .retrieve_sky import *
from .cross_correlation import *
from scipy.interpolate import splrep, splev


class SimulateObservation:

    def __init__(self, star_name, era, molecule, band, obs_type, instrument_R=1e5, verbose=False, noise_scalar=1., add_clouds=False, coronagraph_contrast=1e-5):#, telescope_name, distance,
                 #Vsystem, Vbary):

        self.star_name = star_name
        self.era = era
        self.molecule = molecule
        self.band = band
        self.obs_type = obs_type
        self.instrument_R = instrument_R
        self.verbose = verbose
        self.noise_scalar = noise_scalar
        self.add_clouds = add_clouds
        self.coronagraph_contrast = coronagraph_contrast
        #self.telescope_name = telescope_name
        #self.distance = distance
        #self.Vsystem = Vsystem
        #self.Vbary = Vbary

    def get_wl_bounds(self):
        molecule_band_dict = {"o2"  : {"690"  : [[14285, 14598], [0.685, 0.7]],
                                  "760"  : [[12903, 13245], [0.755, 0.775]],
                                  "1270" : [[7782, 7968], [1.255, 1.285]]},
                         "ch4" : {"890"  : [[10989, 11428], [0.875, 0.91]],
                                  "1100" : [[8333, 9090], [1.1, 1.2]],
                                  "1300" : [[6451, 7692], [1.3, 1.55]],
                                  #"1600" : [[5000, 6250], [1.6, 2.0]]},
                                  "1600" : [[5555, 6250], [1.6, 1.8]]},
                         "co2" : {"1590" : [[6024, 6578], [1.52, 1.66]],
                                  "2000" : [[4807, 5154], [1.94, 2.08]]},
                         "h2o" : {"900"  : [[10101, 11235], [0.89, 0.99]],
                                  "940"  : [[10526, 10752], [0.93, 0.95]],
                                  "1100" : [[8333, 9090], [1.1, 1.2]],
                                  "1300" : [[6451, 7692], [1.3, 1.55]],
                                  "all" :  [[6250, 11764], [0.87, 1.57]]},
                         "co"  : {"1550" : [[6250, 6430], [1.555, 1.60]],
                                  "2300" : [[4290, 4312], [2.319, 2.331]]},
                         "o3"  : {"630"  : [[15772, 15948], [0.627, 0.634]],
                                  "650"  : [[15151, 15503], [0.645, 0.66]],
                                  "3200" : [[2941, 3125], [3.2, 3.4]]},
                         "c2h6": {"3330" : [[2971, 3003],[3.33, 3.365]]},
                         "ocs" : {"2400" : [[4032, 4132],[2.42, 2.48]],
                                  "3200" : [[3058, 3125],[3.2, 3.27]],
                                  "3400" : [[2873, 2941],[3.4, 3.48]]},
                         "so2" : {"2400" : [[3921, 4081],[2.45, 2.55]]},
                         "full" : {"full": [[1886, 33333], [0.3, 5.]]}
                         }
        self.wnmin, self.wnmax = molecule_band_dict[self.molecule][self.band][0]
        self.wlmin, self.wlmax = molecule_band_dict[self.molecule][self.band][1]



    def load_smart_spectra(self, path=None):
        if path is None:
            assert False, "Specify path to smart spectra"
        def edge_effects(spec, fl_type, lam_min, lam_max):
            lam_buffer = 0.001
            lam_min+=lam_buffer
            lam_max-=lam_buffer

            lam = spec.lam

            if fl_type == "rad":
                fstar = spec.sflux
                fplan = spec.pflux
            elif fl_type == "trnst":
                tdepth = spec.tdepth

            lam_inds = np.where((lam >= lam_min) & (lam <= lam_max))

            lam = lam[lam_inds]

            if fl_type == "rad":
                fstar = fstar[lam_inds]
                fplan = fplan[lam_inds]
                return lam, fstar, fplan

            elif fl_type == "trnst":
                tdepth = tdepth[lam_inds]
                return lam, tdepth

        # open smart files:

        smart_rad = smart.readsmart.Rad(path+'/{}_{}_{}_{}_{}_{}cm_toa.rad'.format(self.star_name, self.era, self.molecule, self.band, self.wnmin, self.wnmax))
        smart_rad_no_mol = smart.readsmart.Rad(path+'/{}_{}_{}_{}_no_{}_{}_{}cm_toa.rad'.format(self.star_name, self.era, self.molecule, self.band, self.molecule, self.wnmin, self.wnmax))

        # get rid of SMART edge effects
        lam, fstar, fplan = edge_effects(smart_rad, "rad", self.wlmin, self.wlmax)
        lam_no_mol, fstar_no_mol, fplan_no_mol = edge_effects(smart_rad_no_mol, "rad", self.wlmin, self.wlmax)

        assert np.array_equal(lam, lam_no_mol)

        sort_inds = np.argsort(lam)

        self.lam = lam[sort_inds]
        self.fstar = fstar[sort_inds]
        self.fplan = fplan[sort_inds]
        self.fplan_no_mol = fplan_no_mol[sort_inds]


        self.fplan_fstar_only_mol = self.fplan / self.fstar - self.fplan_no_mol/self.fstar
        


        self.model_A = self.fplan / self.fstar


        if self.obs_type == "tran":
            smart_tran = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.wnmin, self.wnmax))
            smart_tran_no_mol = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_no_{}_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.molecule, self.wnmin, self.wnmax))

            # get rid of SMART edge effects
            lam_trnst, tdepth = edge_effects(smart_tran, "trnst", self.wlmin, self.wlmax)
            lam_no_mol_trnst, tdepth_no_mol = edge_effects(smart_tran_no_mol, "trnst", self.wlmin, self.wlmax)

            assert np.array_equal(lam_trnst, lam_no_mol_trnst)
            
            sort_inds_trnst = np.argsort(lam_trnst)
            assert np.array_equal(self.lam, lam_trnst[sort_inds_trnst])
            
            self.tdepth = tdepth[sort_inds_trnst]
            self.tdepth_no_mol = tdepth_no_mol[sort_inds_trnst]
            self.tdepth_onlymol = self.tdepth - self.tdepth_no_mol
            
            if self.add_clouds:
                tdepth_clear = np.copy(self.tdepth)
                tdepth_no_mol_clear = np.copy(self.tdepth_no_mol)
    
                smart_trnst_strato = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_strato_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.wnmin, self.wnmax))
                lam_strato, tdepth_strato = edge_effects(smart_trnst_strato, "trnst", self.wlmin, self.wlmax)
    
                smart_trnst_no_mol_strato = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_strato_no_{}_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.molecule, self.wnmin, self.wnmax))
                lam_no_mol_strato, tdepth_no_mol_strato = edge_effects(smart_trnst_no_mol_strato, "trnst", self.wlmin, self.wlmax)
    
                smart_trnst_cirrus = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_cirrus_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.wnmin, self.wnmax))
                lam_cirrus, tdepth_cirrus = edge_effects(smart_trnst_cirrus, "trnst", self.wlmin, self.wlmax)
    
                smart_trnst_no_mol_cirrus = smart.readsmart.Trnst(path+'/{}_{}_{}_{}_cirrus_no_{}_{}_{}cm.trnst'.format(self.star_name, self.era, self.molecule, self.band, self.molecule, self.wnmin, self.wnmax))
                lam_no_mol_cirrus, tdepth_no_mol_cirrus = edge_effects(smart_trnst_no_mol_cirrus, "trnst", self.wlmin, self.wlmax)
                
                smart_trnst_strato = tdepth_strato[sort_inds_trnst]
                tdepth_no_mol_strato = tdepth_no_mol_strato[sort_inds_trnst]
                tdepth_cirrus = tdepth_cirrus[sort_inds_trnst]
                tdepth_no_mol_cirrus = tdepth_no_mol_cirrus[sort_inds_trnst]
                
                tdepth_with_clouds = 0.5*tdepth_clear + 0.25*tdepth_strato + 0.25*tdepth_cirrus
                tdepth_no_mol_with_clouds = 0.5*tdepth_no_mol_clear + 0.25*tdepth_no_mol_strato + 0.25*tdepth_no_mol_cirrus
            
                self.tdepth = tdepth_with_clouds
                self.tdepth_onlymol = self.tdepth - tdepth_no_mol_with_clouds

    def run(self, inclination, R_star, P_rot_star, P_orb, R_plan, P_rot_plan,
            a_plan, M_star, M_plan, RV_sys, RV_bary, texp, phases, dist, tele_diam):

        """
        Parameters
        ----------
        inclination : scalar
            inclination of system [deg]
        R_star : scalar
            radius of the star [solar radii]
        P_rot_star : scalar
            period of stellar rotation [day]
        P_orb : scalar
            period of planet rotation [s]
        R_plan : scalar
            radius of the planet [earth radii]
        P_rot_plan : scalar
            period of planet rotation [s]
        a_plan : float
            semi-major axis of planet [AU]
        M_star : float
            mass of the star [solar masses]
        M_plan : float
            mass of the planet [earth masses]
        RV_sys : float
            radial velocity of the system [km/s]
        RV_bary : float
            barycentric radial velocity [km/s]
        texp :  float
            exposure time per frame [s]
        phases : 1d arr
            phases of planet to be observed
        dist : float
            distance to system [pc]
        tele_diam : float
            diameter of telescope [m]
        """

# =============================================================================
#         # Define two photon paths:
#         # Path 1: star to observer
#         # Path 2: star to planet to observer
# =============================================================================
        fstar_path1 = np.copy(self.fstar)
        
        if self.obs_type == "refl":
            
            fstar_path2 = np.copy(self.fstar)
            fplan_path2 = np.copy(self.fplan)
            
        elif self.obs_type == "tran":
            
            tdepth_path2 = np.copy(self.tdepth)

        # assign units to system/orbital properties
        inclination = (inclination * unit.deg).to(unit.radian)
        R_star = (R_star * unit.solRad).to(unit.km)
        P_rot_star = (P_rot_star*unit.day).to(unit.s) # from https://arxiv.org/pdf/1608.07834.pdf
        P_orb = P_orb * unit.s
        R_plan = (1*(unit.earthRad)).to(unit.km)
        P_rot_plan = P_rot_plan* unit.s

        a_plan = a_plan *   unit.AU
        M_star = M_star *   unit.solMass
        M_plan = M_plan *   unit.earthMass
        RV_sys = RV_sys *   unit.km/unit.s
        RV_bary = RV_bary * unit.km / unit.s

        self.RV_sys = RV_sys.value
        self.RV_bary = RV_bary.value

        dist = dist * unit.pc

        self.dist = dist
        self.incl = inclination
        self.a_plan = a_plan

        q = 1. # quantum efficiency
        fpa = 1. # fraction of planet light observed
        T = 0.1 # total throughput of telescope/instrument/detector

        self.tele_diam = tele_diam

        norders = 1 # number of orders observed

        # calculate rotational velocities:
        v_proj_star = calc_v_proj_star(inclination, R_star, P_rot_star)
        v_star_planframe = calc_v_star_planframe(R_star, P_rot_star, P_orb)
        v_proj_plan = calc_v_proj_plan(inclination, R_plan, P_rot_plan)
        v_refl_plan = calc_v_refl_plan(v_star_planframe, v_proj_plan)

        # calculate K coefficients
        K_s = calc_Ks(a_plan, P_orb, M_star, M_plan, inclination=inclination)
        K_p = calc_Kp(a_plan, P_orb, M_star, M_plan, inclination=inclination)
        self.K_p = K_p

# =============================================================================
#         ############### Step 1a rotationally broaden spectra ##################
#         fstar_path1 = rotational_broadening(self.lam, fstar_path1, v_proj_star.value, 0.6)
# 
#         if self.obs_type == "refl":
#             fstar_path2 = rotational_broadening(self.lam, fstar_path2, v_proj_star.value, 0.6)
#             fplan_path2 = rotational_broadening(self.lam, fplan_path2, v_refl_plan.value, 0.6)
#         elif self.obs_type == "tran":
#             tdepth_path2 = rotational_broadening(self.lam, tdepth_path2, v_refl_plan.value, 0.6)
#         ########################################################################
# 
# =============================================================================

        ############### Step 1b doppler shift the spectra ##################

        fstar_path1_matrix = np.empty((norders, len(phases), len(self.fstar)))
        
        if self.obs_type == "refl":
            fstar_path2_matrix = np.empty((norders, len(phases), len(self.fstar)))
            fplan_path2_matrix = np.empty((norders, len(phases), len(self.fstar)))
        elif self.obs_type == "tran":
            tdepth_path2_matrix = np.empty((norders, len(phases), len(self.fstar)))
        total_plan_RV = np.empty(len(phases))
        star_RVs = np.empty(len(phases))
        for order in range(norders):
            for i, phase in enumerate(phases):

                # calculate radial velocities
                RV_s = calc_RVs(K_s, phase)
                star_RVs[i] = RV_s.value
                RV_p = calc_RVp(K_p, phase)

                if self.verbose:
                    print("#### Phase {} ####".format(phase))
                    print("star RV:", RV_s)
                    print("planet RV:", RV_p)
                    print("total plan RV shift:", RV_sys + RV_bary + RV_p)
                    print("total star RV shift:", RV_sys + RV_bary + RV_s)
                    print("########################\n")
                total_plan_RV[i] = (RV_sys + RV_bary + RV_p).value

                fstar_path1_i = np.copy(fstar_path1)
                fstar_path1_i = doppler_shift(self.lam, fstar_path1_i, (RV_sys + RV_bary + RV_s).value)
                fstar_path1_matrix[order, i] = fstar_path1_i
                
                if self.obs_type == "refl":
                    fstar_path2_i = np.copy(fstar_path2)
                    fplan_path2_i = np.copy(fplan_path2)

                    fstar_path2_i = doppler_shift(self.lam, fstar_path2_i, (RV_sys + RV_bary + RV_s).value)
                    fplan_path2_i = doppler_shift(self.lam, fplan_path2_i, (RV_sys + RV_bary + RV_p).value)

                    fstar_path2_matrix[order, i] = fstar_path2_i
                    fplan_path2_matrix[order, i] = fplan_path2_i
                
                elif self.obs_type == "tran":
                    tdepth_path2_i = np.copy(tdepth_path2)
                    tdepth_path2_i = doppler_shift(self.lam, tdepth_path2_i, (RV_sys + RV_bary + RV_p).value)
                    tdepth_path2_matrix[order, i] = tdepth_path2_i
                    
        self.total_plan_RV = total_plan_RV
        self.star_RVs = star_RVs
        ########################################################################

        ############### Step 1c add tellurics to the spectra ##################

        # set telluric resolution to be retrieved
        skycalc_R = 1e5

        # retrieve tellurics
        skycalc = SkyFlux()
        skycalc.wmin = self.wlmin * 1000 - 100
        skycalc.wmax = self.wlmax * 1000 + 100
        skycalc.wgrid_mode = "fixed_spectral_resolution"
        skycalc.wres = skycalc_R
        skycalc.airmass = 1.15
        skycalc.incl_moon = "N"
        # thermal
        skycalc.incl_therm = "Y"
        skycalc.therm_t1 = 285
        skycalc.therm_e1 = 0.14
        skycalc.therm_t2 = 90
        skycalc.therm_e2 = 0.14
        

        skycalc.run_skycalc('../metadata/sky/skycalc_{}_{}.fits'.format(self.molecule, self.band))

        # interpolate onto spectrum wl grid
        f = interp1d(skycalc.lam, skycalc.trans, fill_value = "extrapolate")
        telluric_transmittance = f(self.lam)
        
        fstar_path1_matrix_no_T = np.copy(fstar_path1_matrix)

        if self.obs_type == "refl":
            fstar_path2_matrix_no_T = np.copy(fstar_path2_matrix)
            fplan_path2_matrix_no_T = np.copy(fplan_path2_matrix)

        # add tellurics to spectra
        fstar_path1_matrix *= telluric_transmittance
        if self.obs_type == "refl":
            fstar_path2_matrix *= telluric_transmittance
            fplan_path2_matrix *= telluric_transmittance

        ########################################################################

        ############### Step 1d instrumental broadening ##################
        # instrumental broadening
        for order in range(norders):
            for i in range(len(phases)):
                fstar_path1_matrix[order, i] = instrumental_broadening(fstar_path1_matrix[order, i], self.lam, self.instrument_R)
                fstar_path1_matrix_no_T[order, i] = instrumental_broadening(fstar_path1_matrix_no_T[order, i], self.lam, self.instrument_R)
                if self.obs_type == "refl":
                    fstar_path2_matrix[order, i] = instrumental_broadening(fstar_path2_matrix[order, i], self.lam, self.instrument_R)
                    fplan_path2_matrix[order, i] = instrumental_broadening(fplan_path2_matrix[order, i], self.lam, self.instrument_R)
                    
                    fstar_path2_matrix_no_T[order, i] = instrumental_broadening(fstar_path2_matrix_no_T[order, i], self.lam, self.instrument_R)
                    fplan_path2_matrix_no_T[order, i] = instrumental_broadening(fplan_path2_matrix_no_T[order, i], self.lam, self.instrument_R)
                
                elif self.obs_type == "tran":
                    tdepth_path2_matrix[order, i] = instrumental_broadening(tdepth_path2_matrix[order, i], self.lam, self.instrument_R)


                

        ############### Step 1e interpolate onto instrument wl grid ##################


        # construct instrument wl grid and interpolate onto it
        instrument_lam, instrument_dlam = construct_instrument_lam(self.wlmin, self.wlmax, self.instrument_R)


        fstar_path1_instrument_matrix = np.empty((norders, len(phases), len(instrument_lam)))
        fstar_path1_instrument_matrix_no_T = np.empty((norders, len(phases), len(instrument_lam)))

        if self.obs_type == "refl":
            fstar_path2_instrument_matrix = np.empty((norders, len(phases), len(instrument_lam)))
            fplan_path2_instrument_matrix = np.empty((norders, len(phases), len(instrument_lam)))
            
            fstar_path2_instrument_matrix_no_T = np.empty((norders, len(phases), len(instrument_lam)))
            fplan_path2_instrument_matrix_no_T = np.empty((norders, len(phases), len(instrument_lam)))
            
        elif self.obs_type == "tran":
            tdepth_path2_instrument_matrix = np.empty((norders, len(phases), len(instrument_lam)))
            

        for order in range(norders):
            for i in range(len(phases)):
                fstar_path1_instrument_matrix[order, i] = bin_to_instrument_lam(fstar_path1_matrix[order, i], self.lam, instrument_lam, instrument_dlam)
                fstar_path1_instrument_matrix_no_T[order, i] = bin_to_instrument_lam(fstar_path1_matrix_no_T[order, i], self.lam, instrument_lam, instrument_dlam)

                if self.obs_type == "refl":
                    fstar_path2_instrument_matrix[order, i] = bin_to_instrument_lam(fstar_path1_matrix[order, i], self.lam, instrument_lam, instrument_dlam)
                    fplan_path2_instrument_matrix[order, i] = bin_to_instrument_lam(fplan_path2_matrix[order, i], self.lam, instrument_lam, instrument_dlam)
                    
                    fstar_path2_instrument_matrix_no_T[order, i] = bin_to_instrument_lam(fstar_path1_matrix_no_T[order, i], self.lam, instrument_lam, instrument_dlam)
                    fplan_path2_instrument_matrix_no_T[order, i] = bin_to_instrument_lam(fplan_path2_matrix_no_T[order, i], self.lam, instrument_lam, instrument_dlam)

                
                elif self.obs_type == "tran":
                    tdepth_path2_instrument_matrix[order, i] = bin_to_instrument_lam(tdepth_path2_matrix[order, i],  self.lam, instrument_lam, instrument_dlam)
                
        ########################################################################

        ############### Step 1f calculate fluxes ################

        Fs_observer_matrix = np.empty_like(fstar_path1_instrument_matrix)
        Fs_observer_matrix_no_T = np.empty_like(fstar_path1_instrument_matrix)
        if self.obs_type == "refl":
            Fp_observer_matrix = np.empty_like(fstar_path1_instrument_matrix)

            Fp_observer_matrix_no_T = np.empty_like(fstar_path1_instrument_matrix)

        for order in range(norders):
            for i in range(len(phases)):
                Fs_observer_matrix[order, i] = Fstar(fstar_path1_instrument_matrix[order, i], R_star, a_plan, dist)
                Fs_observer_matrix_no_T[order, i] = Fstar(fstar_path1_instrument_matrix_no_T[order, i], R_star, a_plan, dist)
                if self.obs_type == "refl":
                    Fp_observer_matrix[order, i] = Fplan(fplan_path2_instrument_matrix[order, i], R_plan, dist)

                    Fp_observer_matrix_no_T[order, i] = Fplan(fplan_path2_instrument_matrix_no_T[order, i], R_plan, dist)
        ########################################################################

        ############### Step 1g planet/star photon counts ################
        if self.obs_type == "refl":
            self.Fp_observer_matrix = Fp_observer_matrix
            
        self.Fs_observer_matrix = Fs_observer_matrix


        cs_matrix = np.empty_like(Fs_observer_matrix)
        cs_matrix_no_T = np.empty_like(Fs_observer_matrix)

        if self.obs_type == "refl":
            cp_matrix = np.empty_like(Fp_observer_matrix)

            cp_matrix_no_T = np.empty_like(Fp_observer_matrix_no_T)

        for order in range(norders):
            for i in range(len(phases)):
                cs_matrix[order, i] = cstar(q, fpa, T, instrument_lam, instrument_dlam, Fs_observer_matrix[order, i], self.tele_diam)
                cs_matrix_no_T[order, i] = cstar(q, fpa, T, instrument_lam, instrument_dlam, Fs_observer_matrix_no_T[order, i], self.tele_diam)
                if self.obs_type == "refl":
                    cp_matrix[order, i] = cplan(q, fpa, T, instrument_lam, instrument_dlam, Fp_observer_matrix[order, i], self.tele_diam)
                    cp_matrix_no_T[order, i] = cplan(q, fpa, T, instrument_lam, instrument_dlam, Fp_observer_matrix_no_T[order, i], self.tele_diam)
        ########################################################################

        ############### Step 1h background noise photon counts ################
        pix_per_res_element = 100
        X = 2 # size of aperture in lam/D
        
        
        
        # sky background--- includes zodi, starlight, moonlight, atmospheric emission, airglow, thermal
        sky_background = skycalc.flux_raw_excl_moon # units are photons/s/m2/μm/arcsec2
        sky_lam = skycalc.lam
        omega_sky = np.pi*(X*sky_lam*1e-6/self.tele_diam*180.*3600./np.pi)**2. # arcsec2
        
        sky_background *= omega_sky # units are photons/s/m2/um
        
        sky_background = sky_background * np.pi * (self.tele_diam/2)**2 * T # units are photons/s/um 

        # instrumental broadening
        sky_background = instrumental_broadening(sky_background, sky_lam, self.instrument_R)

        # interpolate onto instrument wl grid
        sky_background = bin_to_instrument_lam(sky_background, sky_lam, instrument_lam, instrument_dlam)

       
        
        # run below if sky flux is W/m2/um
        #csky = ctherm_earth(q, X, T, instrument_lam, instrument_dlam, self.tele_diam, sky_background, CIRC=False)
        
        csky = sky_background
        
        ### THERMAL IS INCLUDED IN SKYCALC
        # # telescope thermal
        # mirror_temp = 285 # K
        # emissivity = 0.14
        # ctele = hrt.photon_counts.ctherm(q, X, T, instrument_lam, instrument_dlam, self.tele_diam, mirror_temp, emissivity)

        # dark current
# =============================================================================
#         dark_photons_per_pix_per_s = 0.00111111 # from ELT ETC
# =============================================================================
        cdark = np.zeros_like(instrument_lam)

        vis_dark_photons_per_pix_per_s = 0.0002
        nir_dark_photons_per_pix_per_s = 0.015

        vis_inds = (instrument_lam < 1.)
        nir_inds = (instrument_lam >= 1.)

        cdark[vis_inds] = cdark[vis_inds] + pix_per_res_element*vis_dark_photons_per_pix_per_s
        cdark[nir_inds] = cdark[nir_inds] + pix_per_res_element*nir_dark_photons_per_pix_per_s

        # read noise
        read_noise_per_pix = 3
        #read_noise_per_exposure = read_noise_per_pix * pix_per_res_element * np.ones_like(instrument_lam)
        
        read_noise = np.sqrt(pix_per_res_element) * read_noise_per_pix * np.ones_like(instrument_lam)

        
        
        # total noise per exposure
        self.sky_noise = np.sqrt(csky*texp)
        self.dark_noise = np.sqrt(cdark*texp)
        self.read_noise = read_noise
        #background_per_exposure = csky*texp + cdark*texp + read_noise_per_exposure
        ########################################################################

        if self.obs_type == "refl":
            ############### Step 1i mask star light with coronagraph ################
    
            
    
            cspeckle_matrix = np.copy(cs_matrix)
            cspeckle_matrix_no_T = np.copy(cs_matrix_no_T)
            for order in range(norders):
                for i in range(len(phases)):
                    cspeckle_matrix[order, i,] *= self.coronagraph_contrast
                    cspeckle_matrix_no_T[order, i,] *= self.coronagraph_contrast



# =============================================================================
#         if self.obs_type == "refl":
#             ################### interpolate model A onto instrument wl grid ########
#     
#             # instrumental broadening
#             self.model_A = instrumental_broadening(self.model_A, self.lam, self.instrument_R)
#     
#             # interpolate onto instrument wl grid
#             self.model_A = bin_to_instrument_lam(self.model_A, self.lam, instrument_lam, instrument_dlam)
#             
# =============================================================================
        
        



        ############### Step 1j construct simulated dataset ################

        naninds = ~np.isnan(cs_matrix[0, 0, :])
        self.data_naninds = naninds
        
        simulated_data = np.empty_like(cs_matrix[:, :, naninds])
        simulated_data_no_noise = np.empty_like(cs_matrix[:, :, naninds])
        simulated_data_no_tellurics = np.empty_like(cs_matrix[:, :, naninds])
        if self.obs_type == "tran":
            simulated_data_oot = np.empty_like(cs_matrix[:, :, naninds])
            simulated_data_oot_no_noise = np.empty_like(cs_matrix[:, :, naninds])


        signal_matrix = np.empty_like(cs_matrix)
        signal_matrix_no_T = np.empty_like(cs_matrix)
        noise_matrix = np.empty_like(cs_matrix)
        
        if self.obs_type == "tran":
            noise_oot_matrix = np.empty_like(cs_matrix)
            signal_oot_matrix = np.empty_like(cs_matrix)
            signal_oot_matrix_no_T = np.empty_like(cs_matrix)
        elif self.obs_type == "refl":
            signal_only_star_matrix = np.empty_like(cs_matrix)
            signal_only_star_matrix_no_T = np.empty_like(cs_matrix)
            signal_only_speckle_matrix = np.empty_like(cs_matrix)
            signal_only_speckle_matrix_no_T = np.empty_like(cs_matrix)

        # noise_matrix_no_T = np.empty_like(cspeckle_matrix)


        # SNR_no_T_matrix = np.empty_like(cspeckle_matrix)
        for order in range(norders):
            rand_nums = np.random.randn(len(phases), len(instrument_lam))            

            if self.obs_type == "refl":
                signal = cp_matrix[order]*texp
                signal_no_T = cp_matrix_no_T[order]*texp
                signal_only_star = cs_matrix[order] * texp
                signal_only_star_no_T = cs_matrix_no_T[order] * texp
                signal_only_star_matrix[order,] = signal_only_star
                signal_only_star_matrix_no_T[order,] = signal_only_star_no_T
                
                signal_only_speckle = cspeckle_matrix[order] * texp
                signal_only_speckle_no_T = cspeckle_matrix_no_T[order] * texp
                signal_only_speckle_matrix[order,] = signal_only_speckle
                signal_only_speckle_matrix_no_T[order,] = signal_only_speckle_no_T

                
                signal_matrix_no_T[order,] = signal_no_T
                
            
            elif self.obs_type == "tran":
                
                signal = cs_matrix[order]*texp * (1 - tdepth_path2_instrument_matrix[order])
                signal_no_T = cs_matrix_no_T[order]*texp * (1 - tdepth_path2_instrument_matrix[order])
                signal_oot = cs_matrix[order]*texp
                signal_oot_no_T = cs_matrix_no_T[order]*texp
                
                signal_oot_matrix[order,] = signal_oot
                signal_oot_matrix_no_T[order,] = signal_oot_no_T
                
                signal_matrix_no_T[order,] = signal_no_T

                
                
                
            signal_matrix[order,] = signal

                
            

        
        
        self.signal_matrix = signal_matrix[:, :, naninds]


        if self.obs_type == "tran":
            self.tdepth_path2_instrument_matrix = tdepth_path2_instrument_matrix
            self.signal_oot_matrix = signal_oot_matrix[:, :, naninds]
            self.signal_oot_matrix_no_T = signal_oot_matrix_no_T[:, :, naninds]
            self.signal_matrix_no_T = signal_matrix_no_T[:, :, naninds]

            
        if self.obs_type == "refl":
            self.signal_matrix_no_T = signal_matrix_no_T[:, :, naninds]
            self.signal_only_star = signal_only_star_matrix[:, :, naninds]
            self.signal_only_star_no_T = signal_only_star_matrix_no_T[:, :, naninds]

            self.signal_only_speckle = signal_only_speckle_matrix[:, :, naninds]
            self.signal_only_speckle_no_T = signal_only_speckle_matrix_no_T[:, :, naninds]

            
            
        tellurics_instrument = instrumental_broadening(telluric_transmittance, self.lam, self.instrument_R)
        tellurics_instrument =  bin_to_instrument_lam(telluric_transmittance, self.lam, instrument_lam, instrument_dlam)
        
        self.tellurics = tellurics_instrument



        self.instrument_lam = np.expand_dims(instrument_lam[naninds], axis=0)
        self.instrument_dlam = np.expand_dims(instrument_dlam[naninds], axis=0)
        


    def new_observation(self):
        new_rand_nums = np.random.randn(self.signal_matrix.shape[0], self.signal_matrix.shape[1], self.signal_matrix.shape[2])
        new_noise = new_rand_nums * np.sqrt(self.noise_matrix)
        new_data = self.signal_matrix + new_noise
        
        return new_data
        



class RemoveTellurics:

    def __init__(self, telluric_removal_func, data, **kwargs):

        self.telluric_removal_func = telluric_removal_func
        self.data = data
        self.kwargs = kwargs

    def run(self):

        data_tellurics_removed = self.telluric_removal_func(self.data, **self.kwargs)

        self.data_tellurics_removed = data_tellurics_removed


class InjectModel:
    def __init__(self, wl_data, spec_data, wl_model, spec_model, kp, ph, rvtot,
                 instrument_R, scale):

        self.wl_data = wl_data
        self.spec_data = spec_data
        self.wl_model = wl_model
        self.spec_model = spec_model
        self.kp = kp
        self.ph = ph
        self.rvtot = rvtot
        self.instrument_R = instrument_R
        self.scale = scale

    def run(self):
        # inject model into data
        data_injected = inject_model(self.wl_data, self.spec_data, self.wl_model, self.spec_model, self.rvtot, self.kp, self.ph, self.scale, self.instrument_R)

        self.data_injected = data_injected



class CrossCorrelation:

    def __init__(self, wl_data, spec_data, wl_model, spec_model, kp, ph, rvtot, ncc):

        self.wl_data = wl_data
        self.spec_data = spec_data
        self.wl_model = wl_model
        self.spec_model = spec_model
        self.kp = kp
        self.ph = ph
        self.rvtot = rvtot
        self.ncc = ncc


    def run(self):

        
        def run_hipass_data(arr):
            nbin = 100
            norders, nph, nlam = arr.shape
        
        
            bins = int(nlam / nbin)
            xb = np.arange(nbin)*bins + bins/2.0
            yb = np.zeros(nbin+2)
            xb = np.append(np.append(0,xb),nlam-1)
            for io in range(norders):
                for j in range(nph):
                    for ib in range(nbin):
                        imin = int(ib*bins)
                        imax = int(imin + bins)
                        yb[ib+1] = np.nanmean(arr[io,j,imin:imax])
                        yb[0] = arr[io, j, 0]
                        yb[-1] = arr[io, j, -1]
                    cs_bin = splrep(xb,yb,s=0.0)
                    fit = splev(np.arange(nlam),cs_bin,der=0)
                    arr[io,j,] -= fit
                    
            return arr

        def run_hipass_model(arr):
            nbin = 100
            nlam = len(arr)
            bins = int(nlam / nbin)
            xb = np.arange(nbin)*bins + bins/2.0
            yb = np.zeros(nbin+2)
            xb = np.append(np.append(0,xb),nlam-1)
            for ib in range(nbin):
                imin = int(ib*bins)
                imax = int(imin + bins)
                yb[ib+1] = np.mean(arr[imin:imax])
                yb[0] = arr[0]
                yb[-1] = arr[-1]
            cs_bin = splrep(xb,yb,s=0.0)
            fit = splev(np.arange(nlam),cs_bin,der=0)
            arr -= fit
            return arr
        
        
        hipass_data = run_hipass_data(self.spec_data[:, :, 10:-10])
        hipass_model = run_hipass_model(self.spec_model)
        
        # get rid of nans
        hipass_data[np.isnan(hipass_data)] = 0
        
        # ccf on real data
        ccf, rv_grid = cc_at_vrest(self.wl_data[:, 10:-10], hipass_data, self.wl_model, # had to include the wl buffer because funky interpolation things happen on the edges when you inject the model
                                   hipass_model, self.kp, self.ph,
                                   self.rvtot, self.ncc)
                                  # hipass=False)

        self.ccf = ccf
        self.rv_grid = rv_grid
        self.integrated_ccf = np.sum(np.sum(self.ccf, axis=0), axis=0)


    def calc_significance(self):

        significance_simple = calc_significance_simple(self.integrated_ccf, self.rv_grid)
        significance_chi2 = calc_significance_chi2(self.integrated_ccf, self.rv_grid)
        significance_ttest = calc_significance_ttest(self.integrated_ccf, self.rv_grid)

        self.significance_simple = significance_simple
        self.significance_chi2 = significance_chi2
        self.significance_ttest = significance_ttest

    def compare_to_injected(self, injected_ccf, trail_bound=25):

        ccf_real = self.integrated_ccf
        ccf_model = injected_ccf - ccf_real
        self.ccf_model = ccf_model

        iout = np.abs(self.rv_grid) >= trail_bound         # boolean
        nout = iout.sum()               # how many 'trues'
        iin = np.abs(self.rv_grid) < trail_bound           # boolean
        nin = iin.sum()                 # how many 'falses'

        # Scale the noiseless model CCF to the real CCF - Need to impose slope > 0 (correlation)
        # so the absolute value of the slope is taken. This means that an anti-correlation
        # (negative slope) will increase the chi-square of the residuals rather than decrease
        # it, resulting into a delta(sigma) value < 0.
        coef = np.polyfit(ccf_model, ccf_real, 1)
        #        if coef[0] < 0: coef[0] = 0
        coef[0] = np.abs(coef[0])
        ccf_fit = coef[0]*ccf_model + coef[1]

        self.ccf_fit = ccf_fit

        # Computing the residual cross correlation function
        ccf_res = ccf_real - ccf_fit

        self.ccf_res = ccf_res

        # Doing statistical tests on residual cross correlation function.
        # Note that dof decreases by 2 because we are fitting an apmplitude and
        # offset of the model CCF.
        var_model = np.std(ccf_res[iout])**2   # CCF variance (away from peak)
        chi2_model = (ccf_res[iin]**2).sum() / var_model
        sigma_model = stats.norm.isf(stats.chi2.sf(chi2_model,nin-3)/2)
        self.sigma_model = sigma_model
        d_sig = self.significance_chi2 - sigma_model

        self.d_sig = d_sig
        return d_sig

class CompareCCFs:

    def __init__(self, ccf_real, ccf_inj, rv_grid, trail_bound=25):
        self.ccf_real = ccf_real
        self.ccf_inj = ccf_inj
        self.rv_grid = rv_grid
        self.trail_bound = trail_bound

    def compare(self):
        self.ccf_model = self.ccf_inj - self.ccf_real

        iout = np.abs(self.rv_grid) >= self.trail_bound         # boolean
        nout = iout.sum()               # how many 'trues'
        iin = np.abs(self.rv_grid) < self.trail_bound           # boolean
        nin = iin.sum()                 # how many 'falses'

        # Scale the noiseless model CCF to the real CCF - Need to impose slope > 0 (correlation)
        # so the absolute value of the slope is taken. This means that an anti-correlation
        # (negative slope) will increase the chi-square of the residuals rather than decrease
        # it, resulting into a delta(sigma) value < 0.
        print(self.ccf_model)
        assert np.isfinite(np.sum(self.ccf_model)), "THERE IS A NAN IN CCF_MODEL"
        assert np.isfinite(np.sum(self.ccf_real)), "THERE IS A NAN IN CCF_REAL"

        coef = np.polyfit(self.ccf_model, self.ccf_real, 1)
        #        if coef[0] < 0: coef[0] = 0
        coef[0] = np.abs(coef[0])
        self.ccf_fit = coef[0]*self.ccf_model + coef[1]

        # Computing the residual cross correlation function
        self.ccf_res = self.ccf_real - self.ccf_fit

        # Doing statistical tests on residual cross correlation function.
        # Note that dof decreases by 2 because we are fitting an apmplitude and
        # offset of the model CCF.
        var_ccf_res = np.std(self.ccf_res[iout])**2   # CCF variance (away from peak)
        chi2_ccf_res = (self.ccf_res[iin]**2).sum() / var_ccf_res
        sigma_ccf_res = stats.norm.isf(stats.chi2.sf(chi2_ccf_res,nin-3)/2)
    
        ccf_real_significance_chi2 = calc_significance_chi2(self.ccf_real, self.rv_grid)

        d_sig = ccf_real_significance_chi2 - sigma_ccf_res

        self.dsig = d_sig

