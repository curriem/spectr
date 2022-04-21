# orbital_properties.py
import numpy as np
import astropy.units as unit
import astropy.constants as const

def calc_Kp(a_p, P, m_s, m_p, ecc=0, inclination=90):

    if inclination.unit.is_equivalent(unit.radian):
        pass
    elif inclination.unit.is_equivalent(unit.deg):
        inclination = (inclination).to(unit.rad)
    else:
        print("WARNING: Inclination has no units")
        inclination = (inclination*unit.deg).to(unit.rad)

    K_p = 2 * np.pi * a_p * np.sin(inclination) / (P * np.sqrt(1 - ecc**2)) # semi amplitude of planet
    K_p = K_p.to(unit.km / unit.s)

    return K_p

def calc_Ks(a_p, P, m_s, m_p, ecc=0, inclination=90):

    if inclination.unit.is_equivalent(unit.radian):
        pass
    elif inclination.unit.is_equivalent(unit.deg):
        inclination = (inclination).to(unit.rad)
    else:
        print("WARNING: Inclination has no units")
        inclination = (inclination*unit.deg).to(unit.rad)

    K_s = ((2*np.pi*const.G) / P)**(1/3) * m_p * np.sin(inclination) / m_s**(2/3) / np.sqrt(1-ecc**2) # semi-amplitude of star
    K_s = K_s.to(unit.km /unit.s)

    return K_s


def calc_RVp(Kp, phase, ecc=0, w_arg_peri=90):
    w_arg_peri = (w_arg_peri*unit.deg).to(unit.rad)
    if phase.unit.is_equivalent(unit.radian):
        pass
    elif phase.unit.is_equivalent(unit.deg):
        phase = phase.to(unit.radian)
    else:
        assert False, "specify phase units"
    RV_p = Kp * (np.cos(phase + w_arg_peri) + ecc*np.cos(w_arg_peri))
    return RV_p

def calc_RVs(Ks, phase, ecc=0, w_arg_peri=90):
    w_arg_peri = (w_arg_peri*unit.deg).to(unit.rad)
    if phase.unit.is_equivalent(unit.radian):
        pass
    elif phase.unit.is_equivalent(unit.deg):
        phase = phase.to(unit.radian)
    else:
        assert False, "specify phase units"
    RV_s = - Ks * (np.cos(phase + w_arg_peri) + ecc*np.cos(w_arg_peri))

    return RV_s



# following from Rodler 2010
def calc_v_proj_star(incl, R_star, P_rot_star):
    """
    ensure incl in radians
    """
    return 2 * np.pi * np.sin(incl) * (R_star / P_rot_star)

def calc_v_star_planframe(R_star, P_rot_star, P_orb):
    """
    P_orb is orbital period of planet
    """
    return 2*np.pi * R_star * (1/P_rot_star - 1/P_orb)

def calc_v_proj_plan(incl, R_plan, P_rot_plan):
    return 2 * np.pi * np.sin(incl) * (R_plan/P_rot_plan)

def calc_v_refl_plan(v_star_planframe, v_proj_plan):
    """two contributions to the broadening of the reflected light spectrum"""
    return np.sqrt(v_star_planframe**2 + v_proj_plan**2)
