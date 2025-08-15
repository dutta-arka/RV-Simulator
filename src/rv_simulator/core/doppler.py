"""
Doppler shift calculations for radial velocity simulations
"""

from ..utils.constants import SPEED_OF_LIGHT

def apply_rv_shift(wavelengths, rv_m_s):
    """
    Applies a radial velocity shift to wavelengths.
    
    Parameters
    ----------
    wavelengths: array_like: Wavelength array in Angstroms
    rv_m_s: float: Radial velocity shift in m/s
        
    Returns
    -------
    array_like: Doppler-shifted wavelengths
    """
    return wavelengths * (1 + rv_m_s / SPEED_OF_LIGHT)
