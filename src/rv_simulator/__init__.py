"""
RV-Simulator: A repository for simulating radial velocity observations
"""

__version__ = "0.1.0"
__author__ = "Arkaprova Dutta"

# Core imports for easy access
from .core.keplerian import simulate_planetary_system
from .core.convolution import convolve_IP
from .core.doppler import apply_rv_shift
from .io.spectrum_reader import read_spectrum_csv, read_FTS_fits
from .io.fits_handler import write_default_fits, slice_orders

__all__ = [
    'simulate_planetary_system',
    'convolve_IP', 
    'apply_rv_shift',
    'read_spectrum_csv',
    'read_FTS_fits',
    'write_default_fits',
    'slice_orders'
]
