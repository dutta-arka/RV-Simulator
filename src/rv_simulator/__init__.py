"""
RV-Simulator: A repository for simulating radial velocity observations
"""

__version__ = "0.1.0"
__author__ = "Arkaprova Dutta"

# Core imports for easy access
from .core.doppler import apply_rv_shift, simulate_stellar_rv
from .core.keplerian import calculate_keplerian_rv
from .core.convolution import convolve_IP_on_uniform_grid, create_instrument_grids
from .io.spectrum_reader import read_spectrum_csv, auto_detect_orders_from_fits
from .io.fits_handler import read_FTS_fits, write_default_fits
from .utils.constants import SPEED_OF_LIGHT, DEFAULT_ORDER_RANGES

__all__ = [
    '__version__',
    'apply_rv_shift',
    'simulate_stellar_rv',
    'calculate_keplerian_rv',
    'convolve_IP_on_uniform_grid',
    'create_instrument_grids',
    'read_spectrum_csv',
    'auto_detect_orders_from_fits',
    'read_FTS_fits',
    'write_default_fits',
    'SPEED_OF_LIGHT',
    'DEFAULT_ORDER_RANGES',
]
