"""Input/Output functionality"""

# from .spectrum_reader import read_spectrum_csv, read_FTS_fits
# from .fits_handler import write_default_fits, slice_orders

# __all__ = [
#     'read_spectrum_csv',
#     'read_FTS_fits', 
#     'write_default_fits',
#     'slice_orders'
# ]

from .spectrum_reader import read_spectrum_csv, read_FTS_fits, read_FTS_fits_auto
from .fits_handler import write_default_fits, slice_orders, load_original_orders, auto_detect_orders_from_fits

__all__ = [
    'read_spectrum_csv',
    'read_FTS_fits',
    'read_FTS_fits_auto',
    'write_default_fits',
    'slice_orders',
    'load_original_orders',
    'auto_detect_orders_from_fits',
]
