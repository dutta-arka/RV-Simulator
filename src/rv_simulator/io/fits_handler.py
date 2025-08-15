"""
FITS file handling for VIPER-compatible output
"""

import numpy as np
from astropy.io import fits
from ..utils.constants import TLS_ORDER_WAVELENGTHS

def slice_orders(wave, flux):
    """Slices the spectrum into echelle orders based on predefined wavelength ranges."""
    wave_orders, flux_orders = [], []
    for i, (start, end) in enumerate(TLS_ORDER_WAVELENGTHS):
        mask = (wave >= start) & (wave <= end)
        wave_cut = wave[mask]
        flux_cut = flux[mask]
        if len(wave_cut) < 2:
            wave_orders.append(np.empty(0))
            flux_orders.append(np.empty(0))
        else:
            wave_orders.append(wave_cut)
            flux_orders.append(flux_cut)
    print(f"Sliced into {len(wave_orders)} orders.")
    return wave_orders, flux_orders

def write_default_fits(filename, wave_orders, flux_orders, date_obs):
    """Writes the processed spectral orders to a FITS file."""
    print(f"Writing FITS file: {filename}")
    hdr = fits.Header()
    hdr.set('DATE-OBS', date_obs)
    hdr.set('RA', 86.819720/15.0, 'Right ascension (hours)')
    hdr.set('DEC', -51.06714)
    hdr.set('EXP', 30)
    hdr.set('TEL GEOELEV', 2648.0)
    hdr.set('TEL GEOLAT', -24.6268)
    hdr.set('TEL GEOLON', -70.4045)
    
    hdu0 = fits.PrimaryHDU(header=hdr)
    hdulist = [hdu0]
    
    for idx, (w, f) in enumerate(zip(wave_orders, flux_orders)):
        if len(w) == 0 or len(f) == 0:
            w = np.array([0.0])
            f = np.array([1.0])
        c1 = fits.Column(name='wave', array=w, format='E')
        c2 = fits.Column(name='flux', array=f, format='E')
        hdu = fits.BinTableHDU.from_columns([c1, c2])
        hdu.name = f'ORDER_{idx}'
        hdulist.append(hdu)
    
    fits.HDUList(hdulist).writeto(filename, overwrite=True)
