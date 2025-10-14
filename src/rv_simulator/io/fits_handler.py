"""
FITS file handling for VIPER-compatible output
"""

import numpy as np
from astropy.io import fits
import warnings
from ..utils.constants import TLS_ORDER_WAVELENGTHS

def slice_orders(wave, flux, order_ranges=None):
    """Slice a 1D spectrum (wave, flux) into a list of orders.

    - If order_ranges is None, fallback to TLS_ORDER_WAVELENGTHS from utils.constants.
    Returns (wave_orders, flux_orders).
    """
    if order_ranges is None:
        if TLS_ORDER_WAVELENGTHS:
            order_ranges = TLS_ORDER_WAVELENGTHS
        else:
            raise ValueError("No order_ranges provided and TLS_ORDER_WAVELENGTHS is not defined.")

    wave = np.asarray(wave)
    flux = np.asarray(flux)
    wave_orders = []
    flux_orders = []

    for (start, end) in order_ranges:
        mask = (wave >= start) & (wave <= end)
        w_cut = wave[mask]
        f_cut = flux[mask]
        if len(w_cut) < 2:
            wave_orders.append(np.empty(0))
            flux_orders.append(np.empty(0))
        else:
            wave_orders.append(w_cut)
            flux_orders.append(f_cut)

    print(f"[io] Sliced into {len(wave_orders)} orders.")
    return wave_orders, flux_orders

# def write_default_fits(filename, wave_orders, flux_orders, date_obs):
#     """Writes the processed spectral orders to a FITS file."""
#     print(f"Writing FITS file: {filename}")
#     hdr = fits.Header()
#     hdr.set('DATE-OBS', date_obs)
#     hdr.set('RA', 86.819720/15.0, 'Right ascension (hours)')
#     hdr.set('DEC', -51.06714)
#     hdr.set('EXP', 30)
#     hdr.set('TEL GEOELEV', 2648.0)
#     hdr.set('TEL GEOLAT', -24.6268)
#     hdr.set('TEL GEOLON', -70.4045)
    
#     hdu0 = fits.PrimaryHDU(header=hdr)
#     hdulist = [hdu0]
    
#     for idx, (w, f) in enumerate(zip(wave_orders, flux_orders)):
#         if len(w) == 0 or len(f) == 0:
#             w = np.array([0.0])
#             f = np.array([1.0])
#         c1 = fits.Column(name='wave', array=w, format='E')
#         c2 = fits.Column(name='flux', array=f, format='E')
#         hdu = fits.BinTableHDU.from_columns([c1, c2])
#         hdu.name = f'ORDER_{idx}'
#         hdulist.append(hdu)
    
#     fits.HDUList(hdulist).writeto(filename, overwrite=True)

def write_default_fits(filename, wave_orders, flux_orders, date_obs):
    """Write a VIPER-style multi-extension FITS with per-order BinTableHDUs."""
    hdr = fits.Header()
    hdr.set('DATE-OBS', date_obs)
    # keep a few default header fields (callers can patch or expand)
    hdr.set('RA', 86.819720 / 15.0, 'Right ascension (hours)')
    hdr.set('DEC', -51.06714)
    hdr.set('EXP', 30)
    hdr.set('TEL GEOELEV', 2648.0)
    hdr.set('TEL GEOLAT', -24.6268)
    hdr.set('TEL GEOLON', -70.4045)

    hdu0 = fits.PrimaryHDU(header=hdr)
    hdulist = [hdu0]

    for idx, (w, f) in enumerate(zip(wave_orders, flux_orders)):
        if len(w) == 0 or len(f) == 0:
            # placeholder arrays so downstream code can still open the FITS
            w = np.array([0.0], dtype=float)
            f = np.array([1.0], dtype=float)

        c1 = fits.Column(name='wave', array=np.asarray(w, dtype='f4'), format='E')
        c2 = fits.Column(name='flux', array=np.asarray(f, dtype='f4'), format='E')
        try:
            hdu = fits.BinTableHDU.from_columns([c1, c2])
        except Exception as exc:
            warnings.warn(f"Failed to create BinTableHDU for order {idx}: {exc}. Writing minimal arrays.")
            hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='wave', array=np.asarray([0.0], dtype='f4'), format='E'),
                fits.Column(name='flux', array=np.asarray([1.0], dtype='f4'), format='E')
            ])
        hdu.name = f'ORDER_{idx}'
        hdulist.append(hdu)

    fits.HDUList(hdulist).writeto(filename, overwrite=True)
    print(f"[io] Wrote FITS to: {filename}")

def load_original_orders(filename):
    """Load wave/flux arrays from a VIPER-style FITS (iterates table HDUs)."""
    wave_orders, flux_orders = [], []
    with fits.open(filename, ignore_blank=True) as hdul:
        for hdu in hdul[1:]:
            if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
                cols = [n.lower() for n in hdu.columns.names]
                if 'wave' in cols and 'flux' in cols:
                    wave_orders.append(np.asarray(hdu.data[hdu.columns.names[cols.index('wave')]], dtype=float))
                    flux_orders.append(np.asarray(hdu.data[hdu.columns.names[cols.index('flux')]], dtype=float))
    return wave_orders, flux_orders


def auto_detect_orders_from_fits(filename, fallback_order_wavs=None):
    """
    Auto-detect order wavelength ranges from a provided FITS file.
    Returns a list of (min, max) pairs.
    """
    hdul = fits.open(filename, ignore_blank=True)
    order_ranges = []
    for idx in range(1, len(hdul)):
        hdu = hdul[idx]
        if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
            continue
        for col in hdu.columns.names:
            if 'wl' in col.lower() or 'wave' in col.lower():
                wave = np.asarray(hdu.data[col], dtype=float)
                order_ranges.append((float(wave.min()), float(wave.max())))
                break
    hdul.close()
    if not order_ranges:
        if fallback_order_wavs is not None:
            return fallback_order_wavs
        raise ValueError("No valid orders detected. Ensure FITS has table HDUs with 'wave'/'wl' columns.")
    print(f"[io] Auto-detected {len(order_ranges)} orders")
    return order_ranges
