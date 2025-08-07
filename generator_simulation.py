#! /usr/bin/env python3
"""
Simulate VIPER FITS Generator
-----------------------------

This script generates a series of synthetic stellar observations in the VIPER FITS format.
It simulates observations by Doppler-shifting a given synthetic spectrum, multiplying it
with an iodine FTS reference, and convolving the result with a user-defined Gaussian
instrumental profile (IP). The output is sliced into echelle orders and saved as
multi-extension FITS files ready for use with VIPER.

Usage:
------
python3 generator_exp.py -num_obs 3 -vel_list "[1050, 1000, 950]" \
  -time_step 5d0h -ip_width 1.2 -ip_type 'bigaussian' -asymmetry 0.2 \
  -file spectrum.csv -template

Arguments:
----------
-num_obs     Number of observations to generate.
-vel_list    RV shifts (m/s) for each observation as a list.
-time_step   Spacing between observations, e.g. '3d0h' for 3 days and 0 hours.
-file        Path to the synthetic spectrum CSV file.
-ip_width    IP width in pixels (Gaussian sigma).
-ip_type     Type of instrumental profile to convolve with: 'gaussian', 'bigaussian', or 'voigt'.
-asymmetry   Asymmetry factor (-1 to 1) for bi-Gaussian IP.
-gamma       Lorentzian width (gamma) for Voigt profile convolution.
-template    Optional flag to create a template FITS file.
-site        Optional flag: Observatory name recognised by astropy (e.g. 'Keck').

Outputs:
--------
- Simulated files: simulated_viper_data_1.fits, ..., simulated_viper_data_N.fits
- Optional: template_viper_data.fits (stellar template only)

Note:
-----
FTS iodine spectrum is fixed to 'FTS_default_TLS.fits'.
CSV must contain columns: 'wave' (or 'wavelength') and 'flux'.
"""
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.special import wofz
# =============================================================================
# from scipy.signal import convolve
# from scipy.signal.windows import gaussian
# =============================================================================
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
import argparse
import ast
import re
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('de430')

# --- Constants ---
SPEED_OF_LIGHT = 299792458.0  # Speed of light in m/s
OUTPUT_TEMPLATE = 'simulated_viper_data_{}.fits'
OUTPUT_TEMPLATE_FILE = 'template_viper_data.fits'
FTS_FILE = 'FTS_default_TLS.fits'

# Echelle order wavelength ranges (Angstrom) TLS
order_wavs = [
    (4513.00, 4593.39), (4549.25, 4630.49), (4586.10, 4668.20),
    (4623.56, 4706.52), (4661.65, 4745.47), (4700.39, 4785.07),
    (4739.79, 4825.34), (4779.87, 4866.28), (4820.64, 4907.92),
    (4862.13, 4950.28), (4904.35, 4993.37), (4947.32, 5037.21),
    (4991.06, 5081.83), (5035.58, 5127.24), (5080.92, 5173.47),
    (5127.09, 5220.54), (5174.12, 5268.47), (5222.02, 5317.29),
    (5270.82, 5367.01), (5320.55, 5417.68), (5371.23, 5469.30),
    (5422.89, 5521.92), (5475.55, 5575.55), (5529.25, 5630.24),
    (5584.02, 5686.01), (5639.89, 5742.89), (5696.88, 5800.91),
    (5755.04, 5860.12), (5814.40, 5920.55), (5875.00, 5982.24),
    (5936.88, 6045.21), (6000.07, 6109.53), (6064.62, 6175.23),
    (6130.57, 6242.35), (6197.97, 6310.94), (6266.87, 6381.06),
    (6337.32, 6452.74), (6409.38, 6526.05), (6483.09, 6601.04),
    (6558.51, 6677.77), (6635.72, 6756.30), (6714.76, 6836.69),
    (6795.72, 6919.01), (6878.65, 7003.33), (6963.64, 7089.73),
    (7050.77, 7178.27), (7140.11, 7269.05), (7231.75, 7362.14),
    (7325.79, 7457.65), (7422.32, 7555.65), (7521.45, 7656.25)
]

# --- Read CSV synthetic spectrum ---
def read_spectrum_csv(file):
    # Reads a synthetic spectrum from a CSV file.
    data = pd.read_csv(file)
    cols = [c.lower() for c in data.columns]
    colmap = dict(zip(cols, data.columns))
    try:
        wave = data[colmap.get('wave', colmap.get('wavelength'))].values
        flux = data[colmap['flux']].values
    except KeyError:
        raise KeyError("CSV must contain 'wave' or 'wavelength', and 'flux' columns. Found: " + str(data.columns.tolist()))
    print(f"Loaded synthetic spectrum with {len(wave)} points.")
    return wave, flux

# --- Read FTS iodine spectrum ---
def read_FTS_fits():
    # Reads the FTS iodine spectrum from a FITS file.
    file = FTS_FILE
    with fits.open(file) as hdul:
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'columns') and hdu.data is not None:
                names = hdu.columns.names
                lower_names = [n.lower() for n in names]
                if 'wave' in lower_names and 'flux' in lower_names:
                    wave_col = names[lower_names.index('wave')]
                    flux_col = names[lower_names.index('flux')]
                    wave = hdu.data[wave_col]
                    flux = hdu.data[flux_col]
                    print(f"Loaded FTS iodine spectrum with {len(wave)} points from HDU {i}.")
                    return wave, flux
        raise KeyError("FTS FITS file must contain 'wave' and 'flux' columns in at least one HDU.")

def _bigaussian_kernel(size: int, sigma: float, asymmetry: float) -> np.ndarray:
    """Create an asymmetric (bi-Gaussian) kernel."""
    x = np.arange(-size, size + 1)
    sigma_left = sigma * (1 + asymmetry)
    sigma_right = sigma * (1 - asymmetry)
    kernel = np.where(x < 0,
                      np.exp(-0.5 * (x / sigma_left)**2),
                      np.exp(-0.5 * (x / sigma_right)**2))
    return kernel / np.sum(kernel)

def _voigt_kernel(size: int, sigma: float, gamma: float) -> np.ndarray:
    """Generate a Voigt profile kernel."""
    x = np.linspace(-size, size, 2 * size + 1)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    kernel = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return kernel / np.sum(kernel)

# =============================================================================
# def convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width):
#     """
#     Convolve the product of a Doppler-shifted stellar spectrum and an iodine FTS reference
#     with a Gaussian instrumental profile (IP).
# 
#     Inputs:
#       shifted_wave : array_like  - wavelength grid of the shifted stellar spectrum (Å)
#       star_flux    : array_like  - stellar flux on shifted_wave
#       fts_wave     : array_like  - wavelength grid of the iodine FTS reference (Å)
#       fts_flux     : array_like  - FTS transmission function on fts_wave
#       width        : float       - Gaussian sigma (in pixels) for the IP
# 
#     Returns:
#       convolved_flux : ndarray - convolved spectrum sampled at shifted_wave
#     """
#     # determine overlap
#     wave_min = max(shifted_wave.min(), fts_wave.min())
#     wave_max = min(shifted_wave.max(), fts_wave.max())
#     if wave_max <= wave_min:
#         raise ValueError("no overlapping wavelength region")
# 
#     # finest wavelength step
#     dx_star = np.min(np.diff(shifted_wave))
#     dx_fts  = np.min(np.diff(fts_wave))
#     finest_dx = min(dx_star, dx_fts)
# 
#     # uniform grid
#     npts = max(2, int(np.floor((wave_max - wave_min) / finest_dx)) + 1)
#     wave_uni = np.linspace(wave_min, wave_max, npts)
# 
#     # resample onto uniform grid (fill outside with 1.0)
#     star_uni = np.interp(wave_uni, shifted_wave, star_flux, left=1.0, right=1.0)
#     fts_uni  = np.interp(wave_uni, fts_wave,    fts_flux, left=1.0, right=1.0)
# 
#     # multiply and convolve with Gaussian IP
#     model_uni = star_uni * fts_uni
#     model_conv = gaussian_filter1d(model_uni, sigma=width, mode='constant', cval=1.0)
# 
#     # interpolate back to original grid
#     return np.interp(shifted_wave, wave_uni, model_conv, left=1.0, right=1.0)
# =============================================================================

# =============================================================================
# def convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width):
#     """
#     Convolve the product of a Doppler-shifted stellar spectrum and an iodine FTS reference
#     with a Gaussian instrumental profile (IP), using scipy.signal.convolve.
# 
#     Parameters
#     ----------
#     shifted_wave : ndarray
#         Wavelength grid of the Doppler-shifted stellar spectrum [Angstrom].
#     star_flux : ndarray
#         Stellar flux at each wavelength in shifted_wave.
#     fts_wave : ndarray
#         Wavelength grid of iodine FTS reference spectrum [Angstrom].
#     fts_flux : ndarray
#         FTS transmission flux at each wavelength in fts_wave.
#     width : float
#         Gaussian sigma (in pixels) for the IP convolution.
# 
#     Returns
#     -------
#     convolved_flux : ndarray
#         Spectrum after convolution with instrumental profile, sampled on shifted_wave.
#     """
# 
#     # Overlap region
#     wave_min = max(np.min(shifted_wave), np.min(fts_wave))
#     wave_max = min(np.max(shifted_wave), np.max(fts_wave))
#     if wave_max <= wave_min:
#         raise ValueError("No overlapping wavelength region between star and FTS.")
# 
#     # Finer of the two spacings
#     dx = min(np.min(np.diff(shifted_wave)), np.min(np.diff(fts_wave)))
#     wave_uni = np.arange(wave_min, wave_max, dx)
# 
#     # Resample with edge padding
#     star_uni = np.interp(wave_uni, shifted_wave, star_flux, left=1.0, right=1.0)
#     fts_uni  = np.interp(wave_uni, fts_wave,    fts_flux,  left=1.0, right=1.0)
# 
#     model_uni = star_uni * fts_uni
# 
#     # Gaussian kernel setup
#     kernel_size = int(np.ceil(8 * width)) | 1
#     kernel = gaussian(kernel_size, std=width)
#     kernel /= np.sum(kernel)
# 
#     # Pad and convolve
#     pad = kernel_size // 2
#     padded_model = np.pad(model_uni, pad, mode='edge')
#     model_conv = convolve(padded_model, kernel, mode='valid')
# 
#     # Interpolate back to original shifted_wave
#     return np.interp(shifted_wave, wave_uni, model_conv, left=1.0, right=1.0)
# =============================================================================

def convolve_IP(shifted_wave: np.ndarray, star_flux: np.ndarray, fts_wave: np.ndarray, fts_flux: np.ndarray, width: float, ip_type: str = 'gaussian', asymmetry: float = 0.0, gamma: float = 0.0) -> np.ndarray:

    """
    Convolve stellar spectrum and FTS reference with selected IP profile.

    Parameters:
      shifted_wave : wavelength grid of shifted stellar spectrum [Å]
      star_flux    : stellar flux on shifted_wave
      fts_wave     : wavelength grid of iodine FTS reference [Å]
      fts_flux     : FTS transmission on fts_wave
      width        : Gaussian sigma (in pixels) for IP
      ip_type      : 'gaussian', 'bigaussian', or 'voigt'
      asymmetry    : asymmetry factor for bi-Gaussian (-1 to 1)
      gamma        : Lorentzian width for Voigt profile

    Returns:
      convolved_flux : interpolated convolved flux on shifted_wave grid
    """
    wave_min = max(shifted_wave.min(), fts_wave.min())
    wave_max = min(shifted_wave.max(), fts_wave.max())
    if wave_max <= wave_min:
        raise ValueError("No overlapping wavelength region")

    dx = min(np.min(np.diff(shifted_wave)), np.min(np.diff(fts_wave)))
    npts = max(2, int(np.floor((wave_max - wave_min) / dx)) + 1)
    wave_uni = np.linspace(wave_min, wave_max, npts)

    star_uni = np.interp(wave_uni, shifted_wave, star_flux, left=1.0, right=1.0)
    fts_uni  = np.interp(wave_uni, fts_wave,    fts_flux, left=1.0, right=1.0)
    model_uni = star_uni * fts_uni

    if ip_type == 'gaussian':
        model_conv = gaussian_filter1d(model_uni, sigma=width, mode='constant', cval=1.0)

    elif ip_type == 'bigaussian':
        kernel = _bigaussian_kernel(int(5 * width), width, asymmetry)
        model_conv = fftconvolve(model_uni, kernel, mode='same')

    elif ip_type == 'voigt':
        kernel = _voigt_kernel(int(5 * width), width, gamma)
        model_conv = fftconvolve(model_uni, kernel, mode='same')

    else:
        raise ValueError(f"Invalid ip_type: {ip_type}. Choose 'gaussian', 'bigaussian', or 'voigt'.")

    return np.interp(shifted_wave, wave_uni, model_conv, left=1.0, right=1.0)

# --- Doppler shift ---
def apply_rv_shift(wavelengths, rv_m_s):
    # Applies a radial velocity shift to wavelengths.
    return wavelengths * (1 + rv_m_s / SPEED_OF_LIGHT)

# --- Slice to orders ---
def slice_orders(wave, flux):
    # Slices the spectrum into echelle orders based on predefined wavelength ranges.
    wave_orders, flux_orders = [], []
    for i, (start, end) in enumerate(order_wavs):
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

# --- Write to DEFAULT format FITS ---
def write_default_fits(filename, wave_orders, flux_orders, date_obs):
    # Writes the processed spectral orders to a FITS file.
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

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Generate simulated VIPER FITS files with synthetic RV shifts and IP convolution.")
parser.add_argument('-num_obs', type=int, default=10, help="Number of observations to generate.")
parser.add_argument('-vel_list', type=str, default=None, help="List of RVs (m/s) for each observation.")
parser.add_argument('-time_step', type=str, default='5d0h', help="Time spacing between observations (e.g., '3d0h' for 3 days and 0 hours).")
parser.add_argument('-template', action='store_true', help="Flag to create a template file.")
parser.add_argument('-file', type=str, default='spectrum.csv', help="Input CSV file for synthetic spectrum.")
parser.add_argument('-ip_width', type=float, default=1.2, help="Gaussian IP width (sigma) for convolution.")
parser.add_argument('-ip_type', type=str, default='gaussian', choices=['gaussian', 'bigaussian', 'voigt'], help="Type of instrumental profile to convolve with: 'gaussian', 'bigaussian', or 'voigt'.")
parser.add_argument('-asymmetry', type=float, default=0.0, help="Asymmetry factor (-1 to 1) for bi-Gaussian IP.")
parser.add_argument('-gamma', type=float, default=0.0, help="Lorentzian width (gamma) for Voigt profile convolution.")
parser.add_argument('-site', type=str, default=None, help="Observatory name recognised by astropy (e.g. 'Keck').")
parser.add_argument('-loc', type=str, default=None, help="Manual 'lat,lon,height' in deg,deg,m. Overrides -site.")
args = parser.parse_args()

NUM_OBS = args.num_obs
SYNTHETIC_CSV = args.file
rv_values = ast.literal_eval(args.vel_list) if args.vel_list else np.linspace(-1000, 1000, NUM_OBS)

# Parse time step from string like '3d0h'
time_step_match = re.match(r'(\d+)d(\d+)h', args.time_step)
if time_step_match:
    delta_days = int(time_step_match.group(1))
    delta_hours = int(time_step_match.group(2))
    time_delta = timedelta(days=delta_days, hours=delta_hours)
else:
    raise ValueError("Invalid time_step format. Use format 'NdMh' like '3d0h'.")

# --- Build EarthLocation ---
def get_observer_location(args):
    """
    Resolve the observer's geodetic position.

    Priority:
      1. -loc 'lat,lon,height'
      2. -site recognised by EarthLocation.of_site
      3. Fallback to TLS constants in write_default_fits()
    """
    if args.loc:
        try:
            lat, lon, h = map(float, args.loc.split(','))
            return EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=h*u.m)
        except ValueError:
            raise ValueError("'-loc' must be 'lat,lon,height' in deg,deg,m.")
    if args.site:
        try:
            return EarthLocation.of_site(args.site)
        except Exception:
            raise ValueError(f"Site '{args.site}' not recognised by Astropy.")
    # TLS defaults (degrees & metres)
    return EarthLocation(lat=-24.6268*u.deg, lon=-70.4045*u.deg, height=2648*u.m)

observer_location = get_observer_location(args)

# --- Main ---
star_wave, star_flux = read_spectrum_csv(SYNTHETIC_CSV) # Read the synthetic stellar spectrum.
fts_wave, fts_flux = read_FTS_fits() # Read the FTS iodine spectrum.

print("RV values used (m/s):", rv_values)
base_time = datetime(2025, 2, 6, 0, 0, 0) # Define the base observation time.

for i in range(NUM_OBS):
    print(f"\n--- Simulation {i+1} ---")
  
    obs_time = base_time + i * time_delta # Calculate the observation time for the current simulation.
    obstime = Time(obs_time, scale='utc')
    target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)

    barycorr = target.radial_velocity_correction(obstime=obstime,
                                                 location=observer_location,
                                                 kind='barycentric')
    barycorr_ms = barycorr.to(u.m/u.s).value

    # Apply total RV shift (user - barycentric)
    total_rv = rv_values[i]
    shifted_wave_rv = apply_rv_shift(star_wave, total_rv)  # Apply RV shift to the stellar spectrum.
    shifted_wave_all = apply_rv_shift(shifted_wave_rv, -1 * barycorr_ms) 

    flux_conv = convolve_IP(shifted_wave_all, star_flux, fts_wave, fts_flux, width=args.ip_width) # Convolve the shifted spectrum with FTS and IP.
    wave_orders, flux_orders = slice_orders(shifted_wave_all, flux_conv) # Slice the convolved flux into echelle orders.
    date_obs_str = obs_time.isoformat(timespec='milliseconds') # Format the observation time as an ISO string.
    write_default_fits(OUTPUT_TEMPLATE.format(i+1), wave_orders, flux_orders, date_obs_str) # Write the simulated data to a FITS file.

if args.template:
    print("\n--- Writing template FITS (no convolution) ---")
  
    obstime_tpl = Time(base_time, scale='utc')
    barycorr_tpl = target.radial_velocity_correction(obstime=obstime_tpl,
                                                     location=observer_location,
                                                     kind='barycentric')
    barycorr_ms_tpl = barycorr_tpl.to(u.m/u.s).value
  
    wave_tpl_corrected = apply_rv_shift(star_wave, -1 * barycorr_ms_tpl)
    w_orders_tpl, f_orders_tpl = slice_orders(wave_tpl_corrected, star_flux)
    write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl, base_time.isoformat(timespec='milliseconds'))
    
    # w_orders_tpl, f_orders_tpl = slice_orders(star_wave, star_flux) # Slice the original stellar spectrum for the template.
    # write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl, base_time.isoformat(timespec='milliseconds')) # Write the template FITS file.
