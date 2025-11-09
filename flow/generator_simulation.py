#! /usr/bin/env python3
"""
Simulate VIPER FITS Generator
-----------------------------

This script generates a series of synthetic stellar observations in the VIPER FITS format.
It simulates observations by Doppler-shifting a given synthetic spectrum, multiplying it
with an iodine FTS reference, and convolving the result with a user-defined Gaussian
instrumental profile (IP). The output is sliced into echelle orders and saved as
multi-extension FITS files ready for use with VIPER.

Three Operation Modes:
---------------------

1. DEFAULT MODE (default):
   Uses synthetic CSV spectrum + hardcoded instrument settings
   python3 generator_simulation.py -file spectrum.csv -num_obs 3 -vel_list "[100,200,300]"

2. AUTO MODE: 
   Auto-infers everything from provided FITS files
   python3 generator_simulation.py -mode auto -observed_spectrum obs.fits -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"
   
3. MANUAL MODE:
   Uses observed spectrum + user-provided order ranges in text format
   python3 generator_simulation.py -mode manual -observed_spectrum obs.fits -orders_file my_orders.txt -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"

Arguments:
----------
-mode                Look at the 'Operation Modes'
-observed_spectrum   provide one sample to make it run (riskier)
-orders_file         Or provide a listed txt file of exact order used in your instrument
-num_obs             Number of observations to generate.
-vel_list            RV shifts (m/s) for each observation as a list.
-date_list           non-equal spacing of dates.
-time_step           Spacing between observations, e.g. '3d0h' for 3 days and 0 hours.
-file                Path to the synthetic spectrum CSV file.
-ip_width            IP width in pixels (Gaussian sigma).
-ip_type             Type of instrumental profile to convolve with: 'gaussian', 'bigaussian', or 'voigt'.
-asymmetry           Asymmetry factor (-1 to 1) for bi-Gaussian IP.
-gamma               Lorentzian width (gamma) for Voigt profile convolution.
-template            Optional flag to create a template FITS file.
-site                Optional flag: Observatory name recognised by astropy (e.g. 'Keck').
-add_noise           Adds noise

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
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import fftconvolve
from scipy.special import wofz
from scipy.interpolate import interp1d
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
OUTPUT_TEMPLATE_FILE = 'template_viper_data_tpl.fits'
FTS_FILE = 'FTS_default_TLS.fits'
VIPER_IP_HALF_SIZE = 50

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
    with fits.open(FTS_FILE) as hdul:
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

def infer_instrument_from_fits(filename):
    """
    Automatically infer instrument type and parameters from FITS file.
    """
    print(f"Auto-inferring instrument from: {filename}")
    try:
        hdul = fits.open(filename, ignore_blank=True)
        header = hdul[0].header
        instrument_name = header.get('INSTRUME', '').upper()
        telescop = header.get('TELESCOP', '').upper()
        resolution = header.get('RESOLUT', header.get('RESOLUTION', None))
        pixel_scale = header.get('PIXSCALE', header.get('CDELT1', None))
        print(f"  Detected instrument: {instrument_name}")
        print(f"  Telescope: {telescop}")
        print(f"  Resolution: {resolution}")
        print(f"  Pixel scale: {pixel_scale}")
        hdul.close()
        return {
            'instrument_name': instrument_name,
            'telescop': telescop,
            'resolution': resolution,
            'pixel_scale': pixel_scale
        }
    except Exception as e:
        print(f"Warning: Could not read header info: {e}")
        return {}

def auto_detect_orders_from_fits(filename):
    """
    Detect order wavelength ranges for:
      • simulated FITS (one 'wave' column per extension)
      • raw multi-WL tables
      • 2D multispec image primaries (fallback to default order_wavs)
    """
    print(f"Auto-detecting orders from: {filename}")
    hdul = fits.open(filename, ignore_blank=True)

    # Case A: 2D multispec image primary -> fallback to default hardcoded
    primary = hdul[0]
    if primary.data is not None and primary.header.get('CTYPE1','').lower().startswith('multi'):
        print("Detected MULTISPE image primary — using hardcoded default order ranges.")
        hdul.close()
        return order_wavs.copy()

    # Case B: simulated FITS (one 'wave' column per BinTableHDU)
    single_wave = True
    for ext in hdul[1:]:
        if not isinstance(ext, fits.BinTableHDU) or ext.data is None:
            single_wave = False
            break
        cols = [c.lower() for c in ext.columns.names]
        if cols != ['wave']:
            single_wave = False
            break
    if single_wave:
        order_ranges = []
        for idx in range(1, len(hdul)):
            wave = hdul[idx].data['wave']
            mn, mx = float(wave.min()), float(wave.max())
            order_ranges.append((mn, mx))
            print(f"HDU {idx}: {mn:.2f} - {mx:.2f} Å")
        hdul.close()
        print(f"Auto-detected {len(order_ranges)} orders")
        return order_ranges

    # Case C: raw tables with *_WL or *wave* columns
    order_ranges = []
    for idx in range(1, len(hdul)):
        hdu = hdul[idx]
        if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
            continue
        for col in hdu.columns.names:
            if 'wl' in col.lower() or 'wave' in col.lower():
                wave = hdu.data[col]
                mn, mx = float(wave.min()), float(wave.max())
                order_ranges.append((mn, mx))
                print(f"HDU {idx}, col '{col}': {mn:.2f} - {mx:.2f} Å")
    hdul.close()

    if not order_ranges:
        raise ValueError("No valid orders detected. Ensure FITS has MULTISPE primary or columns with 'wl'/'wave'.")
    print(f"Auto-detected {len(order_ranges)} orders")
    return order_ranges

def infer_fts_format(filename):
    """
    Automatically infer FTS file format and parameters.
    """
    print(f"Auto-inferring FTS format from: {filename}")
    hdul = fits.open(filename, ignore_blank=True, output_verify='silentfix')
    hdr = hdul[0].header if len(hdul) > 0 else {}
    wavetype = hdr.get('wavetype', hdr.get('WAVETYPE', 'wavelength')).lower()
    unit = hdr.get('unit', hdr.get('BUNIT', hdr.get('CUNIT1', 'angstrom'))).lower()
    data_info = {}
    for hdu in hdul:
        if hasattr(hdu, 'data') and hdu.data is not None:
            if hasattr(hdu.data, 'names'):
                data_info['format'] = 'table'
                data_info['columns'] = list(hdu.data.names)
                if 'wave' in hdu.data.names:
                    sample = hdu.data['wave'][:10]
                    data_info['wave_range'] = (float(sample.min()), float(sample.max()))
                break
            elif hdu.data.ndim >= 2:
                data_info['format'] = 'image'
                data_info['shape'] = hdu.data.shape
                break
    hdul.close()
    if 'wave_range' in data_info:
        wmin, wmax = data_info['wave_range']
        if wmax < 100:
            unit = 'micrometer' if wmax < 2 else 'nm'
        elif wmax > 10000:
            wavetype, unit = 'wavenumber', 'cm-1'
        else:
            unit = 'angstrom'
    print(f"Inferred - wavetype: {wavetype}, unit: {unit}")
    print(f"Data format: {data_info.get('format', 'unknown')}")
    return {'wavetype': wavetype, 'unit': unit, 'data_info': data_info}
    
def read_orders_from_file(filename):
    """
    Read order wavelength ranges from text file.
    Expected format: one range per line as (min_wav, max_wav) tuples.
    """
    print(f"Reading order ranges from: {filename}")
    order_ranges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('(') and line.endswith(')'):
                a, b = map(float, line[1:-1].split(','))
            else:
                a, b = map(float, line.split(','))
            order_ranges.append((a, b))
    if not order_ranges:
        raise ValueError("No valid order ranges found in file")
    print(f"Loaded {len(order_ranges)} order ranges")
    for i,(a,b) in enumerate(order_ranges[:5]):
        print(f"Order {i}: {a:.2f} - {b:.2f} Å")
    if len(order_ranges)>5:
        print(f"... and {len(order_ranges)-5} more orders")
    return order_ranges

def read_FTS_fits_auto(fts_file, format_info=None):
    file = fts_file or FTS_FILE
    print(f"Reading FTS spectrum from: {file}")
    hdul = fits.open(file, ignore_blank=True, output_verify='silentfix')
    hdr0 = hdul[0].header
    wavetype = hdr0.get('wavetype', hdr0.get('WAVETYPE', '')).lower()
    unit = hdr0.get('unit', hdr0.get('BUNIT', hdr0.get('CUNIT1', ''))).lower()
    print(f"Header info: wavetype='{wavetype}', unit='{unit}'")
    
    # Try HDU[1] if it exists and looks like a table with needed columns
    if len(hdul) > 1 and hasattr(hdul[1], 'data') and hdul[1].data is not None and \
       hasattr(hdul[1].data, 'names') and 'wave' in hdul[1].data.names and 'flux' in hdul[1].data.names:
        w = hdul[1].data['wave']
        f = hdul[1].data['flux']
        print("Using HDU[1] with columns: wave, flux")
    else:
        hdul.close()
        # Instead of fallback, raise error immediately
        raise ValueError(f"FTS FITS file '{file}' does not contain a valid HDU[1] with 'wave' and 'flux' columns.")
    
    # Unit conversion
    if wavetype == 'wavenumber':
        print("Converting wavenumber to wavelength")
        w = 1e8 / w[::-1]
        f = f[::-1]
    if unit == 'nm':
        print(f" Converting {unit} to Angstrom")
        w *= 10
    
    hdul.close()
    print(f"Final FTS spectrum: {len(w)} points")
    print(f"Wavelength range: {w.min():.1f} - {w.max():.1f} Å")
    return w, f

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Creates a standard Gaussian kernel."""
    if sigma <= 0:
        kernel = np.zeros(size)
        kernel[size // 2] = 1.0
        return kernel
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma)**2)
    return kernel / np.sum(kernel)

def _bigaussian_kernel(size: int, sigma: float, asymmetry: float) -> np.ndarray:
    """Creates an asymmetric (bi-Gaussian) kernel."""
    if sigma <= 0:
        kernel = np.zeros(2 * size + 1)
        kernel[size] = 1.0
        return kernel
    x = np.arange(-size, size + 1)
    sigma_left = sigma * (1 + asymmetry)
    sigma_right = sigma * (1 - asymmetry)
    kernel = np.where(x < 0, np.exp(-0.5 * (x / sigma_left)**2), np.exp(-0.5 * (x / sigma_right)**2))
    return kernel / np.sum(kernel)

def _voigt_kernel(size: int, sigma: float, gamma: float) -> np.ndarray:
    """Generates a Voigt profile kernel."""
    if sigma <= 0 and gamma <= 0:
        kernel = np.zeros(2 * size + 1)
        kernel[size] = 1.0
        return kernel
    sigma = max(sigma, 1e-6)
    x = np.linspace(-size, size, 2 * size + 1)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    kernel = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return kernel / np.sum(kernel)

# =============================================================================
# def simulate_stellar_rv(times_s, i, rng=None, rv_amp_qp=2.0, prot_days=25.0, tau_qp_days=20.0, tau_ou_hours=2.0, sigma_ou=1.5, noise_scale=1.0):
#     """
#     Generate stellar RV activity noise (in m/s) with adjustable noise scale.
# 
#     Parameters
#     ----------
#     times_s : array
#         Array of times (in seconds) for all observations.
#     i : int
#         Index of current observation.
#     rng : np.random.Generator
#         Random number generator (optional).
#     noise_scale : float
#         Factor to reduce overall noise amplitude (default=0.5).
#         Values < 1.0 => lower noise, > 1.0 => higher noise.
#     """
#     rng = np.random.default_rng() if rng is None else rng
# 
#     # Quasi-periodic (rotation-like, decaying)
#     Prot_s = prot_days * 86400.0
#     tau_qp_s = tau_qp_days * 86400.0
#     rv_qp = noise_scale * rv_amp_qp * np.exp(-(times_s[i]-times_s[0])/tau_qp_s) * \
#             np.cos(2*np.pi*(times_s[i]-times_s[0])/Prot_s)
# 
#     # Ornstein-Uhlenbeck (granulation-like)
#     if i == 0:
#         simulate_stellar_rv.rv_ou_prev = 0.0
#         simulate_stellar_rv.last_time = times_s[0]
# 
#     dt = times_s[i] - simulate_stellar_rv.last_time
#     a = np.exp(-dt / (tau_ou_hours*3600.0))
#     var = (noise_scale * sigma_ou)**2 * (1 - a*a)
#     rv_ou = a*simulate_stellar_rv.rv_ou_prev + rng.normal(0.0, np.sqrt(var))
# 
#     # Update memory
#     simulate_stellar_rv.rv_ou_prev = rv_ou
#     simulate_stellar_rv.last_time = times_s[i]
# 
#     return rv_qp + rv_ou  # [m/s]
# =============================================================================

def simulate_stellar_rv(times_s, i, noise_scale=1.0):
    
    rng = np.random.default_rng()
    rv_noise = rng.normal(0.0, 1.5 * noise_scale)
    return rv_noise  # [m/s]

def create_instrument_grids(order_ranges, oversample_factor=1, default_pixels=4096):
    """
    Creates a list of fixed, uniform log-lambda grids for the instrument.
    Returns a list of tuples: (wave_grid, d_log_wave)
    """
    print("Creating fixed instrument grids...")
    instrument_grids = []
    
    # Estimate the best d_log_wave to use
    min_dlog = np.inf
    for start, end in order_ranges:
        # Estimate resolution (dlambda/lambda)
        dlambda = (end - start) / default_pixels # Assume N pixels
        dlog = dlambda / start
        if dlog < min_dlog:
            min_dlog = dlog
    
    # Use an oversampled, fixed d_log_wave for all orders
    d_log_wave = min_dlog / oversample_factor
    print(f"Using master d_log_wave: {d_log_wave:e} (dv = {d_log_wave * SPEED_OF_LIGHT:.2f} m/s)")
    
    for i, (start, end) in enumerate(order_ranges):
        log_start = np.log(start)
        log_end = np.log(end)
        n_pts = int(np.ceil((log_end - log_start) / d_log_wave))
        if n_pts < 2:
            print(f"WARNING: Order {i} ({start:.1f}-{end:.1f} Å) has < 2 points. Skipping.")
            continue
        
        log_wave_grid = np.linspace(log_start, log_end, n_pts)
        wave_grid = np.exp(log_wave_grid)
        instrument_grids.append((wave_grid, d_log_wave))
        # print(f"Order {i}: {start:.1f}-{end:.1f} Å, {n_pts} points")
        
    print(f"Created {len(instrument_grids)} fixed grids.")
    return instrument_grids

def convolve_IP_on_uniform_grid(flux_uni, d_log_wave, width_kms, ip_type='gaussian', asymmetry=0.0, gamma=0.0):
    """
    Convolves flux *already on a uniform log-lambda grid*.
    width_kms is the IP width in km/s.
    d_log_wave is the log-lambda step of the grid.
    """
    # --- Build IP kernel ---
    dv_per_bin = d_log_wave * (SPEED_OF_LIGHT / 1000.0)  # km/s per bin
    if dv_per_bin <= 0:
        raise ValueError("d_log_wave must be positive")
        
    sigma_bins = width_kms / dv_per_bin
    gamma_bins = gamma / dv_per_bin
    
    # We use a FIXED half-size to match VIPER's internal 'IP_hs'
    half_bins = VIPER_IP_HALF_SIZE

    if ip_type == 'gaussian':
        kernel = _gaussian_kernel(2 * half_bins + 1, sigma_bins)
    elif ip_type == 'bigaussian':
        kernel = _bigaussian_kernel(half_bins, sigma_bins, asymmetry)
    elif ip_type == 'voigt':
        kernel = _voigt_kernel(half_bins, sigma_bins, gamma_bins)
    else:
        raise ValueError(f"Invalid ip_type: {ip_type}")

    # --- Convolution ---
    return np.convolve(flux_uni, kernel, mode='valid')

# =============================================================================
# def build_master_lnwave_and_order_slices(fts_wave, order_ranges, oversample_factor=3, default_pixels=4096):
#     print("Building master ln(wave) grid (VIPER-style)...")
#     min_wav = min(start for start, _ in order_ranges)
#     max_wav = max(end for _, end in order_ranges)
#     dlambda = (max_wav - min_wav) / default_pixels
#     d_log_wave = (dlambda / min_wav) / oversample_factor
#     ln_min, ln_max = np.log(min_wav), np.log(max_wav)
#     n_pts = int(np.ceil((ln_max - ln_min) / d_log_wave))
#     lnwave_j_full = np.linspace(ln_min, ln_max, n_pts)
# 
#     order_slices = []
#     for start, end in order_ranges:
#         mask = (np.exp(lnwave_j_full) >= start) & (np.exp(lnwave_j_full) <= end)
#         indices = np.where(mask)[0]
#         if len(indices) > 1:
#             order_slices.append(slice(indices[0], indices[-1]+1))
#     print(f"Built lnwave grid with {len(lnwave_j_full)} points, {len(order_slices)} orders.")
#     return lnwave_j_full, order_slices, d_log_wave
# 
# 
# def make_ip_kernel_on_lngrid(d_log_wave, width_kms, ip_type='gaussian', asymmetry=0.0, gamma=0.0, half_size=VIPER_IP_HALF_SIZE):
#     dv_per_bin_kms = d_log_wave * (SPEED_OF_LIGHT / 1000.0)
#     sigma_bins = width_kms / dv_per_bin_kms
#     gamma_bins = gamma / dv_per_bin_kms
#     half_bins = max(int(np.ceil(6.0 * sigma_bins)), half_size)
#     size = 2 * half_bins + 1
# 
#     if ip_type == 'gaussian':
#         kernel = _gaussian_kernel(size, sigma_bins)
#     elif ip_type == 'bigaussian':
#         kernel = _bigaussian_kernel(half_bins, sigma_bins, asymmetry)
#     elif ip_type == 'voigt':
#         kernel = _voigt_kernel(half_bins, sigma_bins, gamma_bins)
#     else:
#         raise ValueError(f"Invalid ip_type: {ip_type}")
# 
#     return kernel / np.sum(kernel), half_bins
# 
# 
# def convolve_on_lnwave_valid(lnwave_j, star_flux_ln, fts_flux_ln, ip_kernel):
#     model = star_flux_ln * fts_flux_ln
#     result = np.convolve(model, ip_kernel, mode='valid')
#     hs = (len(ip_kernel) - 1)//2
#     lnwave_j_eff = lnwave_j[hs:-hs]
#     return result, lnwave_j_eff
# 
# 
# def get_pixel_lnwave(wave_array, degree=2):
#     n = len(wave_array)
#     pixels = np.arange(n)
#     xcen = (n - 1)/2.0
#     coeffs = np.polyfit(pixels - xcen, wave_array, degree)
#     def lnwave_func(pix):
#         return np.log(np.polyval(coeffs, pix - xcen))
#     return coeffs, lnwave_func
# =============================================================================


# =============================================================================
# def convolve_IP_on_uniform_grid(flux_in, d_log_wave, ip_width_kms, ip_type='Gaussian', asymmetry=0.0, gamma=2.0):
#     """
#     Convolves a flux array (on a uniform log-lambda grid) with an IP kernel
#     using np.convolve (VIPER-style).
#     
#     ip_width_kms is the standard deviation (s) in km/s.
#     """
#     dv_step = d_log_wave * SPEED_OF_LIGHT # velocity step in km/s
#     if dv_step <= 0:
#         raise ValueError("d_log_wave must be positive for convolution.")
# 
#     # Kernel size (e.g., +/- 5 sigma width)
#     # Ensure kernel is large enough, at least 11 points
#     half_width_bins = int(np.ceil(5 * ip_width_kms / dv_step))
#     half_bins = max(half_width_bins, 11) 
#     
#     kernel_points = 2 * half_bins + 1
#     vk_center = half_bins
#     vk_indices = np.arange(kernel_points) - vk_center
#     
#     # Scale vk_indices to km/s
#     vk_km_s = vk_indices * dv_step 
#     
#     # --- IP Kernel Generation (Following VIPER core functions) ---
#     if ip_type.lower() == 'gaussian':
#         IP_k = np.exp(-(vk_km_s/ip_width_kms)**2/2)
#     elif ip_type.lower() == 'super-gaussian':
#         IP_k = np.exp(-np.abs(vk_km_s/ip_width_kms)**gamma) 
#     else:
#         print(f"WARNING: IP type {ip_type} not implemented. Using Gaussian.")
#         IP_k = np.exp(-(vk_km_s/ip_width_kms)**2/2)
# 
#     IP_k /= IP_k.sum() # normalise IP
#     
#     # Convolve the flux using the numpy method
#     return np.convolve(flux_in, IP_k, mode='same')
# =============================================================================

# =============================================================================
# # --- Template helper ---
# def make_template_convolved(star_wave, star_flux, ip_width, ip_type='gaussian', asymmetry=0.0, gamma=0.0, oversample_factor=3):
#     fts_wave_local = star_wave.copy()
#     fts_flux_local = np.ones_like(star_flux)
#     tpl_conv = convolve_IP_on_uniform_grid(star_wave, star_flux, fts_wave_local, fts_flux_local,
#                            width=ip_width, ip_type=ip_type,
#                            asymmetry=asymmetry, gamma=gamma,
#                            oversample_factor=oversample_factor)
#     med = np.nanmedian(tpl_conv)
#     tpl_conv /= med if med > 0 and np.isfinite(med) else 1.0
#     return tpl_conv
# =============================================================================

# =============================================================================
# # --- Doppler shift ---
# def apply_rv_shift(wavelengths, rv_m_s):
#     # Applies a radial velocity shift to wavelengths.
#     return wavelengths * (1 + rv_m_s / SPEED_OF_LIGHT)
# =============================================================================

# --- Slice to orders ---
def slice_orders(wave, flux, order_ranges=None):
    # Slices the spectrum into echelle orders based on provided or default wavelength ranges.
    if order_ranges is None:
        order_ranges = order_wavs
    
    wave_orders, flux_orders = [], []
    for i, (start, end) in enumerate(order_ranges):
        mask = (wave >= start) & (wave <= end)
        wave_cut = wave[mask]
        flux_cut = flux[mask]
        
        if len(wave_cut) < 2:
            wave_orders.append(np.empty(0))
            flux_orders.append(np.empty(0))
        else:
            wave_orders.append(wave_cut)
            flux_orders.append(flux_cut)
    
    print(f"Sliced into {len(wave_orders)} orders using {'default' if order_ranges is order_wavs else 'custom'} ranges.")
    return wave_orders, flux_orders

def load_original_orders(filename):
    wave_orders, flux_orders = [], []
    with fits.open(filename, ignore_blank=True) as hdul:
        for hdu in hdul[1:]:
            if isinstance(hdu, fits.BinTableHDU):
                if 'wave' in hdu.columns.names and 'flux' in hdu.columns.names:
                    wave_orders.append(hdu.data['wave'])
                    flux_orders.append(hdu.data['flux'])
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
parser.add_argument('-mode', type=str, choices=['default', 'auto', 'manual'], default='default', help="Input mode: 'default' (synthetic CSV + hardcoded), 'auto' (auto-infer from files), 'manual' (user-provided text format)")
parser.add_argument('-observed_spectrum', type=str, default=None, help="Path to observed spectrum FITS file (for auto/manual modes)")
parser.add_argument('-fts_file', type=str, default=None, help="Path to FTS cell FITS file (for auto/manual modes)")
parser.add_argument('-orders_file', type=str, default=None, help="Path to text file with order wavelength ranges (for manual mode)")
parser.add_argument('-num_obs', type=int, default=10, help="Number of observations to generate.")
parser.add_argument('-vel_list', type=str, default=None, help="List of RVs (m/s) for each observation.")
parser.add_argument('-date_list', type=str, default=None, help="List of observation dates as ISO strings, e.g. ['2025-02-06T00:00:00', '2025-02-08T12:00:00'].")
parser.add_argument('-time_step', type=str, default='5d0h', help="Time spacing between observations (e.g., '3d0h' for 3 days and 0 hours).")
parser.add_argument('-template', action='store_true', help="Flag to create a template file.")
parser.add_argument('-file', type=str, default='spectrum.csv', help="Input CSV file for synthetic spectrum.")
parser.add_argument('-ip_width', type=str, default="1.2", help="Gaussian IP width (sigma) for convolution; can be a float or a list like [1.8, 2.2].")
parser.add_argument('-ip_type', type=str, default='gaussian', choices=['gaussian', 'bigaussian', 'voigt'], help="Type of instrumental profile to convolve with: 'gaussian', 'bigaussian', or 'voigt'.")
parser.add_argument('-asymmetry', type=float, default=0.0, help="Asymmetry factor (-1 to 1) for bi-Gaussian IP.")
parser.add_argument('-gamma', type=float, default=0.0, help="Lorentzian width (gamma) for Voigt profile convolution.")
parser.add_argument('-site', type=str, default=None, help="Observatory name recognised by astropy (e.g. 'Keck').")
parser.add_argument('-loc', type=str, default=None, help="Manual 'lat,lon,height' in deg,deg,m. Overrides -site.")
parser.add_argument('-add_noise', action='store_true', help="If set, add synthetic stellar RV noise (simulate_stellar_rv).")

args = parser.parse_args()
NUM_OBS = args.num_obs
SYNTHETIC_CSV = args.file

# --- Parse RV list immediately ---
rv_values = ast.literal_eval(args.vel_list) if args.vel_list else np.linspace(-1000, 1000, NUM_OBS)

# --- Read synthetic stellar spectrum (always) ---
star_wave, star_flux = read_spectrum_csv(SYNTHETIC_CSV)

# --- Read default FTS (always) ---
fts_wave, fts_flux = read_FTS_fits()

# --- Mode logic (no fallbacks) ---
if args.mode == 'default':
    order_ranges = order_wavs

elif args.mode == 'auto':
    order_ranges = auto_detect_orders_from_fits(args.observed_spectrum)
    if args.fts_file:
        fmt = infer_fts_format(args.fts_file)
        fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, fmt)

elif args.mode == 'manual':
    order_ranges = read_orders_from_file(args.orders_file)
    if args.fts_file:
        fmt = infer_fts_format(args.fts_file)
        fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, fmt)
        
# === INSERT OVERLAP CHECK HERE ===
# Check for wavelength overlap between order ranges and FTS spectrum
fts_min, fts_max = fts_wave.min(), fts_wave.max()
filtered_orders = []
for mn, mx in order_ranges:
    if mx >= fts_min and mn <= fts_max:
        filtered_orders.append((max(mn, fts_min), min(mx, fts_max)))

if not filtered_orders:
    raise RuntimeError(
        f"No overlapping wavelength regions found between detected orders and FTS spectrum.\n"
        f"Order wavelength ranges span {min(mn for mn, _ in order_ranges):.1f} - {max(mx for _, mx in order_ranges):.1f} Å\n"
        f"FTS wavelength range is {fts_min:.1f} - {fts_max:.1f} Å\n"
        f"Cannot generate synthetic data without overlap.")

order_ranges = filtered_orders
print(f"Filtered to {len(order_ranges)} orders overlapping FTS wavelength range.")

# Parse time step from string like '3d0h'
time_step_match = re.match(r'(\d+)d(\d+)h', args.time_step)
if time_step_match:
    delta_days = int(time_step_match.group(1))
    delta_hours = int(time_step_match.group(2))
    time_delta = timedelta(days=delta_days, hours=delta_hours)
else:
    raise ValueError("Invalid time_step format. Use format 'NdMh' like '3d0h'.")

if args.date_list:
    try:
        date_list = ast.literal_eval(args.date_list)
        if not isinstance(date_list, list):
            raise ValueError()
        obs_times = [datetime.fromisoformat(d) for d in date_list]
        NUM_OBS = len(obs_times)
        print(f"Using explicit date_list with {NUM_OBS} observations.")
    except Exception:
        raise ValueError("Invalid format for -date_list. Example: \"['2025-02-06T00:00:00', '2025-02-08T12:00:00']\"")
else:
    base_time = datetime(2025, 2, 6, 0, 0, 0)
    obs_times = [base_time + i * time_delta for i in range(NUM_OBS)]

# Optional sanity check: length-match
if len(rv_values) != NUM_OBS:
    raise ValueError(f"Length of velocity list ({len(rv_values)}) does not match number of observations ({NUM_OBS}).")

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

def parse_ip_width(ip_width_arg, num_obs):
    """
    Parses ip_width, which can be a float or a range list (e.g. [1.8,2.2]).
    Returns a list of ip_width values (length num_obs) for each simulation.
    """
    try:
        ip_width_range = ast.literal_eval(ip_width_arg)
        if isinstance(ip_width_range, (list, tuple)) and len(ip_width_range) == 2:
            low, high = float(ip_width_range[0]), float(ip_width_range[1])
            ip_widths = list(np.random.uniform(low, high, num_obs))
            return ip_widths
        elif isinstance(ip_width_range, float) or isinstance(ip_width_range, int):
            return [float(ip_width_range)]*num_obs
        else:
            raise ValueError
    except Exception:
        try:
            ipw = float(ip_width_arg)
            return [ipw]*num_obs
        except Exception:
            raise ValueError(f"Invalid format for ip_width: {ip_width_arg}")

ip_widths = parse_ip_width(str(args.ip_width), NUM_OBS)
np.savetxt("ip_widths.txt", ip_widths, fmt="%.6f")

observer_location = get_observer_location(args)

# --- Initialize spectrum and order data based on mode ---
print(f"\n=== MODE: {args.mode.upper()} ===")

if args.mode == 'default':
    print("Using default mode: CSV spectrum + hardcoded order ranges")
    # star_wave, star_flux = read_spectrum_csv(SYNTHETIC_CSV)
    fts_wave, fts_flux = read_FTS_fits()
    order_ranges = order_wavs  # Initialize for default mode

# --- Extract location from observed spectrum if provided ---
elif args.mode == 'auto' and args.observed_spectrum:  # REMOVE 'manual' from this condition
    try:
        hdul = fits.open(args.observed_spectrum, ignore_blank=True)
        hdr = hdul[0].header
        lat = hdr.get('TEL GEOLAT') or hdr.get('GEOLAT') or hdr.get('LAT')
        lon = hdr.get('TEL GEOLON') or hdr.get('GEOLON') or hdr.get('LON')
        elev = hdr.get('TEL GEOELEV') or hdr.get('GEOELEV') or hdr.get('ELEV')
        
        if lat is not None and lon is not None and elev is not None:
            observer_location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
            print(f"Using location from FITS header: lat={lat}, lon={lon}, elev={elev}")
        hdul.close()
    except Exception as e:
        print(f"Warning: Could not extract location from FITS header: {e}")


elif args.mode == 'manual':
    print("Using manual mode: user-provided order ranges from text file")
    if not args.orders_file:
        raise ValueError("Manual mode requires -orders_file argument")
    
    # Read order ranges from text file
    order_ranges = read_orders_from_file(args.orders_file)
    
    # Read synthetic spectrum (same as other modes)
    # star_wave, star_flux = read_spectrum_csv(SYNTHETIC_CSV)
    
    # Read FTS
    if args.fts_file:
        format_info = infer_fts_format(args.fts_file)
        fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, format_info)
    else:
        fts_wave, fts_flux = read_FTS_fits()

# --- Extract location from observed spectrum if provided ---
if args.mode in ['auto', 'manual'] and args.observed_spectrum:
    try:
        hdul = fits.open(args.observed_spectrum, ignore_blank=True)
        hdr = hdul[0].header
        lat = hdr.get('TEL GEOLAT') or hdr.get('GEOLAT') or hdr.get('LAT')
        lon = hdr.get('TEL GEOLON') or hdr.get('GEOLON') or hdr.get('LON')
        elev = hdr.get('TEL GEOELEV') or hdr.get('GEOELEV') or hdr.get('ELEV')
        
        if lat is not None and lon is not None and elev is not None:
            observer_location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
            print(f"Using location from FITS header: lat={lat}, lon={lon}, elev={elev}")
        hdul.close()
    except Exception as e:
        print(f"Warning: Could not extract location from FITS header: {e}")

print("RV values used (m/s):", rv_values)
t0 = obs_times[0]
times_s = np.array([(ot - t0).total_seconds() for ot in obs_times])

barycorr_values = []


# --- Create the fixed instrument grids BEFORE the loop ---
instrument_grids = create_instrument_grids(order_ranges, oversample_factor=3)
if not instrument_grids:
    raise RuntimeError("Failed to create any valid instrument grids from order_ranges.")

# =============================================================================
# fts_interp = interp1d(fts_wave, fts_flux, kind='cubic',
#                       bounds_error=False, fill_value=1.0)
#
# star_interp_master = interp1d(star_wave, star_flux, kind='cubic',
#                               bounds_error=False, fill_value=1.0)
# =============================================================================

# Create a single, wide-range interpolator for the FTS
fts_interp = lambda x: np.interp(x, fts_wave, fts_flux, left=fts_flux[0], right=fts_flux[-1])
star_interp_master = lambda x: np.interp(x, star_wave, star_flux, left=star_flux[0], right=star_flux[-1])

# =============================================================================
# 
# barycorr_values = []
# 
# # Build master log-lambda grid and order slices
# lnwave_j_full, order_slices, d_log_wave = build_master_lnwave_and_order_slices(fts_wave, order_ranges, oversample_factor=3, default_pixels=4096)
# print(f"Master lnwave grid built ({len(lnwave_j_full)} points).")
# 
# # Create interpolators
# print("Creating interpolators...")
# fts_interp = interp1d(fts_wave, fts_flux, kind='cubic', bounds_error=False, fill_value=1.0)
# star_interp_master = interp1d(star_wave, star_flux, kind='cubic', bounds_error=False, fill_value=1.0)
# 
# # Main synthesis loop
# print(f"\n=== GENERATING {NUM_OBS} OBSERVATIONS ===")
# for i in range(NUM_OBS):
#     print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")
#     obs_time = obs_times[i]
#     obstime = Time(obs_time, scale='utc')
#     target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
#     barycorr = target.radial_velocity_correction(obstime=obstime, location=observer_location, kind='barycentric')
#     barycorr_ms = barycorr.to(u.m/u.s).value
#     total_rv = rv_values[i]
#     rv_activity = simulate_stellar_rv(times_s, i) if args.add_noise else 0.0
#     total_shift = total_rv + rv_activity - barycorr_ms
#     print(f" RV={total_rv:.2f} m/s, BERV={barycorr_ms:.2f} m/s, Final shift={total_shift:.2f} m/s")
#     barycorr_values.append(barycorr_ms)
# 
#     # Doppler factor
#     doppler_factor = (1.0 + total_shift / SPEED_OF_LIGHT)
# 
#     # Build IP kernel once
#     ip_kernel, ip_hs_used = make_ip_kernel_on_lngrid(
#         d_log_wave, ip_widths[i], ip_type=args.ip_type,
#         asymmetry=args.asymmetry, gamma=args.gamma,
#         half_size=VIPER_IP_HALF_SIZE
#     )
# 
#     wave_orders_final, flux_orders_final = [], []
# 
#     for k, sl in enumerate(order_slices):
#         lnwave_j = lnwave_j_full[sl]
#         wave_j = np.exp(lnwave_j)
# 
#         if len(wave_j) < (2 * ip_hs_used + 3):
#             wave_orders_final.append(np.array([0.0]))
#             flux_orders_final.append(np.array([1.0]))
#             continue
# 
#         # Sample star and FTS on log-grid
#         star_flux_ln = star_interp_master(wave_j / doppler_factor)
#         fts_flux_ln = fts_interp(wave_j)
# 
#         # Convolve valid
#         Sj_eff, lnwave_eff = convolve_on_lnwave_valid(lnwave_j, star_flux_ln, fts_flux_ln, ip_kernel)
# 
#         # Map to pixel grid
#         wave_eff = np.exp(lnwave_eff)
#         coeffs, lnwave_func = get_pixel_lnwave(wave_eff)
#         npix = len(wave_eff)
#         pixels = np.arange(npix)
#         lnwave_obs = lnwave_func(pixels)
#         Si_eff = np.interp(lnwave_obs, lnwave_eff, Sj_eff, left=1.0, right=1.0)
# 
#         wave_orders_final.append(np.exp(lnwave_obs))
#         flux_orders_final.append(Si_eff)
# 
#     # Write FITS (unchanged format)
#     date_obs_str = obs_time.isoformat(timespec='milliseconds')
#     output_file = OUTPUT_TEMPLATE.format(i+1)
#     write_default_fits(output_file, wave_orders_final, flux_orders_final, date_obs_str)
#     print(f" -> {output_file}")
# 
#     # Diagnostics
#     flux_median = np.median(flux_orders_final[0])
#     with open("barycorr_flux_check.txt", "a") as f:
#         f.write(f"{i} {barycorr_ms:.3f} {flux_median:.6e}\n")
# 
# # --- Template generation ---
# if args.template:
#     print("\n=== GENERATING TEMPLATE ===")
#     first_obs_time = obs_times[0]
#     obstime_tpl = Time(first_obs_time, scale='utc')
#     target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
#     barycorr_tpl = target.radial_velocity_correction(obstime=obstime_tpl, location=observer_location, kind='barycentric')
#     barycorr_ms_tpl = barycorr_tpl.to(u.m/u.s).value
#     total_shift_tpl = -barycorr_ms_tpl
#     doppler_tpl = (1.0 + total_shift_tpl / SPEED_OF_LIGHT)
# 
#     ip_kernel_tpl, _ = make_ip_kernel_on_lngrid(d_log_wave, np.mean(ip_widths), ip_type=args.ip_type)
#     wave_orders_tpl, flux_orders_tpl = [], []
# 
#     for k, sl in enumerate(order_slices):
#         lnwave_j = lnwave_j_full[sl]
#         wave_j = np.exp(lnwave_j)
#         star_flux_tpl = star_interp_master(wave_j / doppler_tpl)
#         fts_flux_ln = fts_interp(wave_j)
#         Sj_eff, lnwave_eff = convolve_on_lnwave_valid(lnwave_j, star_flux_tpl, fts_flux_ln, ip_kernel_tpl)
#         wave_eff = np.exp(lnwave_eff)
#         coeffs, lnwave_func = get_pixel_lnwave(wave_eff)
#         npix = len(wave_eff)
#         lnwave_obs = lnwave_func(np.arange(npix))
#         Si_eff = np.interp(lnwave_obs, lnwave_eff, Sj_eff)
#         wave_orders_tpl.append(np.exp(lnwave_obs))
#         flux_orders_tpl.append(Si_eff)
# 
#     write_default_fits(OUTPUT_TEMPLATE_FILE, wave_orders_tpl, flux_orders_tpl, first_obs_time.isoformat(timespec='milliseconds'))
#     print(f" -> {OUTPUT_TEMPLATE_FILE}")
# =============================================================================

# =============================================================================
# # --- Main Loop ---
# print(f"\n=== GENERATING {NUM_OBS} OBSERVATIONS ===")
# for i in range(NUM_OBS):
#     print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")
#     obs_time = obs_times[i]
#     obstime = Time(obs_time, scale='utc')
#     target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
#     
#     barycorr = target.radial_velocity_correction(obstime=obstime, location=observer_location, kind='barycentric')
#     barycorr_ms = barycorr.to(u.m/u.s).value
#     total_rv = rv_values[i]
#     
#     # Optional stellar activity noise
#     if args.add_noise:
#         rv_activity = simulate_stellar_rv(times_s, i)
#         print(f" Stellar activity RV: {rv_activity:.2f} m/s")
#     else:
#         rv_activity = 0.0
#         
#     # Apply total shift: user RV + activity - barycentric correction
#     total_shift = total_rv + rv_activity - barycorr_ms
#     print(f" Total RV: {total_rv:.2f} m/s, Barycentric: {barycorr_ms:.2f} m/s")
#     print(f" Final shift applied: {total_shift:.2f} m/s")
#     barycorr_values.append(barycorr_ms)
# 
#     # -----------------------------------------------------------------
#     doppler_factor = (1.0 + total_shift / SPEED_OF_LIGHT)
#     star_interp = lambda w: star_interp_master(w / doppler_factor)
#     # -----------------------------------------------------------------
# 
#     wave_orders_final = []
#     flux_orders_final = []
#     
#     # print(f"Interpolating and convolving on {len(instrument_grids)} fixed order grids...")
#     
#     # --- New Inner Loop (per order) ---
#     for k, (wave_k, d_log_wave_k) in enumerate(instrument_grids):
#         
#         # 1. Interpolate star and FTS onto fixed instrument grid
#         #    The 'star_interp' call now uses our new lambda function
#         star_flux_k = star_interp(wave_k)
#         fts_flux_k  = fts_interp(wave_k)
#         
#         # 2. Multiply
#         model_k = star_flux_k * fts_flux_k
#         
#         # 3. Convolve on this uniform grid
#         flux_conv_k = convolve_IP_on_uniform_grid(
#             model_k, 
#             d_log_wave_k, 
#             ip_widths[i], # Use the width for this observation
#             ip_type=args.ip_type, 
#             asymmetry=args.asymmetry, 
#             gamma=args.gamma
#         )
#         
#         # Trim the wavelength grid to match the 'valid' convolution output
#         # The length (N - 2*IP_hs) will now perfectly match flux_conv_k
#         wave_k_eff = wave_k[VIPER_IP_HALF_SIZE : -VIPER_IP_HALF_SIZE]
#         wave_orders_final.append(wave_k_eff)
# 
#         flux_orders_final.append(flux_conv_k)
# 
#     # Write science FITS (Only write once)
#     date_obs_str = obs_time.isoformat(timespec='milliseconds')
#     output_file = OUTPUT_TEMPLATE.format(i+1)
#     # Use the new lists
#     write_default_fits(output_file, wave_orders_final, flux_orders_final, date_obs_str)
#     print(f" -> {output_file}")
#     
#     # === Diagnostic test for barycentric-flux correlation ===
#     flux_median = np.median(flux_orders_final[0])
#     debug_line = f"{i} {barycorr_ms:.3f} {flux_median:.6e}"
#     print(f"DEBUG {debug_line}")
#     
#     with open("barycorr_flux_check.txt", "a") as f:
#         f.write(debug_line + "\n")
# =============================================================================

# --- Main Loop ---
print(f"\n=== GENERATING {NUM_OBS} OBSERVATIONS ===")
for i in range(NUM_OBS):
    print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")
    obs_time = obs_times[i]
    obstime = Time(obs_time, scale='utc')
    target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
    
    barycorr = target.radial_velocity_correction(obstime=obstime, location=observer_location, kind='barycentric')
    barycorr_ms = barycorr.to(u.m/u.s).value
    total_rv = rv_values[i]
    
    if args.add_noise:
        rv_activity = simulate_stellar_rv(times_s, i)
    else:
        rv_activity = 0.0
        
    total_shift = total_rv + rv_activity - barycorr_ms
    print(f"Total RV: {total_rv:.2f} m/s, Barycentric: {barycorr_ms:.2f} m/s")
    print(f"Final shift applied: {total_shift:.2f} m/s")
    
    doppler_factor = (1.0 + total_shift / SPEED_OF_LIGHT)

    wave_orders_final = []
    flux_orders_final = []
    
    # --- New Inner Loop (per order) ---
    for k, (wave_k_instrument, d_log_wave_k) in enumerate(instrument_grids):
        
        # 1. Define the grid in the STAR'S REST FRAME.
        wave_k_rest = wave_k_instrument / doppler_factor
        
        # 2. Interpolate the star onto this REST FRAME grid.
        star_flux_k = star_interp_master(wave_k_rest)
        
        # 3. Interpolate the FTS onto the original INSTRUMENT FRAME grid.
        fts_flux_k  = fts_interp(wave_k_instrument)
        
        # 4. Multiply them.
        model_k = star_flux_k * fts_flux_k
        
        # 5. Convolve on this uniform grid.
        flux_conv_k = convolve_IP_on_uniform_grid(
            model_k, 
            d_log_wave_k, 
            ip_widths[i],
            ip_type=args.ip_type, 
            asymmetry=args.asymmetry, 
            gamma=args.gamma
        )
        
        # 6. The final wavelength grid is the original instrument grid, trimmed.
        wave_k_eff = wave_k_instrument[VIPER_IP_HALF_SIZE : -VIPER_IP_HALF_SIZE]
        
        wave_orders_final.append(wave_k_eff)
        flux_orders_final.append(flux_conv_k)

    # Write science FITS
    date_obs_str = obs_time.isoformat(timespec='milliseconds')
    output_file = OUTPUT_TEMPLATE.format(i+1)
    write_default_fits(output_file, wave_orders_final, flux_orders_final, date_obs_str)
    print(f" -> {output_file}")
    
    # === Diagnostic test ===
    flux_median = np.median(flux_orders_final[0])
    debug_line = f"{i} {barycorr_ms:.3f} {flux_median:.6e}"
    print(f"DEBUG {debug_line}")
    
    with open("barycorr_flux_check.txt", "a") as f:
        f.write(debug_line + "\n")

# --- Independent Template Generation ---
if args.template:
    print(f"\n=== GENERATING TEMPLATE ===")
    first_obs_time = obs_times[0]
    obstime_tpl = Time(first_obs_time, scale='utc')
    target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
    
    barycorr_tpl = target.radial_velocity_correction(obstime=obstime_tpl, location=observer_location, kind='barycentric')
    barycorr_ms_tpl = barycorr_tpl.to(u.m/u.s).value
    
    # -----------------------------------------------------------------
    # The shift is only the barycentric correction
    total_shift_tpl = -barycorr_ms_tpl
    
    # This is the Doppler factor for the template
    doppler_factor_tpl = (1.0 + total_shift_tpl / SPEED_OF_LIGHT)

    # Create the template interpolator function using the *same master*
    star_interp_tpl = lambda w: star_interp_master(w / doppler_factor_tpl)
    # -----------------------------------------------------------------
    
    w_orders_tpl = []
    f_orders_tpl = []
    
    # Interpolate onto the *fixed instrument grids*
    for k, (wave_k, d_log_wave_k) in enumerate(instrument_grids):
        flux_k = star_interp_tpl(wave_k)
        w_orders_tpl.append(wave_k)
        f_orders_tpl.append(flux_k)
        
    write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl, first_obs_time.isoformat(timespec='milliseconds'))
    print(f" -> {OUTPUT_TEMPLATE_FILE}")

# =============================================================================
# # --- Independent Template Generation (VIPER-like) ---
# if args.template:
#     print(f"\n=== GENERATING TEMPLATE ===")
#     
#     # 1. Get observation time for FITS header
#     first_obs_time = obs_times[0]
#     obstime_tpl = Time(first_obs_time, scale='utc')
#     target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
#     
#     # 2. Calculate the barycentric correction
#     barycorr_tpl = target.radial_velocity_correction(obstime=obstime_tpl, location=observer_location, kind='barycentric')
#     barycorr_ms_tpl = barycorr_tpl.to(u.m/u.s).value
#     
#     # 3. Apply the shift for an RV=0 star
#     shift_frac_tpl = -barycorr_ms_tpl / SPEED_OF_LIGHT
#     
#     # 4. Use the high-res uniform log grid from the main loop logic (G_HR)
#     # Recalculate G_HR to ensure independence and consistency
#     log_star_wave = np.log(star_wave)
#     min_dlog_star = np.min(np.diff(log_star_wave))
#     N_star_pts = int(np.ceil((log_star_wave.max() - log_star_wave.min()) / min_dlog_star))
#     lnwave_j = np.linspace(log_star_wave.min(), log_star_wave.max(), N_star_pts)
#     wave_j = np.exp(lnwave_j)
# 
#     # 5. Shift the log-lambda grid
#     lnwave_j_shifted = lnwave_j - shift_frac_tpl # shift star back to RV=0 in analysis frame
# 
#     # Interpolate ORIGINAL star flux onto the SHIFTED log-lambda grid
#     star_interp_orig_log = interp1d(log_star_wave, star_flux, kind='linear', bounds_error=False, fill_value=1.0)
#     star_flux_shifted_j = star_interp_orig_log(lnwave_j_shifted)
# 
#     # 6. Slice and Downsample to the Final Detector Grid (G_obs)
#     w_orders_tpl = []
#     f_orders_tpl = []
#     
#     # Final interpolation from G_HR (wave_j) to G_obs (wave_k)
#     final_interp = interp1d(wave_j, star_flux_shifted_j, kind='linear', bounds_error=False, fill_value=1.0)
#     
#     for k, (wave_k, d_log_wave_k) in enumerate(instrument_grids):
#         flux_k = final_interp(wave_k)
#         w_orders_tpl.append(wave_k)
#         f_orders_tpl.append(flux_k)
#         
#     write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl, first_obs_time.isoformat(timespec='milliseconds'))
#     print(f" -> {OUTPUT_TEMPLATE_FILE}")
# =============================================================================