"""
Instrumental profile convolution functions
"""

import numpy as np
# from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import wofz
from ..utils.constants import SPEED_OF_LIGHT

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
    kernel = np.where(x < 0,
                      np.exp(-0.5 * (x / sigma_left)**2),
                      np.exp(-0.5 * (x / sigma_right)**2))
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
# def convolve_IP(shifted_wave: np.ndarray, star_flux: np.ndarray, fts_wave: np.ndarray, fts_flux: np.ndarray, width: float, ip_type: str = 'gaussian', asymmetry: float = 0.0, gamma: float = 0.0, oversample_factor: int = 3) -> np.ndarray:
#     """
#     Convolve stellar spectrum × iodine FTS with instrumental profile (IP).
#     Works in log-lambda space with oversampling for accuracy.
#     """
#     # --- Step 1: Common log grid ---
#     wave_min = np.max([np.min(shifted_wave), np.min(fts_wave)])
#     wave_max = np.min([np.max(shifted_wave), np.max(fts_wave)])
#     min_dlog_star = np.min(np.diff(shifted_wave) / shifted_wave[:-1])
#     min_dlog_fts  = np.min(np.diff(fts_wave) / fts_wave[:-1])
#     base_dlog = np.min([min_dlog_star, min_dlog_fts])
#     dlog = base_dlog / oversample_factor
#     log_wave_uni = np.arange(np.log(wave_min), np.log(wave_max), dlog)
# 
#     # --- Step 2: Interpolate star × iodine ---
#     star_interp = interp1d(np.log(shifted_wave), star_flux, kind='cubic', bounds_error=False, fill_value=np.nan)
#     fts_interp  = interp1d(np.log(fts_wave),    fts_flux,  kind='cubic', bounds_error=False, fill_value=np.nan)
#     model_uni = star_interp(log_wave_uni) * fts_interp(log_wave_uni)
#     model_uni = np.nan_to_num(model_uni, nan=np.nanmedian(star_flux))
# 
#     # --- Step 3: Build IP kernel ---
#     dv_per_bin = dlog * (SPEED_OF_LIGHT / 1000.0)  # km/s per pixel
#     sigma_bins = width / dv_per_bin
#     gamma_bins = gamma / dv_per_bin
#     half_bins  = int(np.ceil(5 * sigma_bins))
#     if ip_type == 'gaussian':
#         kernel = _gaussian_kernel(2*half_bins+1, sigma_bins)
#     elif ip_type == 'bigaussian':
#         kernel = _bigaussian_kernel(half_bins, sigma_bins, asymmetry)
#     elif ip_type == 'voigt':
#         kernel = _voigt_kernel(half_bins, sigma_bins, gamma_bins)
#     else:
#         raise ValueError(f"Invalid ip_type: {ip_type}")
# 
#     # --- Step 4: Convolution ---
#     model_conv = fftconvolve(model_uni, kernel, mode='same')
# 
#     # --- Step 5: Back to original grid ---
#     conv_interp = interp1d(log_wave_uni, model_conv, kind='cubic', bounds_error=False, fill_value=np.nanmedian(model_conv))
#     return conv_interp(np.log(shifted_wave))
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
