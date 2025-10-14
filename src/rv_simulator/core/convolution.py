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

def convolve_IP(shifted_wave: np.ndarray, star_flux: np.ndarray, fts_wave: np.ndarray, fts_flux: np.ndarray, width: float, ip_type: str = 'gaussian', asymmetry: float = 0.0, gamma: float = 0.0, oversample_factor: int = 3) -> np.ndarray:
    """
    Convolve stellar spectrum × iodine FTS with instrumental profile (IP).
    Works in log-lambda space with oversampling for accuracy.
    """

    # --- Step 1: Common log grid (fixed reference dv per bin) ---
    wave_min = np.max([np.min(shifted_wave), np.min(fts_wave)])
    wave_max = np.min([np.max(shifted_wave), np.max(fts_wave)])

    if wave_max <= wave_min:
        raise ValueError("No overlap between star and FTS spectra")

    dlog_star = np.min(np.diff(np.log(shifted_wave)))
    dlog_fts  = np.min(np.diff(np.log(fts_wave)))
    base_dlog = np.nanmin([dlog_star, dlog_fts])
    if not np.isfinite(base_dlog) or base_dlog <= 0:
        raise ValueError("Invalid wavelength sampling")

    dlog = base_dlog / oversample_factor
    log_wave_uni = np.arange(np.log(wave_min), np.log(wave_max) + dlog, dlog)

    # --- Step 2: Interpolate star × iodine ---
    star_interp = interp1d(np.log(shifted_wave), star_flux, kind='cubic', bounds_error=False, fill_value=np.nan)
    fts_interp  = interp1d(np.log(fts_wave), fts_flux,  kind='cubic', bounds_error=False, fill_value=np.nan)

    star_vals = star_interp(log_wave_uni)
    fts_vals  = fts_interp(log_wave_uni)
    model_uni = star_vals * fts_vals

    if np.isnan(model_uni).any():
        median_val = np.nanmedian(model_uni)
        model_uni = np.where(np.isnan(model_uni), median_val, model_uni)

    # --- Step 3: Build IP kernel (constant physical width) ---
    dv_per_bin = dlog * (SPEED_OF_LIGHT / 1000)
    sigma_bins = width / dv_per_bin
    gamma_bins = gamma / dv_per_bin
    sigma_bins = np.clip(sigma_bins, 1e-6, None)
    half_bins  = int(np.ceil(5 * sigma_bins))

    if ip_type == 'gaussian':
        kernel = _gaussian_kernel(2*half_bins+1, sigma_bins)
    elif ip_type == 'bigaussian':
        kernel = _bigaussian_kernel(half_bins, sigma_bins, asymmetry)
    elif ip_type == 'voigt':
        kernel = _voigt_kernel(half_bins, sigma_bins, gamma_bins)
    else:
        raise ValueError(f"Invalid ip_type: {ip_type}")

    # print(f"[DEBUG] dv_per_bin = {dv_per_bin:.6f} km/s, "
    #   f"sigma_bins = {sigma_bins:.3f}, "
    #   f"kernel_size = {kernel.size}, "
    #   f"wave_range = [{wave_min:.2f}, {wave_max:.2f}] Å")  
    # print(f"[KERNEL DEBUG] width (input) = {width}, dv_per_bin = {dv_per_bin:.6f} km/s, sigma_bins = {sigma_bins:.3f}")

    # --- Step 4: Convolution (with reflection padding to avoid wrap) ---
    pad_len = kernel.size * 2
    padded_model = np.pad(model_uni, pad_len, mode='reflect')
    model_conv_padded = fftconvolve(padded_model, kernel, mode='same')
    model_conv = model_conv_padded[pad_len:-pad_len]

    # --- Step 5: Interpolate back to original grid ---
    conv_interp = interp1d(log_wave_uni, model_conv, kind='cubic',
                            bounds_error=False, fill_value=np.nan)
    conv_flux = conv_interp(np.log(shifted_wave))

    # Replace NaNs (edge zones) with original flux
    if np.isnan(conv_flux).any():
        conv_flux = np.where(np.isnan(conv_flux), star_flux, conv_flux)

    return conv_flux  # [same shape as star_flux]
