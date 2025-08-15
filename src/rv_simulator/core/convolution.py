"""
Instrumental profile convolution functions
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.special import wofz

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

def convolve_IP(shifted_wave: np.ndarray, star_flux: np.ndarray, fts_wave: np.ndarray, 
                fts_flux: np.ndarray, width: float, ip_type: str = 'gaussian', 
                asymmetry: float = 0.0, gamma: float = 0.0) -> np.ndarray:
    """
    Convolve the stellar spectrum and FTS reference with the selected IP profile.
    
    Parameters
    ----------
    shifted_wave: wavelength grid of shifted stellar spectrum [Å]
    star_flux: stellar flux on shifted_wave
    fts_wave: wavelength grid of iodine FTS reference [Å]
    fts_flux: FTS transmission on fts_wave
    width: Gaussian sigma (in pixels) for IP
    ip_type: 'gaussian', 'bigaussian', or 'voigt'
    asymmetry: asymmetry factor for bi-Gaussian (-1 to 1)
    gamma: Lorentzian width for Voigt profile
    
    Returns
    -------
    convolved_flux: interpolated convolved flux on shifted_wave grid
    """
    wave_min = max(shifted_wave.min(), fts_wave.min())
    wave_max = min(shifted_wave.max(), fts_wave.max())
    if wave_max <= wave_min:
        raise ValueError("No overlapping wavelength region")

    dx = min(np.min(np.diff(shifted_wave)), np.min(np.diff(fts_wave)))
    npts = max(2, int(np.floor((wave_max - wave_min) / dx)) + 1)
    wave_uni = np.linspace(wave_min, wave_max, npts)

    star_uni = np.interp(wave_uni, shifted_wave, star_flux, left=1.0, right=1.0)
    fts_uni = np.interp(wave_uni, fts_wave, fts_flux, left=1.0, right=1.0)
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


# def convolve_IP(
#     shifted_wave: np.ndarray, star_flux: np.ndarray, fts_wave: np.ndarray, fts_flux: np.ndarray, width: float, ip_type: str = 'gaussian', asymmetry: float = 0.0, gamma: float = 0.0) -> np.ndarray:
#     """
#     Performs a physically correct convolution by operating in velocity space,
#     while maintaining compatibility with the original function signature.

#     Parameters:
#       shifted_wave: Wavelength grid of the stellar spectrum [Å].
#       star_flux:    Stellar flux.
#       fts_wave:     Wavelength grid of the iodine FTS reference [Å].
#       fts_flux:     FTS transmission.
#       width:        The primary IP width (sigma) in units of [km/s].
#       ip_type:      The profile type: 'gaussian', 'bigaussian', or 'voigt'.
#       asymmetry:    Asymmetry factor for the bi-Gaussian profile.
#       gamma:        Lorentzian width (HWHM) in [km/s] for the Voigt profile.
#     """

#     # --- Helper functions for creating IP kernels ---
#     def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
#         """Creates a standard Gaussian kernel."""
#         if sigma <= 0:
#             kernel = np.zeros(size)
#             kernel[size // 2] = 1.0
#             return kernel
#         x = np.arange(size) - size // 2
#         kernel = np.exp(-0.5 * (x / sigma)**2)
#         return kernel / np.sum(kernel)

#     def _bigaussian_kernel(size: int, sigma: float, asymmetry: float) -> np.ndarray:
#         """Creates an asymmetric (bi-Gaussian) kernel."""
#         if sigma <= 0:
#             kernel = np.zeros(2 * size + 1)
#             kernel[size] = 1.0
#             return kernel
#         x = np.arange(-size, size + 1)
#         sigma_left = sigma * (1 + asymmetry)
#         sigma_right = sigma * (1 - asymmetry)
#         kernel = np.where(x < 0,
#                           np.exp(-0.5 * (x / sigma_left)**2),
#                           np.exp(-0.5 * (x / sigma_right)**2))
#         return kernel / np.sum(kernel)

#     def _voigt_kernel(size: int, sigma: float, gamma: float) -> np.ndarray:
#         """Generates a Voigt profile kernel."""
#         if sigma <= 0 and gamma <= 0:
#             kernel = np.zeros(2 * size + 1)
#             kernel[size] = 1.0
#             return kernel
#         sigma = max(sigma, 1e-6)
#         x = np.linspace(-size, size, 2 * size + 1)
#         z = (x + 1j * gamma) / (sigma * np.sqrt(2))
#         kernel = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
#         return kernel / np.sum(kernel)

#     # ---------------------------------------------------------

#     C_KMS = 299792.458 # Speed of light in km/s

#     # -- Step 1: Create a uniform grid in log-wavelength (velocity) space --
#     wave_min = max(shifted_wave.min(), fts_wave.min())
#     wave_max = min(shifted_wave.max(), fts_wave.max())
    
#     # Oversample the finest resolution of the input spectrum by a factor of 2
#     min_delta_wave = np.min(np.diff(shifted_wave))
#     d_log_wave = (min_delta_wave / wave_min) / 2.0
    
#     log_wave_uni = np.arange(np.log(wave_min), np.log(wave_max), d_log_wave)
#     wave_uni = np.exp(log_wave_uni)

#     # -- Step 2: Resample the model onto the log-uniform grid --
#     star_uni = np.interp(log_wave_uni, np.log(shifted_wave), star_flux, left=1.0, right=1.0)
#     fts_uni = np.interp(log_wave_uni, np.log(fts_wave), fts_flux, left=1.0, right=1.0)
#     model_uni = star_uni * fts_uni

#     # -- Step 3: Create the IP kernel in velocity space --
#     # Velocity resolution of our grid in km/s per bin
#     dv_per_bin = d_log_wave * C_KMS
    
#     # Convert width parameters from physical units [km/s] to grid units [bins]
#     # This is required by the kernel-generating functions.
#     sigma_in_bins = width / dv_per_bin
#     gamma_in_bins = gamma / dv_per_bin

#     # Define kernel size to be ~5 sigma wide.
#     kernel_half_bins = int(np.ceil(5 * sigma_in_bins))

#     if ip_type == 'gaussian':
#         # For a simple Gaussian, the size is the full width
#         kernel = _gaussian_kernel(2 * kernel_half_bins + 1, sigma_in_bins)
#     elif ip_type == 'bigaussian':
#         # The bi-Gaussian helper takes the half-size
#         kernel = _bigaussian_kernel(kernel_half_bins, sigma_in_bins, asymmetry)
#     elif ip_type == 'voigt':
#         # The Voigt helper also takes the half-size
#         kernel = _voigt_kernel(kernel_half_bins, sigma_in_bins, gamma_in_bins)
#     else:
#         raise ValueError(f"Invalid ip_type: '{ip_type}'. Choose 'gaussian', 'bigaussian', or 'voigt'.")

#     # -- Step 4: Perform the convolution --
#     # Use fftconvolve for speed with large kernels, mode='same' keeps array size.
#     model_conv = fftconvolve(model_uni, kernel, mode='same')
    
#     # -- Step 5: Interpolate the result back to the original wavelength grid --
#     convolved_flux = np.interp(shifted_wave, wave_uni, model_conv, left=1.0, right=1.0)
    
#     return convolved_flux
