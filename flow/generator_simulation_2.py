# #! /usr/bin/env python3
# """
# Simulate VIPER FITS Generator
# -----------------------------

# Simulates synthetic stellar observations in the VIPER FITS format: Doppler-shifts
# a synthetic spectrum, multiplies it with an iodine FTS reference, convolves with a
# user-defined instrumental profile (IP), and slices the result into echelle orders.

# Three Operation Modes:
# -----------------------
# 1. DEFAULT: synthetic CSV spectrum + hardcoded order ranges.
#    python3 generator_simulation.py -file spectrum.csv -num_obs 3 -vel_list "[100,200,300]"
# 2. AUTO: order ranges + observer location auto-inferred from an observed FITS file.
#    The stellar flux is ALWAYS taken from -file (the CSV) in every mode; -observed_spectrum
#    is metadata-only (orders + site location), never a flux source.
#    python3 generator_simulation.py -mode auto -observed_spectrum obs.fits -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"
# 3. MANUAL: order ranges from a user-supplied text file.
#    python3 generator_simulation.py -mode manual -orders_file my_orders.txt -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"

# Key argument notes:
# --------------------
# -ip_width   IP sigma in PIXELS of the internal convolution grid (matches VIPER's
#             IP_hs sampling-knot convention). That grid runs at native resolution
#             (the finer of the stellar/FTS sampling) -- no oversampling is applied.
# -gamma      Voigt Lorentzian width, same pixel units as -ip_width.

# Outputs:
# --------
# - simulated_viper_data_1.fits, ..., simulated_viper_data_N.fits
# - Optional: template_viper_data_tpl.fits (clean, barycentric-corrected stellar template;
#   no IP convolution / no FTS multiplication, by design)
# """
# import re
# import ast
# import argparse
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from astropy.io import fits
# from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
# from astropy.time import Time
# import astropy.units as u
# from scipy.signal import fftconvolve
# from scipy.special import wofz
# from scipy.interpolate import interp1d

# solar_system_ephemeris.set('de430')

# # --- Constants ---
# SPEED_OF_LIGHT = 299792458.0  # m/s
# OUTPUT_TEMPLATE = 'simulated_viper_data_{}.fits'
# OUTPUT_TEMPLATE_FILE = 'template_viper_data_tpl.fits'
# FTS_FILE = 'FTS_default_TLS.fits'
# TLS_SITE = dict(lat=-24.6268, lon=-70.4045, elev=2648.0)  # fallback site (TLS)
# TARGET = SkyCoord(ra=86.819720 / 15.0 * u.hourangle, dec=-51.06714 * u.deg)

# # Default echelle order wavelength ranges (Angstrom), TLS
# order_wavs = [
#     (4513.00, 4593.39), (4549.25, 4630.49), (4586.10, 4668.20),
#     (4623.56, 4706.52), (4661.65, 4745.47), (4700.39, 4785.07),
#     (4739.79, 4825.34), (4779.87, 4866.28), (4820.64, 4907.92),
#     (4862.13, 4950.28), (4904.35, 4993.37), (4947.32, 5037.21),
#     (4991.06, 5081.83), (5035.58, 5127.24), (5080.92, 5173.47),
#     (5127.09, 5220.54), (5174.12, 5268.47), (5222.02, 5317.29),
#     (5270.82, 5367.01), (5320.55, 5417.68), (5371.23, 5469.30),
#     (5422.89, 5521.92), (5475.55, 5575.55), (5529.25, 5630.24),
#     (5584.02, 5686.01), (5639.89, 5742.89), (5696.88, 5800.91),
#     (5755.04, 5860.12), (5814.40, 5920.55), (5875.00, 5982.24),
#     (5936.88, 6045.21), (6000.07, 6109.53), (6064.62, 6175.23),
#     (6130.57, 6242.35), (6197.97, 6310.94), (6266.87, 6381.06),
#     (6337.32, 6452.74), (6409.38, 6526.05), (6483.09, 6601.04),
#     (6558.51, 6677.77), (6635.72, 6756.30), (6714.76, 6836.69),
#     (6795.72, 6919.01), (6878.65, 7003.33), (6963.64, 7089.73),
#     (7050.77, 7178.27), (7140.11, 7269.05), (7231.75, 7362.14),
#     (7325.79, 7457.65), (7422.32, 7555.65), (7521.45, 7656.25),
# ]

# # =============================================================================
# # --- I/O ---
# # =============================================================================

# def read_spectrum_csv(file):
#     data = pd.read_csv(file)
#     colmap = {c.lower(): c for c in data.columns}
#     try:
#         wave = data[colmap.get('wave', colmap.get('wavelength'))].values
#         flux = data[colmap['flux']].values
#     except KeyError:
#         raise KeyError(f"CSV must contain 'wave'/'wavelength' and 'flux'. Found: {data.columns.tolist()}")
#     print(f"Loaded synthetic spectrum with {len(wave)} points.")
#     return wave, flux


# def read_FTS_fits():
#     with fits.open(FTS_FILE) as hdul:
#         for i, hdu in enumerate(hdul):
#             if hasattr(hdu, 'columns') and hdu.data is not None:
#                 names = [n.lower() for n in hdu.columns.names]
#                 if 'wave' in names and 'flux' in names:
#                     wave = hdu.data[hdu.columns.names[names.index('wave')]]
#                     flux = hdu.data[hdu.columns.names[names.index('flux')]]
#                     print(f"Loaded FTS iodine spectrum with {len(wave)} points from HDU {i}.")
#                     return wave, flux
#     raise KeyError("FTS FITS file must contain 'wave' and 'flux' columns in at least one HDU.")


# def infer_fts_format(filename):
#     print(f"Auto-inferring FTS format from: {filename}")
#     with fits.open(filename, ignore_blank=True, output_verify='silentfix') as hdul:
#         hdr0 = hdul[0].header
#         wavetype = hdr0.get('wavetype', hdr0.get('WAVETYPE', 'wavelength')).lower()
#         unit = hdr0.get('unit', hdr0.get('BUNIT', hdr0.get('CUNIT1', 'angstrom'))).lower()
#         wave_range = None
#         for hdu in hdul:
#             if hdu.data is not None and hasattr(hdu.data, 'names') and hdu.data.names and 'wave' in hdu.data.names:
#                 sample = hdu.data['wave'][:10]
#                 wave_range = (float(sample.min()), float(sample.max()))
#                 break
#     if wave_range:
#         wmin, wmax = wave_range
#         if wmax < 100:
#             unit = 'micrometer' if wmax < 2 else 'nm'
#         elif wmax > 10000:
#             wavetype, unit = 'wavenumber', 'cm-1'
#         else:
#             unit = 'angstrom'
#     print(f"  Inferred - wavetype: {wavetype}, unit: {unit}")
#     return {'wavetype': wavetype, 'unit': unit}


# def read_FTS_fits_auto(fts_file, format_info):
#     print(f"Reading FTS spectrum from: {fts_file}")
#     with fits.open(fts_file, ignore_blank=True, output_verify='silentfix') as hdul:
#         if len(hdul) > 1 and hdul[1].data is not None and hasattr(hdul[1].data, 'names') \
#            and 'wave' in hdul[1].data.names and 'flux' in hdul[1].data.names:
#             w = np.array(hdul[1].data['wave'], dtype=float)
#             f = np.array(hdul[1].data['flux'], dtype=float)
#         else:
#             raise ValueError(f"'{fts_file}' has no valid HDU[1] with 'wave'/'flux' columns.")

#     if format_info['wavetype'] == 'wavenumber':
#         print(" Converting wavenumber to wavelength")
#         w, f = 1e8 / w[::-1], f[::-1]
#     if format_info['unit'] == 'nm':
#         print(" Converting nm to Angstrom")
#         w = w * 10

#     print(f" Final FTS spectrum: {len(w)} points, {w.min():.1f}-{w.max():.1f} Å")
#     return w, f


# def auto_detect_orders_from_fits(filename):
#     """Detect order ranges from a simulated/raw FITS file; falls back to the
#     hardcoded default table for 2D multispec image primaries."""
#     print(f"Auto-detecting orders from: {filename}")
#     with fits.open(filename, ignore_blank=True) as hdul:
#         primary = hdul[0]
#         if primary.data is not None and primary.header.get('CTYPE1', '').lower().startswith('multi'):
#             print("  Detected MULTISPE image primary -> using hardcoded default order ranges.")
#             return order_wavs.copy()

#         # One 'wave' column per extension (simulated-style FITS)
#         single_wave = all(
#             isinstance(ext, fits.BinTableHDU) and ext.data is not None
#             and [c.lower() for c in ext.columns.names] == ['wave']
#             for ext in hdul[1:]
#         )
#         order_ranges = []
#         if single_wave and len(hdul) > 1:
#             for idx in range(1, len(hdul)):
#                 wave = hdul[idx].data['wave']
#                 order_ranges.append((float(wave.min()), float(wave.max())))
#         else:
#             for hdu in hdul[1:]:
#                 if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
#                     continue
#                 for col in hdu.columns.names:
#                     if 'wl' in col.lower() or 'wave' in col.lower():
#                         wave = hdu.data[col]
#                         order_ranges.append((float(wave.min()), float(wave.max())))

#     if not order_ranges:
#         raise ValueError("No valid orders detected (need a MULTISPE primary or 'wl'/'wave' columns).")
#     print(f"  Auto-detected {len(order_ranges)} orders")
#     return order_ranges


# def read_orders_from_file(filename):
#     print(f"Reading order ranges from: {filename}")
#     order_ranges = []
#     with open(filename) as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
#             line = line.strip('()')
#             a, b = map(float, line.split(','))
#             order_ranges.append((a, b))
#     if not order_ranges:
#         raise ValueError("No valid order ranges found in file")
#     print(f"  Loaded {len(order_ranges)} order ranges")
#     return order_ranges


# def get_observer_location(args):
#     """Priority: -loc 'lat,lon,height' > -site (astropy) > observed_spectrum header > TLS default."""
#     if args.loc:
#         try:
#             lat, lon, h = map(float, args.loc.split(','))
#         except ValueError:
#             raise ValueError("'-loc' must be 'lat,lon,height' in deg,deg,m.")
#         return EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=h * u.m)
#     if args.site:
#         try:
#             return EarthLocation.of_site(args.site)
#         except Exception:
#             raise ValueError(f"Site '{args.site}' not recognised by Astropy.")
#     if args.observed_spectrum:
#         try:
#             with fits.open(args.observed_spectrum, ignore_blank=True) as hdul:
#                 hdr = hdul[0].header
#                 lat = hdr.get('TEL GEOLAT') or hdr.get('GEOLAT') or hdr.get('LAT')
#                 lon = hdr.get('TEL GEOLON') or hdr.get('GEOLON') or hdr.get('LON')
#                 elev = hdr.get('TEL GEOELEV') or hdr.get('GEOELEV') or hdr.get('ELEV')
#             if lat is not None and lon is not None and elev is not None:
#                 print(f"Using location from FITS header: lat={lat}, lon={lon}, elev={elev}")
#                 return EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
#         except Exception as e:
#             print(f"Warning: could not extract location from FITS header: {e}")
#     return EarthLocation(lat=TLS_SITE['lat'] * u.deg, lon=TLS_SITE['lon'] * u.deg, height=TLS_SITE['elev'] * u.m)


# # =============================================================================
# # --- IP kernels ---
# # =============================================================================

# def _gaussian_kernel(size, sigma):
#     if sigma <= 0:
#         k = np.zeros(size); k[size // 2] = 1.0; return k
#     x = np.arange(size) - size // 2
#     k = np.exp(-0.5 * (x / sigma) ** 2)
#     return k / k.sum()


# def _bigaussian_kernel(half, sigma, asymmetry):
#     if sigma <= 0:
#         k = np.zeros(2 * half + 1); k[half] = 1.0; return k
#     x = np.arange(-half, half + 1)
#     sigma_l, sigma_r = sigma * (1 + asymmetry), sigma * (1 - asymmetry)
#     k = np.where(x < 0, np.exp(-0.5 * (x / sigma_l) ** 2), np.exp(-0.5 * (x / sigma_r) ** 2))
#     return k / k.sum()


# def _voigt_kernel(half, sigma, gamma):
#     if sigma <= 0 and gamma <= 0:
#         k = np.zeros(2 * half + 1); k[half] = 1.0; return k
#     sigma = max(sigma, 1e-6)
#     x = np.linspace(-half, half, 2 * half + 1)
#     z = (x + 1j * gamma) / (sigma * np.sqrt(2))
#     k = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
#     return k / k.sum()


# def simulate_stellar_rv(times_s, i, rng=None, rv_amp_qp=2.0, prot_days=25.0,
#                          tau_qp_days=20.0, tau_ou_hours=2.0, sigma_ou=1.5, noise_scale=1.0):
#     """Quasi-periodic (rotation) + Ornstein-Uhlenbeck (granulation) stellar RV jitter [m/s]."""
#     rng = np.random.default_rng() if rng is None else rng
#     Prot_s, tau_qp_s = prot_days * 86400.0, tau_qp_days * 86400.0
#     rv_qp = noise_scale * rv_amp_qp * np.exp(-(times_s[i] - times_s[0]) / tau_qp_s) * \
#         np.cos(2 * np.pi * (times_s[i] - times_s[0]) / Prot_s)

#     if i == 0:
#         simulate_stellar_rv.rv_ou_prev = 0.0
#         simulate_stellar_rv.last_time = times_s[0]
#     dt = times_s[i] - simulate_stellar_rv.last_time
#     a = np.exp(-dt / (tau_ou_hours * 3600.0))
#     var = (noise_scale * sigma_ou) ** 2 * (1 - a * a)
#     rv_ou = a * simulate_stellar_rv.rv_ou_prev + rng.normal(0.0, np.sqrt(var))
#     simulate_stellar_rv.rv_ou_prev, simulate_stellar_rv.last_time = rv_ou, times_s[i]
#     return rv_qp + rv_ou


# def convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width,
#                  ip_type='gaussian', asymmetry=0.0, gamma=0.0):
#     """
#     Multiplies stellar x FTS on a common uniform log-wavelength grid, convolves with
#     the IP kernel, and interpolates back onto `shifted_wave`.

#     No oversampling: the grid step equals the FINER of the stellar/FTS native
#     sampling (needed to resolve narrow iodine lines without aliasing) -- never finer.

#     width / gamma : IP sigma / Lorentzian width, in PIXELS of that internal grid
#                      (VIPER's IP_hs convention), not km/s and not the output pixel grid.
#     """
#     wave_min = max(shifted_wave.min(), fts_wave.min())
#     wave_max = min(shifted_wave.max(), fts_wave.max())
#     if wave_max <= wave_min:
#         raise ValueError("No overlapping wavelength region between stellar and FTS spectra.")

#     dx = min(np.min(np.diff(np.log(shifted_wave))), np.min(np.diff(np.log(fts_wave))))
#     half = max(1, int(np.ceil(5 * max(width, gamma, 1e-6))))  # kernel half-size, pixels
#     buffer = half * dx
#     log_grid = np.arange(np.log(wave_min) - buffer, np.log(wave_max) + buffer, dx)

#     star_interp = interp1d(np.log(shifted_wave), star_flux, kind='linear', bounds_error=False, fill_value=1.0)
#     fts_interp = interp1d(np.log(fts_wave), fts_flux, kind='linear', bounds_error=False, fill_value=1.0)
#     model = star_interp(log_grid) * fts_interp(log_grid)

#     if ip_type == 'gaussian':
#         kernel = _gaussian_kernel(2 * half + 1, width)
#     elif ip_type == 'bigaussian':
#         kernel = _bigaussian_kernel(half, width, asymmetry)
#     elif ip_type == 'voigt':
#         kernel = _voigt_kernel(half, width, gamma)
#     else:
#         raise ValueError(f"Invalid ip_type: '{ip_type}'.")

#     model_conv = fftconvolve(model, kernel, mode='same')
#     back_interp = interp1d(log_grid, model_conv, kind='linear', bounds_error=False, fill_value=1.0)
#     return back_interp(np.log(shifted_wave))


# def apply_rv_shift(wavelengths, rv_m_s):
#     return wavelengths * (1 + rv_m_s / SPEED_OF_LIGHT)


# def slice_orders(wave, flux, order_ranges, target_pixels=2048):
#     """Slices the spectrum into echelle orders by wavelength range, then thins each
#     order down toward ~target_pixels by simple integer decimation (every Nth native
#     sample) if it's denser than that. Never interpolates/invents points, and never
#     upsamples a sparser order -- it just leaves sparse orders as-is."""
#     wave_orders, flux_orders = [], []
#     for start, end in order_ranges:
#         mask = (wave >= start) & (wave <= end)
#         w, f = wave[mask], flux[mask]
#         if len(w) < 2:
#             wave_orders.append(np.empty(0))
#             flux_orders.append(np.empty(0))
#             continue
#         step = max(1, len(w) // target_pixels)
#         wave_orders.append(w[::step])
#         flux_orders.append(f[::step])
#     print(f"Sliced into {len(wave_orders)} orders (~{target_pixels} px target, decimation only).")
#     return wave_orders, flux_orders


# def write_default_fits(filename, wave_orders, flux_orders, date_obs, location):
#     print(f"Writing FITS file: {filename}")
#     hdr = fits.Header()
#     hdr.set('DATE-OBS', date_obs)
#     hdr.set('RA', 86.819720 / 15.0, 'Right ascension (hours)')
#     hdr.set('DEC', -51.06714)
#     hdr.set('EXP', 30)
#     hdr.set('TEL GEOLAT', location.lat.to(u.deg).value)
#     hdr.set('TEL GEOLON', location.lon.to(u.deg).value)
#     hdr.set('TEL GEOELEV', location.height.to(u.m).value)

#     hdulist = [fits.PrimaryHDU(header=hdr)]
#     for idx, (w, f) in enumerate(zip(wave_orders, flux_orders)):
#         if len(w) == 0:
#             w, f = np.array([0.0]), np.array([1.0])
#         # c1 = fits.Column(name='wave', array=w, format='D')
#         # c2 = fits.Column(name='flux', array=f, format='D')
#         c1 = fits.Column(name='wave', array=np.asarray(w, dtype=np.float64), format='D')
#         c2 = fits.Column(name='flux', array=np.asarray(f, dtype=np.float64), format='D')
#         hdu = fits.BinTableHDU.from_columns([c1, c2])
#         hdu.name = f'ORDER_{idx}'
#         hdulist.append(hdu)
#     fits.HDUList(hdulist).writeto(filename, overwrite=True)


# def parse_ip_width(ip_width_arg, num_obs):
#     """Float -> constant; [low, high] -> per-observation uniform draw."""
#     try:
#         val = ast.literal_eval(ip_width_arg)
#     except Exception:
#         val = float(ip_width_arg)
#     if isinstance(val, (list, tuple)) and len(val) == 2:
#         return list(np.random.uniform(float(val[0]), float(val[1]), num_obs))
#     if isinstance(val, (int, float)):
#         return [float(val)] * num_obs
#     raise ValueError(f"Invalid format for ip_width: {ip_width_arg}")


# # =============================================================================
# # --- CLI ---
# # =============================================================================

# parser = argparse.ArgumentParser(description="Generate simulated VIPER FITS files with synthetic RV shifts and IP convolution.")
# parser.add_argument('-mode', choices=['default', 'auto', 'manual'], default='default',
#                      help="'default' (CSV + hardcoded orders), 'auto' (infer orders+location from -observed_spectrum), 'manual' (orders from -orders_file)")
# parser.add_argument('-observed_spectrum', default=None, help="FITS file used for order/location metadata (auto/manual modes). Never a flux source.")
# parser.add_argument('-fts_file', default=None, help="Path to FTS iodine cell FITS file. Defaults to FTS_default_TLS.fits.")
# parser.add_argument('-orders_file', default=None, help="Text file of order wavelength ranges (manual mode).")
# parser.add_argument('-num_obs', type=int, default=10, help="Number of observations to generate.")
# parser.add_argument('-vel_list', default=None, help="List of RVs (m/s) for each observation, e.g. '[100,200,300]'.")
# parser.add_argument('-date_list', default=None, help="List of ISO observation dates, e.g. \"['2025-02-06T00:00:00']\".")
# parser.add_argument('-time_step', default='5d0h', help="Spacing between observations, e.g. '3d0h'.")
# parser.add_argument('-template', action='store_true', help="Also write a clean stellar template FITS file.")
# parser.add_argument('-file', default='spectrum.csv', help="Synthetic spectrum CSV (always the stellar flux source).")
# parser.add_argument('-ip_width', default="1.2", help="IP sigma in pixels of the internal convolution grid; float or '[low,high]'.")
# parser.add_argument('-ip_type', choices=['gaussian', 'bigaussian', 'voigt'], default='gaussian')
# parser.add_argument('-asymmetry', type=float, default=0.0, help="Asymmetry (-1 to 1) for bi-Gaussian IP.")
# parser.add_argument('-gamma', type=float, default=0.0, help="Voigt Lorentzian width, same pixel units as -ip_width.")
# parser.add_argument('-site', default=None, help="Observatory name recognised by astropy (e.g. 'Keck').")
# parser.add_argument('-loc', default=None, help="Manual 'lat,lon,height' in deg,deg,m. Overrides -site.")
# parser.add_argument('-add_noise', action='store_true', help="Add synthetic stellar RV activity jitter.")
# args = parser.parse_args()

# # --- Load spectra (stellar flux is always the CSV; FTS defaults unless overridden) ---
# star_wave, star_flux = read_spectrum_csv(args.file)
# fts_wave, fts_flux = read_FTS_fits()

# # --- Resolve order ranges + FTS override, once, per mode ---
# print(f"\n=== MODE: {args.mode.upper()} ===")
# if args.mode == 'default':
#     order_ranges, order_source = order_wavs, 'hardcoded default'

# elif args.mode == 'auto':
#     if not args.observed_spectrum:
#         raise ValueError("Auto mode requires -observed_spectrum.")
#     order_ranges, order_source = auto_detect_orders_from_fits(args.observed_spectrum), 'auto-detected'
#     if args.fts_file:
#         fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, infer_fts_format(args.fts_file))

# elif args.mode == 'manual':
#     if not args.orders_file:
#         raise ValueError("Manual mode requires -orders_file.")
#     order_ranges, order_source = read_orders_from_file(args.orders_file), 'user-supplied'
#     if args.fts_file:
#         fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, infer_fts_format(args.fts_file))

# # --- Keep only orders that overlap the FTS wavelength range ---
# fts_min, fts_max = fts_wave.min(), fts_wave.max()
# order_ranges = [(max(mn, fts_min), min(mx, fts_max)) for mn, mx in order_ranges if mx >= fts_min and mn <= fts_max]
# if not order_ranges:
#     raise RuntimeError(f"No orders overlap the FTS wavelength range ({fts_min:.1f}-{fts_max:.1f} Å).")
# print(f"Using {len(order_ranges)} {order_source} orders overlapping the FTS range.")

# observer_location = get_observer_location(args)

# # --- Observation timestamps (finalizes NUM_OBS, since -date_list can override -num_obs) ---
# NUM_OBS = args.num_obs
# if args.date_list:
#     date_list = ast.literal_eval(args.date_list)
#     obs_times = [datetime.fromisoformat(d) for d in date_list]
#     NUM_OBS = len(obs_times)
#     print(f"Using explicit date_list with {NUM_OBS} observations.")
# else:
#     m = re.match(r'(\d+)d(\d+)h', args.time_step)
#     if not m:
#         raise ValueError("Invalid -time_step format. Use 'NdMh', e.g. '3d0h'.")
#     time_delta = timedelta(days=int(m.group(1)), hours=int(m.group(2)))
#     base_time = datetime(2025, 2, 6, 0, 0, 0)
#     obs_times = [base_time + i * time_delta for i in range(NUM_OBS)]

# # --- Everything sized to NUM_OBS is derived only now, after NUM_OBS is final ---
# rv_values = ast.literal_eval(args.vel_list) if args.vel_list else list(np.linspace(-1000, 1000, NUM_OBS))
# if len(rv_values) != NUM_OBS:
#     raise ValueError(f"vel_list length ({len(rv_values)}) != num_obs ({NUM_OBS}).")

# ip_widths = parse_ip_width(str(args.ip_width), NUM_OBS)
# np.savetxt("ip_widths.txt", ip_widths, fmt="%.6f")

# t0 = obs_times[0]
# times_s = np.array([(t - t0).total_seconds() for t in obs_times])
# print("RV values used (m/s):", rv_values)

# # =============================================================================
# # --- Main loop: simulated observations ---
# # =============================================================================

# barycorr_values = []
# print(f"\n=== GENERATING {NUM_OBS} OBSERVATIONS ===")
# for i in range(NUM_OBS):
#     print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")
#     obstime = Time(obs_times[i], scale='utc')
#     barycorr_ms = TARGET.radial_velocity_correction(
#         obstime=obstime, location=observer_location, kind='barycentric').to(u.m / u.s).value
#     barycorr_values.append(barycorr_ms)

#     rv_activity = simulate_stellar_rv(times_s, i) if args.add_noise else 0.0
#     total_shift = rv_values[i] + rv_activity - barycorr_ms
#     shifted_wave = apply_rv_shift(star_wave, total_shift)
#     print(f" Total RV: {rv_values[i]:.2f} m/s, Barycentric: {barycorr_ms:.2f} m/s, Applied shift: {total_shift:.2f} m/s")

#     flux_conv = convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width=ip_widths[i],
#                              ip_type=args.ip_type, asymmetry=args.asymmetry, gamma=args.gamma)
#     wave_orders, flux_orders = slice_orders(shifted_wave, flux_conv, order_ranges)

#     output_file = OUTPUT_TEMPLATE.format(i + 1)
    
#     write_default_fits(output_file, wave_orders, flux_orders,
#                         obs_times[i].isoformat(timespec='milliseconds'), observer_location)
#     print(f" -> {output_file}")

# # =============================================================================
# # --- Optional template: clean, barycentric-corrected stellar spectrum only
# #     (no IP convolution, no FTS multiplication -- a "ground truth" template) ---
# # =============================================================================

# if args.template:
#     print("\n=== GENERATING TEMPLATE ===")
#     obstime_tpl = Time(obs_times[0], scale='utc')
#     barycorr_ms_tpl = TARGET.radial_velocity_correction(
#         obstime=obstime_tpl, location=observer_location, kind='barycentric').to(u.m / u.s).value
#     wave_tpl = apply_rv_shift(star_wave, -barycorr_ms_tpl)
#     w_orders_tpl, f_orders_tpl = slice_orders(wave_tpl, star_flux, order_ranges)
#     write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl,
#                         obs_times[0].isoformat(timespec='milliseconds'), observer_location)
#     print(f" -> {OUTPUT_TEMPLATE_FILE}")

# print(f"\nGenerated {NUM_OBS} observation file(s){' + 1 template' if args.template else ''}.")
# print(f"Mode: {args.mode} | Order ranges: {order_source} ({len(order_ranges)} orders)")

# with open('barycorr_values.txt', 'w') as f:
#     for v in barycorr_values:
#         f.write(f"{v:.6f}\n")
# print(f"Saved {len(barycorr_values)} barycentric correction values to barycorr_values.txt")


#! /usr/bin/env python3
"""
Simulate VIPER FITS Generator
-----------------------------

Simulates synthetic stellar observations in the VIPER FITS format: Doppler-shifts
a synthetic spectrum, multiplies it with an iodine FTS reference, convolves with a
user-defined instrumental profile (IP), and slices the result into echelle orders.

Three Operation Modes:
-----------------------
1. DEFAULT: synthetic CSV spectrum + hardcoded order ranges.
   python3 generator_simulation.py -file spectrum.csv -num_obs 3 -vel_list "[100,200,300]"
2. AUTO: order ranges + observer location auto-inferred from an observed FITS file.
   The stellar flux is ALWAYS taken from -file (the CSV) in every mode; -observed_spectrum
   is metadata-only (orders + site location), never a flux source.
   python3 generator_simulation.py -mode auto -observed_spectrum obs.fits -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"
3. MANUAL: order ranges from a user-supplied text file.
   python3 generator_simulation.py -mode manual -orders_file my_orders.txt -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]"

Key argument notes:
--------------------
-ip_width   IP sigma in PIXELS of the internal convolution grid (matches VIPER's
            IP_hs sampling-knot convention). That grid runs at native resolution
            (the finer of the stellar/FTS sampling) -- no oversampling is applied.
-gamma      Voigt Lorentzian width, same pixel units as -ip_width.

Outputs:
--------
- simulated_viper_data_1.fits, ..., simulated_viper_data_N.fits
- Optional: template_viper_data_tpl.fits (clean, barycentric-corrected stellar template;
  no IP convolution / no FTS multiplication, by design)
"""
import re
import ast
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u
from scipy.signal import fftconvolve
from scipy.special import wofz
from scipy.interpolate import interp1d

solar_system_ephemeris.set('de430')

# --- Constants ---
SPEED_OF_LIGHT = 299792458.0  # m/s
OUTPUT_TEMPLATE = 'simulated_viper_data_{}.fits'
OUTPUT_TEMPLATE_FILE = 'template_viper_data_tpl.fits'
FTS_FILE = 'FTS_default_TLS.fits'
TLS_SITE = dict(lat=-24.6268, lon=-70.4045, elev=2648.0)  # fallback site (TLS)
TARGET = SkyCoord(ra=86.819720 / 15.0 * u.hourangle, dec=-51.06714 * u.deg)

# Default echelle order wavelength ranges (Angstrom), TLS
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
    (7325.79, 7457.65), (7422.32, 7555.65), (7521.45, 7656.25),
]

# =============================================================================
# --- I/O ---
# =============================================================================

def read_spectrum_csv(file):
    data = pd.read_csv(file)
    colmap = {c.lower(): c for c in data.columns}
    try:
        wave = data[colmap.get('wave', colmap.get('wavelength'))].values
        flux = data[colmap['flux']].values
    except KeyError:
        raise KeyError(f"CSV must contain 'wave'/'wavelength' and 'flux'. Found: {data.columns.tolist()}")
    print(f"Loaded synthetic spectrum with {len(wave)} points.")
    return wave, flux


def read_FTS_fits():
    with fits.open(FTS_FILE) as hdul:
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'columns') and hdu.data is not None:
                names = [n.lower() for n in hdu.columns.names]
                if 'wave' in names and 'flux' in names:
                    wave = hdu.data[hdu.columns.names[names.index('wave')]]
                    flux = hdu.data[hdu.columns.names[names.index('flux')]]
                    print(f"Loaded FTS iodine spectrum with {len(wave)} points from HDU {i}.")
                    return wave, flux
    raise KeyError("FTS FITS file must contain 'wave' and 'flux' columns in at least one HDU.")


def infer_fts_format(filename):
    print(f"Auto-inferring FTS format from: {filename}")
    with fits.open(filename, ignore_blank=True, output_verify='silentfix') as hdul:
        hdr0 = hdul[0].header
        wavetype = hdr0.get('wavetype', hdr0.get('WAVETYPE', 'wavelength')).lower()
        unit = hdr0.get('unit', hdr0.get('BUNIT', hdr0.get('CUNIT1', 'angstrom'))).lower()
        wave_range = None
        for hdu in hdul:
            if hdu.data is not None and hasattr(hdu.data, 'names') and hdu.data.names and 'wave' in hdu.data.names:
                sample = hdu.data['wave'][:10]
                wave_range = (float(sample.min()), float(sample.max()))
                break
    if wave_range:
        wmin, wmax = wave_range
        if wmax < 100:
            unit = 'micrometer' if wmax < 2 else 'nm'
        elif wmax > 10000:
            wavetype, unit = 'wavenumber', 'cm-1'
        else:
            unit = 'angstrom'
    print(f"  Inferred - wavetype: {wavetype}, unit: {unit}")
    return {'wavetype': wavetype, 'unit': unit}


def read_FTS_fits_auto(fts_file, format_info):
    print(f"Reading FTS spectrum from: {fts_file}")
    with fits.open(fts_file, ignore_blank=True, output_verify='silentfix') as hdul:
        if len(hdul) > 1 and hdul[1].data is not None and hasattr(hdul[1].data, 'names') \
           and 'wave' in hdul[1].data.names and 'flux' in hdul[1].data.names:
            w = np.array(hdul[1].data['wave'], dtype=float)
            f = np.array(hdul[1].data['flux'], dtype=float)
        else:
            raise ValueError(f"'{fts_file}' has no valid HDU[1] with 'wave'/'flux' columns.")

    if format_info['wavetype'] == 'wavenumber':
        print(" Converting wavenumber to wavelength")
        w, f = 1e8 / w[::-1], f[::-1]
    if format_info['unit'] == 'nm':
        print(" Converting nm to Angstrom")
        w = w * 10

    print(f" Final FTS spectrum: {len(w)} points, {w.min():.1f}-{w.max():.1f} Å")
    return w, f


def auto_detect_orders_from_fits(filename):
    """Detect order ranges from a simulated/raw FITS file; falls back to the
    hardcoded default table for 2D multispec image primaries."""
    print(f"Auto-detecting orders from: {filename}")
    with fits.open(filename, ignore_blank=True) as hdul:
        primary = hdul[0]
        if primary.data is not None and primary.header.get('CTYPE1', '').lower().startswith('multi'):
            print("  Detected MULTISPE image primary -> using hardcoded default order ranges.")
            return order_wavs.copy()

        # One 'wave' column per extension (simulated-style FITS)
        single_wave = all(
            isinstance(ext, fits.BinTableHDU) and ext.data is not None
            and [c.lower() for c in ext.columns.names] == ['wave']
            for ext in hdul[1:]
        )
        order_ranges = []
        if single_wave and len(hdul) > 1:
            for idx in range(1, len(hdul)):
                wave = hdul[idx].data['wave']
                order_ranges.append((float(wave.min()), float(wave.max())))
        else:
            for hdu in hdul[1:]:
                if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
                    continue
                for col in hdu.columns.names:
                    if 'wl' in col.lower() or 'wave' in col.lower():
                        wave = hdu.data[col]
                        order_ranges.append((float(wave.min()), float(wave.max())))

    if not order_ranges:
        raise ValueError("No valid orders detected (need a MULTISPE primary or 'wl'/'wave' columns).")
    print(f"  Auto-detected {len(order_ranges)} orders")
    return order_ranges


def read_orders_from_file(filename):
    print(f"Reading order ranges from: {filename}")
    order_ranges = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.strip('()')
            a, b = map(float, line.split(','))
            order_ranges.append((a, b))
    if not order_ranges:
        raise ValueError("No valid order ranges found in file")
    print(f"  Loaded {len(order_ranges)} order ranges")
    return order_ranges


def get_observer_location(args):
    """Priority: -loc 'lat,lon,height' > -site (astropy) > observed_spectrum header > TLS default."""
    if args.loc:
        try:
            lat, lon, h = map(float, args.loc.split(','))
        except ValueError:
            raise ValueError("'-loc' must be 'lat,lon,height' in deg,deg,m.")
        return EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=h * u.m)
    if args.site:
        try:
            return EarthLocation.of_site(args.site)
        except Exception:
            raise ValueError(f"Site '{args.site}' not recognised by Astropy.")
    if args.observed_spectrum:
        try:
            with fits.open(args.observed_spectrum, ignore_blank=True) as hdul:
                hdr = hdul[0].header
                lat = hdr.get('TEL GEOLAT') or hdr.get('GEOLAT') or hdr.get('LAT')
                lon = hdr.get('TEL GEOLON') or hdr.get('GEOLON') or hdr.get('LON')
                elev = hdr.get('TEL GEOELEV') or hdr.get('GEOELEV') or hdr.get('ELEV')
            if lat is not None and lon is not None and elev is not None:
                print(f"Using location from FITS header: lat={lat}, lon={lon}, elev={elev}")
                return EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        except Exception as e:
            print(f"Warning: could not extract location from FITS header: {e}")
    return EarthLocation(lat=TLS_SITE['lat'] * u.deg, lon=TLS_SITE['lon'] * u.deg, height=TLS_SITE['elev'] * u.m)


# =============================================================================
# --- IP kernels ---
# =============================================================================

def _gaussian_kernel(size, sigma):
    if sigma <= 0:
        k = np.zeros(size); k[size // 2] = 1.0; return k
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _bigaussian_kernel(half, sigma, asymmetry):
    if sigma <= 0:
        k = np.zeros(2 * half + 1); k[half] = 1.0; return k
    x = np.arange(-half, half + 1)
    sigma_l, sigma_r = sigma * (1 + asymmetry), sigma * (1 - asymmetry)
    k = np.where(x < 0, np.exp(-0.5 * (x / sigma_l) ** 2), np.exp(-0.5 * (x / sigma_r) ** 2))
    return k / k.sum()


def _voigt_kernel(half, sigma, gamma):
    if sigma <= 0 and gamma <= 0:
        k = np.zeros(2 * half + 1); k[half] = 1.0; return k
    sigma = max(sigma, 1e-6)
    x = np.linspace(-half, half, 2 * half + 1)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    k = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return k / k.sum()


def simulate_stellar_rv(times_s, i, rng=None, rv_amp_qp=2.0, prot_days=25.0,
                         tau_qp_days=20.0, tau_ou_hours=2.0, sigma_ou=1.5, noise_scale=1.0):
    """Quasi-periodic (rotation) + Ornstein-Uhlenbeck (granulation) stellar RV jitter [m/s]."""
    rng = np.random.default_rng() if rng is None else rng
    Prot_s, tau_qp_s = prot_days * 86400.0, tau_qp_days * 86400.0
    rv_qp = noise_scale * rv_amp_qp * np.exp(-(times_s[i] - times_s[0]) / tau_qp_s) * \
        np.cos(2 * np.pi * (times_s[i] - times_s[0]) / Prot_s)

    if i == 0:
        simulate_stellar_rv.rv_ou_prev = 0.0
        simulate_stellar_rv.last_time = times_s[0]
    dt = times_s[i] - simulate_stellar_rv.last_time
    a = np.exp(-dt / (tau_ou_hours * 3600.0))
    var = (noise_scale * sigma_ou) ** 2 * (1 - a * a)
    rv_ou = a * simulate_stellar_rv.rv_ou_prev + rng.normal(0.0, np.sqrt(var))
    simulate_stellar_rv.rv_ou_prev, simulate_stellar_rv.last_time = rv_ou, times_s[i]
    return rv_qp + rv_ou


def convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width,
                 ip_type='gaussian', asymmetry=0.0, gamma=0.0):
    """
    Multiplies stellar x FTS on a common uniform log-wavelength grid, convolves with
    the IP kernel, and interpolates back onto `shifted_wave`.

    No oversampling: the grid step equals the FINER of the stellar/FTS native
    sampling (needed to resolve narrow iodine lines without aliasing) -- never finer.

    width / gamma : IP sigma / Lorentzian width, in PIXELS of that internal grid
                     (VIPER's IP_hs convention), not km/s and not the output pixel grid.
    """
    wave_min = max(shifted_wave.min(), fts_wave.min())
    wave_max = min(shifted_wave.max(), fts_wave.max())
    if wave_max <= wave_min:
        raise ValueError("No overlapping wavelength region between stellar and FTS spectra.")

    dx = min(np.min(np.diff(np.log(shifted_wave))), np.min(np.diff(np.log(fts_wave))))
    half = max(1, int(np.ceil(5 * max(width, gamma, 1e-6))))  # kernel half-size, pixels
    buffer = half * dx
    log_grid = np.arange(np.log(wave_min) - buffer, np.log(wave_max) + buffer, dx)

    star_interp = interp1d(np.log(shifted_wave), star_flux, kind='linear', bounds_error=False, fill_value=1.0)
    fts_interp = interp1d(np.log(fts_wave), fts_flux, kind='linear', bounds_error=False, fill_value=1.0)
    model = star_interp(log_grid) * fts_interp(log_grid)

    if ip_type == 'gaussian':
        kernel = _gaussian_kernel(2 * half + 1, width)
    elif ip_type == 'bigaussian':
        kernel = _bigaussian_kernel(half, width, asymmetry)
    elif ip_type == 'voigt':
        kernel = _voigt_kernel(half, width, gamma)
    else:
        raise ValueError(f"Invalid ip_type: '{ip_type}'.")

    model_conv = fftconvolve(model, kernel, mode='same')
    back_interp = interp1d(log_grid, model_conv, kind='linear', bounds_error=False, fill_value=1.0)
    return back_interp(np.log(shifted_wave))


def apply_rv_shift(wavelengths, rv_m_s):
    return wavelengths * (1 + rv_m_s / SPEED_OF_LIGHT)


def compute_order_indices(rest_wave, order_ranges, target_pixels=2048):
    """Computes, ONCE, the native-array index set selected for each order, using the
    REST-FRAME (unshifted) wavelength grid only. This index set (mask + decimation
    step + resulting indices) is fixed and identical for every exposure -- it must
    NOT be recomputed per-observation on an RV/BERV-shifted wave array, since that
    would let the per-exposure Doppler shift change which native pixel indices land
    inside an order's boundary and at what decimation phase, producing a spurious
    sub-pixel, BERV-correlated wobble in the sliced grid (and downstream wavelength-
    solution coefficients) even though no physical effect should cause one.

    Same selection logic as before (boundary mask, then integer decimation toward
    ~target_pixels, never interpolating/inventing points, never upsampling a sparser
    order) -- only WHEN it is computed changes, not how."""
    order_indices = []
    for start, end in order_ranges:
        idx = np.where((rest_wave >= start) & (rest_wave <= end))[0]
        if len(idx) < 2:
            order_indices.append(np.empty(0, dtype=int))
            continue
        step = max(1, len(idx) // target_pixels)
        order_indices.append(idx[::step])
    print(f"Computed {len(order_indices)} order index sets from rest-frame grid "
          f"(~{target_pixels} px target, decimation only, fixed across exposures).")
    return order_indices


def slice_orders(wave, flux, order_indices):
    """Applies a FIXED, precomputed set of native-array indices (one array per order,
    from compute_order_indices) to this exposure's wave/flux. Pure indexing -- no
    masking, no re-stepping, no interpolation -- so every exposure samples exactly
    the same native pixels per order, regardless of that exposure's RV/BERV shift."""
    wave_orders, flux_orders = [], []
    for idx in order_indices:
        if len(idx) == 0:
            wave_orders.append(np.empty(0))
            flux_orders.append(np.empty(0))
            continue
        wave_orders.append(wave[idx])
        flux_orders.append(flux[idx])
    return wave_orders, flux_orders


def write_default_fits(filename, wave_orders, flux_orders, date_obs, location):
    print(f"Writing FITS file: {filename}")
    hdr = fits.Header()
    hdr.set('DATE-OBS', date_obs)
    hdr.set('RA', 86.819720 / 15.0, 'Right ascension (hours)')
    hdr.set('DEC', -51.06714)
    hdr.set('EXP', 30)
    hdr.set('TEL GEOLAT', location.lat.to(u.deg).value)
    hdr.set('TEL GEOLON', location.lon.to(u.deg).value)
    hdr.set('TEL GEOELEV', location.height.to(u.m).value)

    hdulist = [fits.PrimaryHDU(header=hdr)]
    for idx, (w, f) in enumerate(zip(wave_orders, flux_orders)):
        if len(w) == 0:
            w, f = np.array([0.0]), np.array([1.0])
        # c1 = fits.Column(name='wave', array=w, format='D')
        # c2 = fits.Column(name='flux', array=f, format='D')
        c1 = fits.Column(name='wave', array=np.asarray(w, dtype=np.float64), format='D')
        c2 = fits.Column(name='flux', array=np.asarray(f, dtype=np.float64), format='D')
        hdu = fits.BinTableHDU.from_columns([c1, c2])
        hdu.name = f'ORDER_{idx}'
        hdulist.append(hdu)
    fits.HDUList(hdulist).writeto(filename, overwrite=True)


def parse_ip_width(ip_width_arg, num_obs):
    """Float -> constant; [low, high] -> per-observation uniform draw."""
    try:
        val = ast.literal_eval(ip_width_arg)
    except Exception:
        val = float(ip_width_arg)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return list(np.random.uniform(float(val[0]), float(val[1]), num_obs))
    if isinstance(val, (int, float)):
        return [float(val)] * num_obs
    raise ValueError(f"Invalid format for ip_width: {ip_width_arg}")


# =============================================================================
# --- CLI ---
# =============================================================================

parser = argparse.ArgumentParser(description="Generate simulated VIPER FITS files with synthetic RV shifts and IP convolution.")
parser.add_argument('-mode', choices=['default', 'auto', 'manual'], default='default',
                     help="'default' (CSV + hardcoded orders), 'auto' (infer orders+location from -observed_spectrum), 'manual' (orders from -orders_file)")
parser.add_argument('-observed_spectrum', default=None, help="FITS file used for order/location metadata (auto/manual modes). Never a flux source.")
parser.add_argument('-fts_file', default=None, help="Path to FTS iodine cell FITS file. Defaults to FTS_default_TLS.fits.")
parser.add_argument('-orders_file', default=None, help="Text file of order wavelength ranges (manual mode).")
parser.add_argument('-num_obs', type=int, default=10, help="Number of observations to generate.")
parser.add_argument('-vel_list', default=None, help="List of RVs (m/s) for each observation, e.g. '[100,200,300]'.")
parser.add_argument('-date_list', default=None, help="List of ISO observation dates, e.g. \"['2025-02-06T00:00:00']\".")
parser.add_argument('-time_step', default='5d0h', help="Spacing between observations, e.g. '3d0h'.")
parser.add_argument('-template', action='store_true', help="Also write a clean stellar template FITS file.")
parser.add_argument('-file', default='spectrum.csv', help="Synthetic spectrum CSV (always the stellar flux source).")
parser.add_argument('-ip_width', default="1.2", help="IP sigma in pixels of the internal convolution grid; float or '[low,high]'.")
parser.add_argument('-ip_type', choices=['gaussian', 'bigaussian', 'voigt'], default='gaussian')
parser.add_argument('-asymmetry', type=float, default=0.0, help="Asymmetry (-1 to 1) for bi-Gaussian IP.")
parser.add_argument('-gamma', type=float, default=0.0, help="Voigt Lorentzian width, same pixel units as -ip_width.")
parser.add_argument('-site', default=None, help="Observatory name recognised by astropy (e.g. 'Keck').")
parser.add_argument('-loc', default=None, help="Manual 'lat,lon,height' in deg,deg,m. Overrides -site.")
parser.add_argument('-add_noise', action='store_true', help="Add synthetic stellar RV activity jitter.")
args = parser.parse_args()

# --- Load spectra (stellar flux is always the CSV; FTS defaults unless overridden) ---
star_wave, star_flux = read_spectrum_csv(args.file)
fts_wave, fts_flux = read_FTS_fits()

# --- Resolve order ranges + FTS override, once, per mode ---
print(f"\n=== MODE: {args.mode.upper()} ===")
if args.mode == 'default':
    order_ranges, order_source = order_wavs, 'hardcoded default'

elif args.mode == 'auto':
    if not args.observed_spectrum:
        raise ValueError("Auto mode requires -observed_spectrum.")
    order_ranges, order_source = auto_detect_orders_from_fits(args.observed_spectrum), 'auto-detected'
    if args.fts_file:
        fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, infer_fts_format(args.fts_file))

elif args.mode == 'manual':
    if not args.orders_file:
        raise ValueError("Manual mode requires -orders_file.")
    order_ranges, order_source = read_orders_from_file(args.orders_file), 'user-supplied'
    if args.fts_file:
        fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, infer_fts_format(args.fts_file))

# --- Keep only orders that overlap the FTS wavelength range ---
fts_min, fts_max = fts_wave.min(), fts_wave.max()
order_ranges = [(max(mn, fts_min), min(mx, fts_max)) for mn, mx in order_ranges if mx >= fts_min and mn <= fts_max]
if not order_ranges:
    raise RuntimeError(f"No orders overlap the FTS wavelength range ({fts_min:.1f}-{fts_max:.1f} Å).")
print(f"Using {len(order_ranges)} {order_source} orders overlapping the FTS range.")

# --- Order index sets: computed ONCE on the REST-FRAME (unshifted) star_wave grid,
#     then reused identically for every exposure. This is what keeps the sliced
#     native-pixel grid fixed across exposures regardless of each exposure's RV/BERV
#     shift -- see compute_order_indices() docstring for why this must happen here,
#     before any per-observation shift exists, rather than per-observation later. ---
order_indices = compute_order_indices(star_wave, order_ranges)

observer_location = get_observer_location(args)

# --- Observation timestamps (finalizes NUM_OBS, since -date_list can override -num_obs) ---
NUM_OBS = args.num_obs
if args.date_list:
    date_list = ast.literal_eval(args.date_list)
    obs_times = [datetime.fromisoformat(d) for d in date_list]
    NUM_OBS = len(obs_times)
    print(f"Using explicit date_list with {NUM_OBS} observations.")
else:
    m = re.match(r'(\d+)d(\d+)h', args.time_step)
    if not m:
        raise ValueError("Invalid -time_step format. Use 'NdMh', e.g. '3d0h'.")
    time_delta = timedelta(days=int(m.group(1)), hours=int(m.group(2)))
    base_time = datetime(2025, 2, 6, 0, 0, 0)
    obs_times = [base_time + i * time_delta for i in range(NUM_OBS)]

# --- Everything sized to NUM_OBS is derived only now, after NUM_OBS is final ---
rv_values = ast.literal_eval(args.vel_list) if args.vel_list else list(np.linspace(-1000, 1000, NUM_OBS))
if len(rv_values) != NUM_OBS:
    raise ValueError(f"vel_list length ({len(rv_values)}) != num_obs ({NUM_OBS}).")

ip_widths = parse_ip_width(str(args.ip_width), NUM_OBS)
np.savetxt("ip_widths.txt", ip_widths, fmt="%.6f")

t0 = obs_times[0]
times_s = np.array([(t - t0).total_seconds() for t in obs_times])
print("RV values used (m/s):", rv_values)

# =============================================================================
# --- Main loop: simulated observations ---
#
# Per-observation pipeline order is physically fixed and must NOT change:
#   1) combine all RV components (science + activity - barycentric)
#   2) apply that combined RV as a wavelength shift to the stellar spectrum
#   3) convolve shifted-star x FTS with the IP   (needs the correct relative
#      Doppler offset between star and FTS lines to be physically meaningful)
#   4) slice into echelle orders
#   5) write FITS
# Steps 3-5 ("everything else") are identical in structure for every file;
# only the RV combination (step 1) and the resulting shift (step 2) differ
# per observation, and that combination is computed as its own explicit,
# isolated step immediately before it is used -- nothing upstream of it
# depends on its value.
# =============================================================================

barycorr_values = []
print(f"\n=== GENERATING {NUM_OBS} OBSERVATIONS ===")
for i in range(NUM_OBS):
    print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")

    # --- everything that does NOT depend on the combined RV: identical in
    #     kind for every file, computed first ---
    obstime = Time(obs_times[i], scale='utc')
    barycorr_ms = TARGET.radial_velocity_correction(
        obstime=obstime, location=observer_location, kind='barycentric').to(u.m / u.s).value
    barycorr_values.append(barycorr_ms)
    rv_activity_ms = simulate_stellar_rv(times_s, i) if args.add_noise else 0.0
    rv_science_ms = rv_values[i]

    # -------------------------------------------------------------------
    # --- RV combination: isolated, explicit, last thing computed before
    #     it is used. Signs match the physical convention used throughout:
    #     science RV + stellar activity jitter - barycentric correction. ---
    # -------------------------------------------------------------------
    total_rv_ms = rv_science_ms + rv_activity_ms - barycorr_ms
    print(f" RV components -> science: {rv_science_ms:.2f} m/s, "
          f"activity: {rv_activity_ms:.2f} m/s, barycentric: {barycorr_ms:.2f} m/s")
    print(f" Combined RV applied as shift: {total_rv_ms:.2f} m/s")

    # --- shift -> convolve -> slice -> write (fixed physical order) ---
    shifted_wave = apply_rv_shift(star_wave, total_rv_ms)
    flux_conv = convolve_IP(shifted_wave, star_flux, fts_wave, fts_flux, width=ip_widths[i],
                             ip_type=args.ip_type, asymmetry=args.asymmetry, gamma=args.gamma)
    wave_orders, flux_orders = slice_orders(shifted_wave, flux_conv, order_indices)

    output_file = OUTPUT_TEMPLATE.format(i + 1)
    write_default_fits(output_file, wave_orders, flux_orders,
                        obs_times[i].isoformat(timespec='milliseconds'), observer_location)
    print(f" -> {output_file}")

# =============================================================================
# --- Optional template: clean, barycentric-corrected stellar spectrum only
#     (no IP convolution, no FTS multiplication -- a "ground truth" template) ---
# =============================================================================

if args.template:
    print("\n=== GENERATING TEMPLATE ===")

    # --- everything that does NOT depend on the combined RV: identical in
    #     kind to the main loop, computed first ---
    obstime_tpl = Time(obs_times[0], scale='utc')
    barycorr_ms_tpl = TARGET.radial_velocity_correction(
        obstime=obstime_tpl, location=observer_location, kind='barycentric').to(u.m / u.s).value

    # -------------------------------------------------------------------
    # --- RV combination: isolated, explicit, last thing computed before
    #     it is used. The template has no science RV and no activity jitter
    #     (it's the clean, barycentric-corrected "ground truth" spectrum),
    #     so its only RV component is the barycentric correction, with the
    #     same sign convention as the main loop (... - barycorr). ---
    # -------------------------------------------------------------------
    total_rv_ms_tpl = -barycorr_ms_tpl
    print(f" RV components -> science: 0.00 m/s, activity: 0.00 m/s, "
          f"barycentric: {barycorr_ms_tpl:.2f} m/s")
    print(f" Combined RV applied as shift: {total_rv_ms_tpl:.2f} m/s")

    # --- shift -> slice -> write (no IP convolution / no FTS by design) ---
    wave_tpl = apply_rv_shift(star_wave, total_rv_ms_tpl)
    w_orders_tpl, f_orders_tpl = slice_orders(wave_tpl, star_flux, order_indices)
    write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl,
                        obs_times[0].isoformat(timespec='milliseconds'), observer_location)
    print(f" -> {OUTPUT_TEMPLATE_FILE}")

print(f"\nGenerated {NUM_OBS} observation file(s){' + 1 template' if args.template else ''}.")
print(f"Mode: {args.mode} | Order ranges: {order_source} ({len(order_ranges)} orders)")

with open('barycorr_values.txt', 'w') as f:
    for v in barycorr_values:
        f.write(f"{v:.6f}\n")
print(f"Saved {len(barycorr_values)} barycentric correction values to barycorr_values.txt")