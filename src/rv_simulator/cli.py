#!/usr/bin/env python3
"""
Command-line interface for RV Simulator.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
# from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
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

# Set ephemeris
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('de430')

from .core.doppler import apply_rv_shift, simulate_stellar_rv
from .core.convolution import create_instrument_grids, convolve_IP_on_uniform_grid
from .io.spectrum_reader import read_spectrum_csv, auto_detect_orders_from_fits, read_orders_from_file
from .io.fits_handler import read_FTS_fits, read_FTS_fits_auto, infer_fts_format, write_default_fits
from .utils.constants import DEFAULT_ORDER_RANGES, OUTPUT_TEMPLATE, OUTPUT_TEMPLATE_FILE, VIPER_IP_HALF_SIZE
from .utils.location import get_observer_location, extract_location_from_fits
from .utils.helpers import (parse_ip_width, parse_time_step, parse_date_list, filter_overlapping_orders)
from scipy.interpolate import interp1d


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Run simulation
    run_simulation(args)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate simulated VIPER FITS files with synthetic RV shifts and IP convolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    )
    
    # Mode selection
    parser.add_argument('-mode', type=str, choices=['default', 'auto', 'manual'], default='default', help="Input mode: 'default' (synthetic CSV + hardcoded), ""'auto' (auto-infer from files), 'manual' (user-provided text format)")
    
    # Input files
    parser.add_argument('-file', type=str, default='spectrum.csv', help="Input CSV file for synthetic spectrum.")
    parser.add_argument('-observed_spectrum', type=str, default=None, help="Path to observed spectrum FITS file (for auto/manual modes)")
    parser.add_argument('-fts_file', type=str, default=None, help="Path to FTS cell FITS file (for auto/manual modes)")
    parser.add_argument('-orders_file', type=str, default=None, help="Path to text file with order wavelength ranges (for manual mode)")
    
    # Observation parameters
    parser.add_argument('-num_obs', type=int, default=10, help="Number of observations to generate.")
    parser.add_argument('-vel_list', type=str, default=None, help="List of RVs (m/s) for each observation.")
    parser.add_argument('-date_list', type=str, default=None, help="List of observation dates as ISO strings.")
    parser.add_argument('-time_step', type=str, default='5d0h', help="Time spacing between observations (e.g., '3d0h' for 3 days and 0 hours).")
    
    # Instrumental profile
    parser.add_argument('-ip_width', type=str, default="1.2", help="Gaussian IP width (sigma) for convolution; can be a float or a list like [1.8, 2.2].")
    parser.add_argument('-ip_type', type=str, default='gaussian', choices=['gaussian', 'bigaussian', 'voigt'], help="Type of instrumental profile to convolve with.")
    parser.add_argument('-asymmetry', type=float, default=0.0, help="Asymmetry factor (-1 to 1) for bi-Gaussian IP.")
    parser.add_argument('-gamma', type=float, default=0.0, help="Lorentzian width (gamma) for Voigt profile convolution.")
    
    # Observatory location
    parser.add_argument('-site', type=str, default=None, help="Observatory name recognized by astropy (e.g. 'Keck').")
    parser.add_argument('-loc', type=str, default=None, help="Manual 'lat,lon,height' in deg,deg,m. Overrides -site.")
    
    # Additional options
    parser.add_argument('-template', action='store_true', help="Flag to create a template file.")
    parser.add_argument('-add_noise', action='store_true', help="If set, add synthetic stellar RV noise.")
    
    return parser


def run_simulation(args):
    """Run the simulation based on parsed arguments."""
    
    print(f"\n{'='*60}")
    print(f"RV SIMULATOR - MODE: {args.mode.upper()}")
    print(f"{'='*60}\n")
    
    # Parse number of observations and RV values
    NUM_OBS = args.num_obs
    
    # Read synthetic stellar spectrum (always needed)
    print("Loading synthetic stellar spectrum...")
    star_wave, star_flux = read_spectrum_csv(args.file)
    
    # Determine order ranges based on mode
    if args.mode == 'default':
        print("Using default mode: CSV spectrum + hardcoded order ranges")
        order_ranges = DEFAULT_ORDER_RANGES
        fts_wave, fts_flux = read_FTS_fits()
    
    elif args.mode == 'auto':
        if not args.observed_spectrum:
            raise ValueError("Auto mode requires -observed_spectrum argument")
        
        print("Using auto mode: inferring parameters from FITS files")
        order_ranges = auto_detect_orders_from_fits(args.observed_spectrum)
        
        if args.fts_file:
            fmt = infer_fts_format(args.fts_file)
            fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, fmt)
        else:
            fts_wave, fts_flux = read_FTS_fits()
    
    elif args.mode == 'manual':
        if not args.orders_file:
            raise ValueError("Manual mode requires -orders_file argument")
        
        print("Using manual mode: user-provided order ranges from text file")
        order_ranges = read_orders_from_file(args.orders_file)
        
        if args.fts_file:
            fmt = infer_fts_format(args.fts_file)
            fts_wave, fts_flux = read_FTS_fits_auto(args.fts_file, fmt)
        else:
            fts_wave, fts_flux = read_FTS_fits()
    
    # Filter orders to overlap with FTS
    order_ranges = filter_overlapping_orders(order_ranges, fts_wave)
    
    # Parse RV values
    if args.vel_list:
        rv_values = ast.literal_eval(args.vel_list)
    else:
        rv_values = np.linspace(-1000, 1000, NUM_OBS).tolist()
    
    if len(rv_values) != NUM_OBS:
        raise ValueError(
            f"Length of velocity list ({len(rv_values)}) does not match "
            f"number of observations ({NUM_OBS})."
        )
    
    print(f"RV values used (m/s): {rv_values}")
    
    # Parse observation times
    if args.date_list:
        obs_times = parse_date_list(args.date_list)
        NUM_OBS = len(obs_times)
        print(f"Using explicit date_list with {NUM_OBS} observations.")
    else:
        time_delta = parse_time_step(args.time_step)
        base_time = datetime(2025, 2, 6, 0, 0, 0)
        obs_times = [base_time + i * time_delta for i in range(NUM_OBS)]
    
    # Get observer location
    observer_location = get_observer_location(site=args.site, loc=args.loc)
    
    # Try to extract location from FITS if provided
    if args.mode in ['auto', 'manual'] and args.observed_spectrum:
        fits_location = extract_location_from_fits(args.observed_spectrum)
        if fits_location is not None:
            observer_location = fits_location
    
    # Parse IP widths
    ip_widths = parse_ip_width(str(args.ip_width), NUM_OBS)
    np.savetxt("ip_widths.txt", ip_widths, fmt="%.6f")
    print(f"IP widths saved to ip_widths.txt")
    
    # Create fixed instrument grids
    instrument_grids = create_instrument_grids(order_ranges, oversample_factor=3)
    
    if not instrument_grids:
        raise RuntimeError("Failed to create any valid instrument grids from order_ranges.")
    
    # Pre-interpolate FTS spectrum
    print("Creating FTS interpolator...")
    fts_interp = interp1d(fts_wave, fts_flux, kind='linear',
                          bounds_error=False, fill_value=1.0)
    
    # Main simulation loop
    print(f"\n{'='*60}")
    print(f"GENERATING {NUM_OBS} OBSERVATIONS")
    print(f"{'='*60}\n")
    
    t0 = obs_times
    times_s = np.array([(ot - t0).total_seconds() for ot in obs_times])
    barycorr_values = []
    
    # Open diagnostic file
    with open("barycorr_flux_check.txt", "w") as diag_file:
        diag_file.write("# idx barycorr_ms flux_median_order0\n")
        
        for i in range(NUM_OBS):
            print(f"\n--- Simulation {i+1}/{NUM_OBS} ---")
            
            obs_time = obs_times[i]
            obstime = Time(obs_time, scale='utc')
            target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
            
            # Calculate barycentric correction
            barycorr = target.radial_velocity_correction(
                obstime=obstime,
                location=observer_location,
                kind='barycentric'
            )
            barycorr_ms = barycorr.to(u.m/u.s).value
            barycorr_values.append(barycorr_ms)
            
            # Get total RV
            total_rv = rv_values[i]
            
            # Optional stellar activity noise
            if args.add_noise:
                rv_activity = simulate_stellar_rv(times_s, i)
                print(f"Stellar activity RV: {rv_activity:.2f} m/s")
            else:
                rv_activity = 0.0
            
            # Apply total shift: user RV + activity - barycentric correction
            total_shift = total_rv + rv_activity - barycorr_ms
            shifted_star_wave = apply_rv_shift(star_wave, total_shift)
            
            print(f"Total RV: {total_rv:.2f} m/s, Barycentric: {barycorr_ms:.2f} m/s")
            print(f"Final shift applied: {total_shift:.2f} m/s")
            
            # Create interpolator for shifted stellar spectrum
            star_interp = interp1d(shifted_star_wave, star_flux, kind='linear', bounds_error=False, fill_value=1.0)
            
            wave_orders_final = []
            flux_orders_final = []
            
            # Process each order
            for k, (wave_k, d_log_wave_k) in enumerate(instrument_grids):
                # Interpolate star and FTS onto fixed instrument grid
                star_flux_k = star_interp(wave_k)
                fts_flux_k = fts_interp(wave_k)
                
                # Multiply
                model_k = star_flux_k * fts_flux_k
                
                # Convolve on uniform grid
                flux_conv_k = convolve_IP_on_uniform_grid(model_k, d_log_wave_k, ip_widths[i], ip_type=args.ip_type, asymmetry=args.asymmetry, gamma=args.gamma)
                
                # Trim wavelength grid to match convolution output
                wave_k_eff = wave_k[VIPER_IP_HALF_SIZE : -VIPER_IP_HALF_SIZE]
                
                wave_orders_final.append(wave_k_eff)
                flux_orders_final.append(flux_conv_k)
            
            # Write science FITS
            date_obs_str = obs_time.isoformat(timespec='milliseconds')
            output_file = OUTPUT_TEMPLATE.format(i+1)
            write_default_fits(output_file, wave_orders_final, flux_orders_final, date_obs_str)
            print(f"-> {output_file}")
            
            # Diagnostic output
            flux_median = np.median(flux_orders_final)
            debug_line = f"{i} {barycorr_ms:.3f} {flux_median:.6e}"
            diag_file.write(debug_line + "\n")
    
    print(f"\n{'='*60}")
    print("Simulation complete!")
    print(f"{'='*60}\n")
    
    # Generate template if requested
    if args.template:
        print(f"\n{'='*60}")
        print("GENERATING TEMPLATE")
        print(f"{'='*60}\n")
        
        first_obs_time = obs_times
        obstime_tpl = Time(first_obs_time, scale='utc')
        target = SkyCoord(ra=86.819720/15.0 * u.hourangle, dec=-51.06714 * u.deg)
        
        barycorr_tpl = target.radial_velocity_correction(obstime=obstime_tpl, location=observer_location, kind='barycentric')
        barycorr_ms_tpl = barycorr_tpl.to(u.m/u.s).value
        
        # Apply barycentric correction
        wave_tpl_corrected = apply_rv_shift(star_wave, -barycorr_ms_tpl)
        
        # Create interpolator for the bary-corrected star
        star_interp_tpl = interp1d(wave_tpl_corrected, star_flux, kind='linear', bounds_error=False, fill_value=1.0)
        
        w_orders_tpl = []
        f_orders_tpl = []
        
        # Interpolate onto the fixed instrument grids
        for k, (wave_k, d_log_wave_k) in enumerate(instrument_grids):
            flux_k = star_interp_tpl(wave_k)
            w_orders_tpl.append(wave_k)
            f_orders_tpl.append(flux_k)
        
        write_default_fits(OUTPUT_TEMPLATE_FILE, w_orders_tpl, f_orders_tpl, first_obs_time.isoformat(timespec='milliseconds'))
        print(f"-> {OUTPUT_TEMPLATE_FILE}")
        print("\nTemplate generation complete!")


if __name__ == "__main__":
    main()
