#!/usr/bin/env python3
"""
generate_spectrum_ATLAS.py

Generating a high-resolution stellar spectrum with BasicATLAS & SYNTHE.

• Defaults to solar parameters (Teff=5770, logg=4.44, Y solar, [M/H]=0, vmic=2 km/s).
• Caches ODFs in ~/BasicATLAS_ODFs/ by parameter set to avoid multi-hour rebuilds.
• Adds microturbulence control via --vmic.
• Optionally flatten the continuum to exactly 1 with --flatten.

Usage:
  # Basic (solar default, non-normalized continuum):
  python generate_spectrum_ATLAS.py

  # Custom (e.g. Teff=6000, logg=4.0, [M/H]=-0.5, vmic=1, C+0.2 dex, flattened):
  python generate_spectrum_ATLAS.py \
    --teff 6000 --logg 3.8 --zscale -0.5 --vmic 1 \
    --enhancements C=0.2 \
    --num-bins 80000 \
    --flatten \
    --out-spectrum custom_spectrum.csv \
    --out-plot custom_spectrum.png
"""

import os
import sys
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# required for --flatten
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
from scipy.signal import medfilt

# 1) Where BasicATLAS lives
atlas_dir = os.path.expanduser('~/BasicATLAS/')
sys.path.insert(0, atlas_dir)
import atlas

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic spectrum with BasicATLAS & SYNTHE"
    )
    p.add_argument('--teff',   type=float, default=5770, help='Teff [K]')
    p.add_argument('--logg',   type=float, default=4.44, help='log g [cgs]')
    p.add_argument('--Y',      type=float, default=-0.1,
                   help='Helium mass fraction (-0.1 → solar)')
    p.add_argument('--zscale', type=float, default=0.0,
                   help='Overall metallicity [M/H]')
    p.add_argument('--vmic',   type=float, default=2,
                   help='Microturbulence [km/s]')
    p.add_argument('--enhancements', type=str, default='',
                   help='Comma-separated ELEMENT=delta_dex list')
    p.add_argument('--wave-start', type=float, default=3800, help='Start λ [Å]')
    p.add_argument('--wave-end',   type=float, default=7500, help='End   λ [Å]')
    p.add_argument('--num-bins',   type=int,   default=68000,
                   help='Number of output wavelength points')
    p.add_argument('--generate-odf', action='store_true',
                   help='Always build ODFs even if cached')
    p.add_argument('--flatten', action='store_true', default=False,
                   help='Divide out an envelope so the continuum sits exactly at 1')
    p.add_argument('--out-spectrum', type=str, default='spectrum.csv',
                   help='Output spectrum file (.csv or .fits)')
    p.add_argument('--out-plot',     type=str, default='spectrum.png',
                   help='PNG plot filename')
    return p.parse_args()

def parse_enhancements(s):
    enh = {}
    if not s.strip():
        return enh
    for kv in s.split(','):
        k, v = kv.split('=', 1)
        enh[k.strip()] = float(v)
    return enh

def odf_cache_path(zscale, enhancements):
    parts = [f"z{int(zscale*100)}"]
    for k in sorted(enhancements):
        parts.append(f"{k}{int(enhancements[k]*100)}")
    name = "_".join(parts) or "solar"
    base = os.path.expanduser('~/BasicATLAS_ODFs/')
    return os.path.join(base, name)

def main():
    args = parse_args()

    # 2) BasicATLAS Settings
    settings = atlas.Settings()
    settings.teff   = args.teff
    settings.logg   = args.logg
    settings.Y      = args.Y
    settings.zscale = args.zscale
    settings.vturb  = int(args.vmic)

    # 3) Apply enhancements
    enh = parse_enhancements(args.enhancements)
    if enh:
        settings.abun = enh

    # 4) ODF caching logic
    solar_case = (abs(args.zscale) < 1e-6) and not enh
    need_odf = args.generate_odf or (not solar_case)
    odf_dir = None
    
    if need_odf:
        odf_dir = odf_cache_path(args.zscale, enh)
        valid_odf = False
    
        if os.path.isdir(odf_dir):
            contents = os.listdir(odf_dir)
            valid_odf = any(fname.endswith(".dat") for fname in contents)
            if not valid_odf:
                print(f"ODF dir exists but is invalid → removing and rebuilding: {odf_dir}")
                shutil.rmtree(odf_dir)
        else:
            print(f"ODF dir does not exist → creating: {odf_dir}")
    
        if not valid_odf or args.generate_odf:
            print("Building ODFs with DFSYNTHE (may take hours)...")
            atlas.dfsynthe(odf_dir, settings)
        else:
            print(f"Using cached ODFs: {odf_dir}")
    else:
        print("Using default solar ODFs (no ODF build).")


    # 5) Run ATLAS-9
    model_dir = os.path.expanduser('~/ATLAS_model')
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    print("Running ATLAS-9…")
    if odf_dir:
        atlas.atlas(model_dir, settings, ODF=odf_dir)
    else:
        atlas.atlas(model_dir, settings)

    # 6) Run SYNTHE (nm)
    wl_min_nm = args.wave_start / 10.0
    wl_max_nm = args.wave_end   / 10.0
    print(f"Running SYNTHE from {wl_min_nm:.1f} to {wl_max_nm:.1f} nm…")
    atlas.synthe(model_dir, wl_min_nm, wl_max_nm)

    # 7) Read synthetic spectrum
    spec = atlas.read_spectrum(model_dir, num_bins=args.num_bins)
    wl, flux = spec['wl'], spec['flux']

    # 8) Trim & interpolate
    mask = (wl >= args.wave_start) & (wl <= args.wave_end)
    if not mask.any():
        sys.exit("No data in requested λ-range; check wave-start/end.")
    wl_trim, flux_trim = wl[mask], flux[mask]
    wl_uni  = np.linspace(args.wave_start, args.wave_end, args.num_bins)
    flux_uni = np.interp(wl_uni, wl_trim, flux_trim)

    # 9) Continuum normalization or flatten
    if args.flatten:
        # 9a) Estimate envelope using sliding-window maximum and median filtering
        # Determine approximate resolution element size
        dlam = np.median(np.diff(wl_uni))
        lam_mid = 0.5 * (args.wave_start + args.wave_end)
        R = lam_mid / dlam
        fwhm = lam_mid / R

        # Window size: ~50 resolution elements
        window_pix = int((50 * fwhm) / dlam)
        if window_pix < 5:
            window_pix = 5
        if window_pix % 2 == 0:
            window_pix += 1

        # 1) Maximum filter to estimate upper envelope
        env_max = maximum_filter1d(flux_uni, size=window_pix, mode='reflect')
        # 2) Median filter to smooth small-scale spikes in the envelope estimate
        env_med = medfilt(env_max, kernel_size=window_pix)
        # 3) Savitzky-Golay smoothing for a smooth continuum
        # Use a smaller polynomial window but odd length
        sg_window = max(5, (window_pix // 4) | 1)
        env_smooth = savgol_filter(env_med, window_length=sg_window,
                                   polyorder=2, mode='mirror')
        # Prevent zeros
        env_smooth[env_smooth <= 0] = np.nanmin(env_smooth[env_smooth > 0])

        # Normalize flux by continuum
        flux_norm = flux_uni / env_smooth
        # Final renormalization: divide by median to center continuum at 1
        medval = np.nanmedian(flux_norm)
        flux_norm /= medval
    else:
        # No normalization
        flux_norm = flux_uni

    # 10) Save spectrum
    df = pd.DataFrame({'wavelength': wl_uni, 'flux': flux_norm})
    out_spec = args.out_spectrum
    if out_spec.lower().endswith('.fits'):
        from astropy.io import fits
        hdu = fits.BinTableHDU(df.to_records(index=False))
        hdu.writeto(out_spec, overwrite=True)
    else:
        df.to_csv(out_spec, index=False)
    print(f"Saved spectrum → {out_spec}")

    # 11) Plot & save
    plt.figure(figsize=(10,4))
    plt.plot(wl_uni, flux_norm, 'k-', lw=0.6)
    ttl = (f"Teff={args.teff}, logg={args.logg}, [M/H]={args.zscale}, "
           f"vmic={args.vmic}")
    if args.flatten:
        ttl += "  (flattened)"
    plt.title(ttl)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Normalized Flux')
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    print(f"Saved plot → {args.out_plot}")

if __name__ == '__main__':
    main()
