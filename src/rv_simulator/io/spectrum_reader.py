"""
Spectrum reading functionality for CSV and FITS files
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from ..utils.constants import DEFAULT_FTS_FILE

def read_spectrum_csv(file):
    """Reads a synthetic spectrum from a CSV file.

    Returns (wave, flux) as numpy arrays (float).
    Accepts columns 'wave' or 'wavelength' (case-insensitive) and 'flux'.
    """
    data = pd.read_csv(file)
    cols_lower = [c.lower() for c in data.columns]
    colmap = dict(zip(cols_lower, data.columns))

    try:
        wave = data[colmap.get('wave', colmap.get('wavelength'))].values
        flux = data[colmap.get('flux', 'flux')].values
    except Exception:
        raise KeyError(
            "CSV must contain 'wave' or 'wavelength' and 'flux' columns. "
            f"Found: {data.columns.tolist()}"
        )

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    print(f"[io] Loaded synthetic spectrum with {len(wave)} points from '{file}'.")
    return wave, flux

def read_FTS_fits(fts_file=None):
    """Read a simple table-style FTS FITS containing 'wave' and 'flux' columns.

    Returns (wave, flux) as numpy arrays. Raises informative errors if missing.
    """
    fts_file = fts_file or DEFAULT_FTS_FILE
    if not os.path.exists(fts_file):
        raise FileNotFoundError(f"FTS file not found: {fts_file}")

    with fits.open(fts_file, ignore_blank=True, output_verify='silentfix') as hdul:
        for i, hdu in enumerate(hdul):
            # table HDU with named columns
            if hasattr(hdu, 'columns') and hdu.data is not None:
                names = hdu.columns.names
                lower_names = [n.lower() for n in names]
                if 'wave' in lower_names and 'flux' in lower_names:
                    wave_col = names[lower_names.index('wave')]
                    flux_col = names[lower_names.index('flux')]
                    wave = np.asarray(hdu.data[wave_col], dtype=float)
                    flux = np.asarray(hdu.data[flux_col], dtype=float)
                    print(f"[io] Loaded FTS iodine spectrum with {len(wave)} points from HDU {i}.")
                    return wave, flux

    raise KeyError("FTS FITS file must contain 'wave' and 'flux' columns in at least one HDU.")

def read_FTS_fits_auto(fts_file=None):
    """More defensive reader — attempts common conversions (wavenumber -> wavelength, nm->Å).

    Returns (wave_A, flux).
    """
    fts_file = fts_file or DEFAULT_FTS_FILE
    if not os.path.exists(fts_file):
        raise FileNotFoundError(f"FTS file not found: {fts_file}")

    with fits.open(fts_file, ignore_blank=True, output_verify='silentfix') as hdul:
        hdr0 = hdul[0].header if len(hdul) > 0 else {}
        wavetype = str(hdr0.get('WAVETYPE', hdr0.get('wavetype', ''))).lower()
        unit = str(hdr0.get('BUNIT', hdr0.get('UNIT', ''))).lower()

        w, f = None, None
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'data') and hdu.data is not None and hasattr(hdu.data, 'names'):
                names = [n.lower() for n in hdu.data.names]
                if 'wave' in names and 'flux' in names:
                    w = np.asarray(hdu.data[hdu.data.names[names.index('wave')]], dtype=float)
                    f = np.asarray(hdu.data[hdu.data.names[names.index('flux')]], dtype=float)
                    break

        if w is None:
            raise ValueError(f"FTS file '{fts_file}' has no 'wave'/'flux' table columns.")

    # handle common conversions (wavenumber cm^-1 -> Å)
    if 'wavenumber' in wavetype or 'wavenumber' in unit or any('cm-1' in str(v).lower() for v in (wavetype, unit)):
        # assume wavenumber in cm^-1: lambda(Å) = 1e8 / wn(cm^-1)
        w = 1e8 / w[::-1]
        f = f[::-1]
        unit = 'angstrom'

    if unit == 'nm':
        w = w * 10.0
        unit = 'angstrom'

    print(f"[io] Final FTS: {len(w)} points, wavelength range {w.min():.1f} - {w.max():.1f} Å (unit detected: {unit})")
    return w, f
