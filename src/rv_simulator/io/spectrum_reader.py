"""
Spectrum reading functionality for CSV and FITS files
"""

import pandas as pd
from astropy.io import fits
from ..utils.constants import DEFAULT_FTS_FILE

def read_spectrum_csv(file):
    """Reads a synthetic spectrum from a CSV file."""
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

def read_FTS_fits(fts_file=None):
    """Reads the FTS iodine spectrum from a FITS file."""
    if fts_file is None:
        fts_file = DEFAULT_FTS_FILE
        
    with fits.open(fts_file) as hdul:
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
