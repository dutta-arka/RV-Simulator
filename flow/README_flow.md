This folder contains two Python files that can be downloaded and easily executed to simulate 'Default' instrument spectra with a given radial velocity for viper to decode. Please *only* download these two files from this folder only. The coherence of the main branch is still a bit less. One needs the following packages as prerequisites:

```
!pip install numpy scipy pandas astropy jplephem
```
Of course, we assume that you will also have literally any high-resolution spectra handy.

Once this step is done, we can first dive into the helper sort of code, simulate_planetary_system.py. One can use this Python file to easily input standard code to generate command code that writes the command for the `generator_simulation.py` file, allowing for the creation of any given number of synthetic files.

## Documentation for `simulate_planetary_system.py`

This script is a command-line tool for simulating the radial velocity (RV) signature of a star due to its orbiting planets. Its main purpose is to calculate the RVs for a given set of observation times and then output a formatted command that can be directly used to run `generator_simulation.py`.

Key Features:
----------
* It allows you to specify the star's mass and the properties of each planet (mass, period, eccentricity, and inclination) directly from the command line.
* You can define your observation schedule in three ways:
  - Set a fixed number of observations with a constant time step between them.
  - Provide a specific list of dates and times for your observations.
  - Generate a set number of observations at random times within a calculated time span.
*  Includes an option to apply a general relativistic correction. 

Command-Line Arguments:
----------
* `--star_mass`: [Required] The mass of the star in solar masses.
* `--planets`: [Required] A string describing the planets. For multiple planets, separate them with a semicolon (;). Each planet is defined by mass (in Earth mass), period (in days), eccentricity, and inclination (in degrees). Use an empty string "" if there are no planets.
* `--num_obs`: The number of observations you want to simulate. It's required if you're not providing a specific list of dates.
* `--time_step`: The interval between uniformly spaced observations (e.g., `10d0h` for 10 days). This is the default method if no other timing is specified.
* `--date_list`: A string containing a Python list of ISO-formatted dates (e.g., `"['2025-02-06T00:00:00', '2025-02-08T12:30:00']"`) for irregularly timed observations.
* `--random_dates`: A flag to generate random observation times. Just add `--random_dates` to the command. It sets a baseline of roughly `7*num_obs`.
* `--use_gr`: A flag to apply the general relativity correction. Add `--use_gr` to enable it.

Usage Examples:
You can run the script from your terminal. Here are examples for the different timing modes:

1. Uniform Observation Spacing
This simulates 5 observations spaced 10 days apart for a star with two planets.

```
python simulate_planetary_system.py \
--star_mass 1.0 \
--planets "1.0,365,0.0,90;0.003,10,0.1,60" \
--num_obs 5 \
--time_step 10d0h
```

2. Specific Observation Dates (Non-Uniform)
This simulates observations on three specific dates that you provide.

```
python simulate_planetary_system.py \
--star_mass 1.0 \
--planets "1.0,365,0.0,90;0.003,10,0.1,60" \
--date_list "['2025-02-06T00:00:00','2025-02-07T12:00:00','2025-02-09T06:30:00']"
```

4. Random Observation Dates
This simulates 200 observations at random times for a star with no planets.

```
python simulate_planetary_system.py \
--star_mass 1.0 \
--planets "" \
--num_obs 200 \
--random_dates
```

This is what we are currently running for the baseline check.

## Documentation for `generator_simulation.py`

This script creates synthetic spectra! Download and keep this file in the same folder where you want to create the synthetic spectra! By simply pasting the outputs from the last command, you can generate any given number of synthetic observations (given that you have a high-resolution spectrum of a star and iodine spectra in hand).

Even without using the previous script, for simple test cases, this file can be used directly. Details about the usage of this code are listed below.

Key Features:
----------
* The simulator injects user-specified radial-velocity shifts.
* It can optionally add synthetic stellar activity noise.
* The code computes accurate barycentric velocity corrections through Astropy’s coordinate and ephemeris utilities.
* It can generate a stellar template spectrum without iodine contamination when the -template flag is used.
* It constructs a full-spectrum, uniform log-lambda grid and extracts individual echelle orders using VIPER-style trimming, ensuring compatibility with pipeline expectations.

Three Operation Modes:
---------------------

1. DEFAULT MODE (default):
   Uses a synthetic CSV spectrum + hardcoded instrument settings
   ```python3 generator_simulation.py -file spectrum.csv -num_obs 3 -vel_list "[100,200,300]" -output_dir temp1```

2. AUTO MODE: 
   Auto-infers everything from the provided FITS files
   ```python3 generator_simulation.py -mode auto -observed_spectrum obs.fits -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]" -output_dir temp1```
   
3. MANUAL MODE:
   Uses observed spectrum + user-provided order ranges in text format
   ```python3 generator_simulation.py -mode manual -observed_spectrum obs.fits -orders_file my_orders.txt -fts_file custom.fits -num_obs 3 -vel_list "[100,200,300]" -output_dir temp1```

Here, `my_orders.txt` file should have the following format:
```
(5000.0, 5100.0)
(5100.0, 5200.0)
...
```

Command-Line Arguments:
----------
* `-mode`                Look at the “Operation Modes” section for a full explanation of the three available modes (default, auto, and manual). This flag determines how the simulator interprets wavelength ranges, FTS information, and instrument settings.
* `-observed_spectrum`   provides one sample observation to match the exact wavelength splitting and echelle orders (riskier). Usage: `-observed_spectrum \path\to\observation1`.
* `-orders_file`         This argument allows you to provide a text file containing the exact order boundaries used in your instrument. Each line in the file must list the wavelength limits of one order.
* `-num_obs`             Number of observations to generate. You can simply add `-num_obs 200` to get 200 synthetic files.
* `-vel_list`            RV shifts (m/s) for each observation as a list. You need to carefully match the list with the number of observations asked to create. If not specified, it will assume evenly spread increasing velocity in the range of -1000 to 1000 m/s. You can also change this if you want to, by setting `-vel_list "[-100, 100]"`.
* `-date_list`           You can use this feature in case you want to have a specific spacing of dates. Be careful to match the length with 'num_obs' if used.
* `-time_step`           Spacing between observations, e.g. '3d0h' for 3 days and 0 hours.
* `-file`                Path to the synthetic spectrum CSV file. Input should be like `-file \path\to\your\spectraum.csv`.
* `-output_dir`          Directory to save output files (default: current directory). One can change this by entering `-output_dir \path\to\new\directory1`.
* `-ip_width`            IP width in pixels (Gaussian sigma). You can choose a single IP for all synthetic observation by setting `-ip_width anyvalue` or set a range of IP values randomly varying: `-ip_width [minmumvalue, maximumvalue]`.
* `-ip_type`             Type of instrumental profile to convolve with: 'gaussian', 'bigaussian', or 'voigt'.
* `-asymmetry`           Asymmetry factor (-1 to 1) for bi-Gaussian IP. You need to set a specific number for this in the range when using 'bigaussian', or it will assume the value to be zero.
* `-gamma`               Lorentzian width (gamma) for Voigt profile convolution. You need to select an appropriate value for this while using 'voigt'; otherwise, it will go to zero.
* `-template`            This optional flag enables the creation of a template FITS file. As mentioned previously, the template contains only the stellar spectrum and excludes all gas-cell (iodine) lines.
* `-site`                This optional flag provides the observatory name, as recognised by Astropy (for example, 'Keck'). The site information is used to compute barycentric corrections accurately.
* `-add_noise`           Adds noise. The level of noise cannot be changed for now!

Usage Examples:
You can run the script from your terminal.

```
python3 generator_simulation.py \
    -num_obs 10 \
    -vel_list "[0,50]" \
    -time_step "3d0h" \
    -file /path/to/synthetic_spectrum.csv \
    -output_dir /path/to/output_directory \
    -ip_width 2.5 \
    -ip_type voigt \
    -gamma 0.8 \
    -template \
    -site TLS \
    -add_noise
```
