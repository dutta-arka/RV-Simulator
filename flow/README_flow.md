This folder contains two Python files that can be downloaded and easily executed to simulate 'Default' instrument spectra with a given radial velocity for viper to decode. One needs the following packages as prerequisites:

```
!pip install numpy scipy pandas astropy
```
Of course, we assume that you will also have literally any high-resolution spectra handy.

Once this step is done, we can first dive into the helper sort of code, simulate_planetary_system.py. One can use this Python file to easily input standard code to generate command code that writes the command for the `generator_simulation.py` file, allowing for the creation of any given number of synthetic files.

## Documentation for `simulate_planetary_system.py`

This script is a command-line tool for simulating the radial velocity (RV) signature of a star due to its orbiting planets. Its main purpose is to calculate the RVs for a given set of observation times and then output a formatted command that can be directly used to run `generator_simulation.py`.

Key Features:
* It allows you to specify the star's mass and the properties of each planet (mass, period, eccentricity, and inclination) directly from the command line.
* You can define your observation schedule in three ways:
  - Set a fixed number of observations with a constant time step between them.
  - Provide a specific list of dates and times for your observations.
  - Generate a set number of observations at random times within a calculated time span.
*  Includes an option to apply a general relativistic correction. 

Command-Line Arguments:
* `--star_mass`: [Required] The mass of the star in solar masses.
* `--planets`: [Required] A string describing the planets. For multiple planets, separate them with a semicolon (;). Each planet is defined by mass (in Earth mass), period (in days), eccentricity, and inclination (in degrees). Use an empty string "" if there are no planets.
* `--num_obs`: The number of observations you want to simulate. It's required if you're not providing a specific list of dates.
* `--time_step`: The interval between uniformly spaced observations (e.g., `10d0h` for 10 days). This is the default method if no other timing is specified.
* `--date_list`: A string containing a Python list of ISO-formatted dates (e.g., `"['2025-02-06T00:00:00', '2025-02-08T12:30:00']"`) for irregularly timed observations.
* `--random_dates`: A flag to generate random observation times. Just add `--random_dates` to the command. It sets a baseline of roughly `7*num_obs`.
* `--use_gr`: A flag to apply the general relativity correction. Add `--use_gr` to enable it.

Usage Examples
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






