This folder contains two Python files that can be downloaded and easily executed to simulate 'Default' instrument spectra with a given radial velocity for viper to decode. One needs the following packages as prerequisites:

```
!pip install numpy scipy pandas astropy
```
Of course, we assume that you will also have literally any high-resolution spectra handy.

Once this step is done, we can first dive into the helper sort of code, simulate_planetary_system.py. One can use this Python file to easily input standard code to generate command code that writes the command for the generator_simulation.py file, allowing for the creation of any given number of synthetic files.

## Documentation for ```simulate_planetary_system.py```

This script is a command-line tool for simulating the radial velocity (RV) signature of a star due to its orbiting planets. Its main purpose is to calculate the RVs for a given set of observation times and then output a formatted command that can be directly used to run generator_simulation.py.

Key Features:
* It allows you to specify the star's mass and the properties of each planet (mass, period, eccentricity, and inclination) directly from the command line.
* You can define your observation schedule in three ways:
  - Set a fixed number of observations with a constant time step between them.
  - Provide a specific list of dates and times for your observations.
  - Generate a set number of observations at random times within a calculated time span.
*  Includes an option to apply a general relativistic correction. 
