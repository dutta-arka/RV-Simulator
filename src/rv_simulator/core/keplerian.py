"""
Keplerian orbital mechanics for planetary systems
"""

import numpy as np
import ast
from datetime import datetime, timedelta
from math import pi, sin, sqrt, radians
from ..utils.constants import G, M_SUN, M_EARTH

def parse_planets(planets_str):
    """Parse planet parameters from semicolon-separated string"""
    planets = []
    for planet_str in planets_str.split(';'):
        mass, period, ecc, inc = map(float, planet_str.split(','))
        planets.append({
            'mass': mass * M_EARTH,
            'period': period * 86400,  # to seconds
            'ecc': ecc,
            'inc': radians(inc)
        })
    return planets

def keplerian_velocity(star_mass, planet, times, use_gr=False):
    """Calculate Keplerian RV curve for a single planet"""
    period = planet['period']
    ecc = planet['ecc']
    inc = planet['inc']
    p_mass = planet['mass']
    total_mass = star_mass + p_mass
    a_cubed = G * total_mass * period**2 / (4 * pi**2)
    a = a_cubed ** (1 / 3)
    
    # RV semi-amplitude
    K = (2 * pi * a * sin(inc)) / (period * sqrt(1 - ecc**2)) * (p_mass / total_mass)
    
    if use_gr:
        c = 299792458  # m/s
        gr_factor = 1 + (3 * G * star_mass) / (a * c**2 * (1 - ecc**2))
        K *= gr_factor
        
    return [K * sin(2 * pi * t / period) for t in times]

def simulate_planetary_system(star_mass, planets_str, obs_times_dt, use_gr=False):
    """
    Simulate an RV curve for a planetary system
    
    Parameters
    ----------
    star_mass: float: Star mass in solar masses
    planets_str: str: Planet parameters as a semicolon-separated string
    obs_times_dt: list: List of datetime objects for observations
    use_gr: bool: Apply general relativistic corrections
        
    Returns
    -------
    total_vel: array: Combined RV curve from all planets
    """
    star_mass_kg = star_mass * M_SUN
    planets = parse_planets(planets_str)
    
    # Convert to elapsed seconds from first observation
    t0 = obs_times_dt[0]
    times_sec = np.array([(t - t0).total_seconds() for t in obs_times_dt])
    
    total_vel = np.zeros_like(times_sec, dtype=float)
    for planet in planets:
        rv = keplerian_velocity(star_mass_kg, planet, times_sec, use_gr)
        total_vel += rv
        
    return total_vel
