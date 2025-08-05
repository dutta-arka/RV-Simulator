"""
Simulate a planetary system and generate radial velocity (RV) outputs.

This script accepts planetary system parameters via command-line interface, supports
multiple planets, and optionally applies general relativistic (GR) corrections.

Example usage:
python simulate_planetary_system.py \
    --star_mass 1.0 \
    --planets "1.0,365,0.0,90;0.003,10,0.1,60" \
    --num_obs 30 \
    --use_gr

Arguments:
- --star_mass : Mass of the star in solar masses
- --planets   : Semi-colon separated list of planets, each as "mass(M_earth),period(days),eccentricity,inc(deg)"
- --num_obs   : Number of observations
- --use_gr    : Optional flag to enable general relativistic correction

Outputs a shell command that can be used with generator_exp.py
"""

import numpy as np
import argparse
from math import pi, sin, sqrt, radians

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_sun = 1.98847e30  # kg
M_earth = 5.9722e24  # kg


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate planetary system RVs.")
    parser.add_argument("--star_mass", type=float, required=True, help="Star mass in solar masses")
    parser.add_argument("--planets", type=str, required=True,
                        help="Semicolon-separated list: 'mass,period,ecc,inc' per planet")
    parser.add_argument("--num_obs", type=int, required=True, help="Number of RV observations")
    parser.add_argument("--use_gr", action="store_true", help="Apply GR correction")
    return parser.parse_args()


def parse_planets(planets_str):
    planets = []
    for planet_str in planets_str.split(';'):
        mass, period, ecc, inc = map(float, planet_str.split(','))
        planets.append({
            'mass': mass * M_earth,
            'period': period * 86400,  # to seconds
            'ecc': ecc,
            'inc': radians(inc)
        })
    return planets


def keplerian_velocity(star_mass, planet, times, use_gr=False):
    period = planet['period']
    ecc = planet['ecc']
    inc = planet['inc']
    p_mass = planet['mass']
    total_mass = star_mass + p_mass
    a_cubed = G * total_mass * period**2 / (4 * pi**2)
    a = a_cubed ** (1 / 3)

    # RV semi-amplitude
    K = (2 * pi * a * sin(inc)) / (period * sqrt(1 - ecc**2)) * (p_mass / total_mass)

    # GR correction (simplified approximation for precession effects)
    if use_gr:
        c = 299792458  # m/s
        gr_factor = 1 + (3 * G * star_mass) / (a * c**2 * (1 - ecc**2))
        K *= gr_factor

    return [K * sin(2 * pi * t / period) for t in times]


def simulate():
    args = parse_args()
    star_mass = args.star_mass * M_sun
    planets = parse_planets(args.planets)
    num_obs = args.num_obs
    use_gr = args.use_gr

    times = np.linspace(0, max(p['period'] for p in planets), num_obs)
    total_vel = np.zeros_like(times)

    for planet in planets:
        rv = keplerian_velocity(star_mass, planet, times, use_gr)
        total_vel += rv

    time_step = "10d0h"
    ip_width = 20.0
    ip_type = "bigaussian"
    asymmetry = 0.2
    out_file = "spectrum.csv"
    use_template = True

    vel_list_str = "[" + ", ".join(f"{v:.2f}" for v in total_vel) + "]"

    cmd = f"python3 generator_exp.py -num_obs {num_obs} -vel_list '{vel_list_str}' " \
          f"-time_step {time_step} -ip_width {ip_width} -ip_type '{ip_type}' " \
          f"-asymmetry {asymmetry} -file {out_file}"

    if use_template:
        cmd += " -template"

    print("\nGenerated command to run generator_exp.py:\n")
    print(cmd)


if __name__ == "__main__":
    simulate()