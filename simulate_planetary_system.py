"""
Simulate a planetary system and generate radial velocity (RV) outputs.
This script accepts planetary system parameters via command-line interface, supports
multiple planets, and optionally applies general relativistic (GR) corrections.

Example usage:
--------------
Uniform spacing:
python simulate_planetary_system.py \
    --star_mass 1.0 \
    --planets "1.0,365,0.0,90;0.003,10,0.1,60" \
    --num_obs 5 \
    --time_step 10d0h

Unequal spacing:
python simulate_planetary_system.py \
    --star_mass 1.0 \
    --planets "1.0,365,0.0,90;0.003,10,0.1,60" \
    --date_list "['2025-02-06T00:00:00','2025-02-07T12:00:00','2025-02-09T06:30:00','2025-02-15T00:00:00','2025-02-18T18:00:00']"
"""
import numpy as np
import argparse
import ast
from datetime import datetime, timedelta
from math import pi, sin, sqrt, radians

# --- Constants ---
G = 6.67430e-11       # m^3 kg^-1 s^-2
M_sun = 1.98847e30    # kg
M_earth = 5.9722e24   # kg

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate planetary system RVs.")
    parser.add_argument("--star_mass", type=float, required=True,
                        help="Star mass in solar masses")
    parser.add_argument("--planets", type=str, required=True,
                        help="Semicolon-separated list: 'mass,period,ecc,inc' per planet")
    parser.add_argument("--num_obs", type=int, default=None,
                        help="Number of RV observations (ignored if date_list supplied)")
    parser.add_argument("--use_gr", action="store_true", help="Apply GR correction")
    parser.add_argument("--time_step", type=str, default="10d0h",
                        help="Spacing between observations: e.g. '3d0h' (ignored if date_list supplied)")
    parser.add_argument("--date_list", type=str, default=None,
                        help="Explicit list of observation datetimes (ISO strings) for non-uniform spacing")
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
    if use_gr:
        c = 299792458  # m/s
        gr_factor = 1 + (3 * G * star_mass) / (a * c**2 * (1 - ecc**2))
        K *= gr_factor
    return [K * sin(2 * pi * t / period) for t in times]

def simulate():
    args = parse_args()
    star_mass = args.star_mass * M_sun
    planets = parse_planets(args.planets)
    use_gr = args.use_gr

    # --- Generate observation times ---
    if args.date_list:
        try:
            date_list = ast.literal_eval(args.date_list)
            if not isinstance(date_list, list):
                raise ValueError()
            obs_times_dt = [datetime.fromisoformat(d) for d in date_list]
        except Exception:
            raise ValueError("Invalid format for --date_list. Example: \"['2025-02-06T00:00:00','2025-02-08T12:00:00']\"")
    else:
        if args.num_obs is None:
            raise ValueError("--num_obs is required if --date_list not given")
        time_step_match = None
        import re
        time_step_match = re.match(r'(\d+)d(\d+)h', args.time_step)
        if not time_step_match:
            raise ValueError("Invalid time_step format. Use e.g. '5d0h'")
        delta_days = int(time_step_match.group(1))
        delta_hours = int(time_step_match.group(2))
        time_delta = timedelta(days=delta_days, hours=delta_hours)
        base_time = datetime(2025, 2, 6, 0, 0, 0)
        obs_times_dt = [base_time + i * time_delta for i in range(args.num_obs)]

    # Convert obs_times_dt to list of elapsed seconds from first obs (for RV simulation)
    t0 = obs_times_dt[0]
    times_sec = np.array([(t - t0).total_seconds() for t in obs_times_dt])

    # --- RV simulation ---
    total_vel = np.zeros_like(times_sec, dtype=float)
    for planet in planets:
        rv = keplerian_velocity(star_mass, planet, times_sec, use_gr)
        total_vel += rv

    # --- Build command for generator_simulation.py ---
    ip_width = 20.0
    ip_type = "bigaussian"
    asymmetry = 0.2
    out_file = "spectrum.csv"
    use_template = True

    vel_list_str = "[" + ", ".join(f"{v:.2f}" for v in total_vel) + "]"

    if args.date_list:
        # Pass dates directly
        date_list_str = "[" + ", ".join(f"'{d.isoformat()}'" for d in obs_times_dt) + "]"
        cmd = f"python3 generator_simulation.py -vel_list '{vel_list_str}' -date_list \"{date_list_str}\" " \
              f"-ip_width {ip_width} -ip_type '{ip_type}' -asymmetry {asymmetry} -file {out_file}"
    else:
        # Pass uniform spacing via -num_obs and -time_step
        cmd = f"python3 generator_simulation.py -num_obs {len(obs_times_dt)} -vel_list '{vel_list_str}' " \
              f"-time_step {args.time_step} -ip_width {ip_width} -ip_type '{ip_type}' " \
              f"-asymmetry {asymmetry} -file {out_file}"

    if use_template:
        cmd += " -template"

    print("\nGenerated command to run generator_simulation.py:\n")
    print(cmd)

if __name__ == "__main__":
    simulate()

