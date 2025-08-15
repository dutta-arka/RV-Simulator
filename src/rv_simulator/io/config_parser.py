"""
YAML configuration file parser
"""
import yaml
from datetime import datetime
from ..core.keplerian import simulate_planetary_system

def load_config(yaml_path):
    """Load and parse YAML configuration file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_cli_args(config):
    """Convert YAML config to CLI arguments for existing scripts"""
    # Extract star mass
    star_mass = config['star']['mass']
    
    # Build planets string
    planets_list = []
    for planet in config['planets']:
        planet_str = f"{planet['mass']},{planet['period']},{planet['eccentricity']},{planet['inclination']}"
        planets_list.append(planet_str)
    planets_str = ";".join(planets_list)
    
    # Handle observation times
    obs_times = None
    if 'times' in config['observations']:
        obs_times = config['observations']['times']
    elif 'uniform' in config['observations']:
        num_obs = config['observations']['uniform']['num_obs']
        cadence = config['observations']['uniform']['cadence']
    
    return {
        'star_mass': star_mass,
        'planets': planets_str,
        'observation_times': obs_times,
        'instrument': config['instrument'],
        'files': config['files']
    }
