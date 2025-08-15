"""
Setup script for rv_simulator package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Only include the packages your scripts actually import
install_requires = [
    "numpy>=1.20",
    "pandas>=1.3",
    "astropy>=5.0",
    "scipy>=1.7"
]

setup(
    name="rv-simulator",
    version="0.1.0",
    author="Arkaprova Dutta, Jana Koehler",
    description="Radial velocity simulator for testing VIPER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dutta-arka/RV-Simulator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: GNU General Public License v3.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "rv-simulate-planets=rv_simulator.cli.simulate_planets:main",
            "rv-generate-fits=rv_simulator.cli.generate_fits:main",
        ],
    },
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8", "mypy"],
    },
)
