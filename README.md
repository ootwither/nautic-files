Extremely quick and dirty vibe-coded dive log parser for files from the SUUNTO Nautic.

# Installation: 

Clone the repo and run:
pip install .
from the project directory.

# Usage:

# Single dive (auto-detects paired .fit file)
divelog.py some_dive.json

# All dives in a directory
divelog.py . --batch

# Override gas if no FIT file
divelog.py some_dive.json --gas "EANx32"

# Different tank size for SAC calculation
divelog.py some_dive.json --tank-volume 10
