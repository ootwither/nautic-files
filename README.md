Usage:

# Single dive (auto-detects paired .fit file)
.venv/bin/python3 divelog.py some_dive.json

# All dives in a directory
.venv/bin/python3 divelog.py . --batch

# Override gas if no FIT file
.venv/bin/python3 divelog.py some_dive.json --gas "EANx32"

# Different tank size for SAC calculation
.venv/bin/python3 divelog.py some_dive.json --tank-volume 10