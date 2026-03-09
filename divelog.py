#!/usr/bin/env python3
"""Convert Suunto dive computer JSON (+FIT) exports to Obsidian markdown dive logs."""

import json
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(data, *keys, default=None):
    """Navigate nested dict safely."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and key < len(current):
            current = current[key]
        else:
            return default
    return current


def format_duration(seconds):
    """Format seconds as mm:ss."""
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_gas_label(o2_pct, he_pct):
    """Format gas blend as a human-readable label.

    21/0  -> Air
    32/0  -> EANx32
    21/35 -> TX21/35
    100/0 -> O2
    """
    if he_pct and he_pct > 0:
        return f"TX{o2_pct}/{he_pct} ({o2_pct}% O2, {he_pct}% He)"
    if o2_pct == 21:
        return "Air (21% O2)"
    if o2_pct == 100:
        return "O2 (100%)"
    return f"EANx{o2_pct} ({o2_pct}% O2)"


def format_coords(lat, lon):
    """Format decimal coordinates with N/S E/W suffixes."""
    if lat is None or lon is None:
        return None
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f} {ns}, {abs(lon):.4f} {ew}"


# ---------------------------------------------------------------------------
# FIT file extraction
# ---------------------------------------------------------------------------

def extract_fit_data(fit_path):
    """Extract dive-specific data from a paired FIT file.

    Returns a dict with keys: gas_label, o2_pct, he_pct, gf_low, gf_high,
    surface_interval, end_cns, o2_toxicity.  All may be None if not found.
    """
    try:
        import fitparse
    except ImportError:
        print("  fitparse not installed, skipping FIT file", file=sys.stderr)
        return {}

    result = {
        "gas_label": None,
        "o2_pct": None,
        "he_pct": None,
        "gf_low": None,
        "gf_high": None,
        "surface_interval": None,
        "end_cns": None,
        "o2_toxicity": None,
        "product_name": None,
    }

    try:
        fit = fitparse.FitFile(str(fit_path))

        for msg in fit.get_messages():
            if msg.name == "dive_gas":
                fields = {f.name: f.value for f in msg.fields}
                o2 = fields.get("oxygen_content")
                he = fields.get("helium_content", 0)
                if o2 is not None:
                    result["o2_pct"] = int(o2)
                    result["he_pct"] = int(he) if he else 0
                    result["gas_label"] = format_gas_label(result["o2_pct"], result["he_pct"])

            elif msg.name == "dive_settings":
                fields = {f.name: f.value for f in msg.fields}
                result["gf_low"] = fields.get("gf_low")
                result["gf_high"] = fields.get("gf_high")

            elif msg.name == "device_info":
                fields = {f.name: f.value for f in msg.fields}
                pn = fields.get("product_name")
                if pn and isinstance(pn, str):
                    result["product_name"] = pn

            elif msg.name == "session":
                for f in msg.fields:
                    if f.def_num == 142:
                        result["surface_interval"] = f.value
                    elif f.def_num == 144:
                        result["end_cns"] = f.value
                    elif f.def_num == 155:
                        result["o2_toxicity"] = f.value

    except Exception as e:
        print(f"  Warning: could not parse FIT file: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def load_dive(path):
    """Load and validate a Suunto JSON export."""
    with open(path) as f:
        data = json.load(f)

    if "DeviceLog" not in data:
        raise ValueError(f"Not a Suunto dive export: missing DeviceLog in {path}")
    if "Header" not in data["DeviceLog"] or "Samples" not in data["DeviceLog"]:
        raise ValueError(f"Incomplete Suunto export: missing Header or Samples in {path}")

    return data


def extract_header(data):
    """Pull summary fields from DeviceLog.Header."""
    h = data["DeviceLog"]["Header"]

    dt_str = h.get("DateTime")
    dt = datetime.fromisoformat(dt_str) if dt_str else None

    return {
        "datetime": dt,
        "max_depth": safe_get(h, "Depth", "Max"),
        "avg_depth": h.get("DepthAverage"),
        "dive_time": h.get("DiveTime"),
        "duration": h.get("Duration"),
        "device_name": safe_get(h, "Device", "Name"),
        "device_serial": safe_get(h, "Device", "SerialNumber"),
    }


def extract_samples(data):
    """Process DeviceLog.Samples into usable time series."""
    samples = data["DeviceLog"]["Samples"]

    # Find the first timestamp as reference
    t0 = None
    for s in samples:
        if "TimeISO8601" in s:
            t0 = datetime.fromisoformat(s["TimeISO8601"])
            break

    if t0 is None:
        return {"depth_profile": [], "temperature": [], "tank_pressure": [], "events": []}

    # Find actual surface pressure for accurate depth conversion
    surface_pressure = 101325.0  # standard, fallback
    for s in samples:
        if "SurfacePressure" in s:
            surface_pressure = s["SurfacePressure"]
            break

    depth_profile = []
    temperature = []
    tank_pressure = []
    events = []
    no_dec_times = []
    ceilings = []
    dive_end_elapsed = None  # timestamp when dive ends (DiveStatus: False)

    for s in samples:
        if "TimeISO8601" not in s:
            continue

        t = datetime.fromisoformat(s["TimeISO8601"])
        elapsed = (t - t0).total_seconds()

        # Track dive end time
        if "DiveEvents" in s:
            ev = s["DiveEvents"]
            if ev.get("DiveStatus") is False:
                dive_end_elapsed = elapsed

        # Depth from absolute pressure
        if "AbsPressure" in s:
            depth = (s["AbsPressure"] - surface_pressure) / 10093.5  # Pa per metre of seawater
            depth = max(0.0, depth)  # clamp negatives
            depth_profile.append((elapsed, depth))

        # Temperature
        if "Temperature" in s and isinstance(s["Temperature"], (int, float)):
            temp_c = s["Temperature"] - 273.15
            temperature.append((elapsed, temp_c))

        # Tank pressure
        if "Cylinders" in s:
            cyl = s["Cylinders"]
            if isinstance(cyl, list) and len(cyl) > 0:
                p = cyl[0].get("Pressure")
                if p is not None:
                    tank_pressure.append((elapsed, p / 100000))  # to bar

        # Dive events
        if "DiveEvents" in s:
            ev = s["DiveEvents"]
            if "Alarm" in ev:
                alarm = ev["Alarm"]
                status = "triggered" if alarm.get("Active") else "cleared"
                events.append((elapsed, f"Alarm: {alarm.get('Type', '?')} {status}"))
            if "DiveState" in ev:
                events.append((elapsed, f"State: {ev['DiveState']}"))
            if "Notify" in ev:
                notify = ev["Notify"]
                status = "active" if notify.get("Active") else "cleared"
                events.append((elapsed, f"Notify: {notify.get('Type', '?')} {status}"))

        # Deco info
        if "NoDecTime" in s:
            no_dec_times.append((elapsed, s["NoDecTime"]))
        if "Ceiling" in s:
            ceilings.append((elapsed, s["Ceiling"]))

    # GPS from first sample
    gps_lat = None
    gps_lon = None
    for s in samples:
        origin = s.get("DiveRouteOrigin")
        if origin:
            gps_lat = origin.get("Latitude")
            gps_lon = origin.get("Longitude")
            break

    # Temperature range from samples (header labels are swapped!)
    temp_min = min(t for _, t in temperature) if temperature else None
    temp_max = max(t for _, t in temperature) if temperature else None

    # Was it a deco dive?
    was_deco = any(c > 0 for _, c in ceilings)

    # Extract alarms (paired start/end)
    alarms = []
    for elapsed, desc in events:
        if desc.startswith("Alarm:") and "triggered" in desc:
            alarm_type = desc.replace("Alarm: ", "").replace(" triggered", "")
            alarms.append((elapsed, alarm_type))

    # Pair each pressure reading with the nearest depth (for surface artefact filtering)
    tank_pressure_with_depth = []
    for tp_elapsed, tp_bar in tank_pressure:
        nearest_depth = 0.0
        if depth_profile:
            nearest_depth = min(depth_profile, key=lambda d: abs(d[0] - tp_elapsed))[1]
        tank_pressure_with_depth.append((tp_elapsed, tp_bar, nearest_depth))

    return {
        "depth_profile": depth_profile,
        "temperature": temperature,
        "tank_pressure": tank_pressure,
        "tank_pressure_with_depth": tank_pressure_with_depth,
        "events": events,
        "no_dec_times": no_dec_times,
        "ceilings": ceilings,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "was_deco": was_deco,
        "alarms": alarms,
        "dive_end_elapsed": dive_end_elapsed,
    }


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def get_dive_pressures(samples):
    """Get start and end tank pressure, filtering out surface artefacts.

    The tank transmitter shows a ~1 ATM pressure cliff when surfacing.
    The computer's displayed end pressure matches the last reading while
    still submerged, so we use the last pressure reading at depth > 0.5m.
    """
    tpwd = samples.get("tank_pressure_with_depth", [])

    if tpwd:
        start_bar = tpwd[0][1]
        # Last reading while still submerged
        submerged = [(t, p, d) for t, p, d in tpwd if d > 0.5]
        if submerged:
            end_bar = submerged[-1][1]
        else:
            end_bar = tpwd[-1][1]
        return start_bar, end_bar

    # Fallback: no depth-paired data
    tp = samples["tank_pressure"]
    if not tp:
        return None, None
    return tp[0][1], tp[-1][1]


def calc_sac_rate(avg_depth, dive_time_s, start_bar, end_bar, tank_volume_l=12.0):
    """Calculate Surface Air Consumption rate in litres/min."""
    if None in (avg_depth, dive_time_s, start_bar, end_bar):
        return None
    if dive_time_s <= 0 or start_bar <= end_bar <= 0:
        return None

    pressure_used = start_bar - end_bar
    gas_consumed = pressure_used * tank_volume_l
    avg_ambient_ata = 1 + (avg_depth / 10)
    time_min = dive_time_s / 60

    if avg_ambient_ata <= 0 or time_min <= 0:
        return None

    return gas_consumed / (avg_ambient_ata * time_min)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown(header, samples, fit_data, gas_label, sac_rate,
                      tank_volume, chart_filename):
    """Build the complete markdown string for Obsidian."""
    dt = header["datetime"]
    date_str = dt.strftime("%Y-%m-%d") if dt else "Unknown"
    title_date = dt.strftime("%d %b %Y, %H:%M") if dt else "Unknown"
    device = (fit_data.get("product_name") if fit_data else None) or header.get("device_name", "Unknown")

    lines = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"tags: [dive-log]")
    lines.append(f"date: {date_str}")
    lines.append(f"dive_computer: {device}")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# Dive Log -- {title_date}")
    lines.append("")

    # Chart embed
    lines.append(f"![[{chart_filename}]]")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Date | {title_date} |")
    lines.append(f"| Duration | {format_duration(header.get('dive_time'))} |")

    max_d = header.get("max_depth")
    lines.append(f"| Max Depth | {max_d:.2f} m |" if max_d else "| Max Depth | N/A |")

    avg_d = header.get("avg_depth")
    lines.append(f"| Avg Depth | {avg_d:.2f} m |" if avg_d else "| Avg Depth | N/A |")

    if samples["temp_min"] is not None and samples["temp_max"] is not None:
        lines.append(f"| Water Temp | {samples['temp_min']:.1f} -- {samples['temp_max']:.1f} °C |")

    coords = format_coords(samples.get("gps_lat"), samples.get("gps_lon"))
    if coords:
        lines.append(f"| Location | {coords} |")

    # Surface interval
    si = fit_data.get("surface_interval") if fit_data else None
    if si is None:
        # Fall back to within-session surface time
        duration = header.get("duration")
        dive_time = header.get("dive_time")
        if duration and dive_time:
            si = duration - dive_time
    if si is not None:
        lines.append(f"| Surface Interval | {format_duration(si)} |")

    lines.append("")

    # Gas table
    start_bar, end_bar = get_dive_pressures(samples)

    if gas_label or start_bar is not None:
        lines.append("## Gas")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")

        if gas_label:
            lines.append(f"| Gas | {gas_label} |")
        if start_bar is not None:
            lines.append(f"| Start Pressure | {start_bar:.1f} bar |")
        if end_bar is not None:
            lines.append(f"| End Pressure | {end_bar:.1f} bar |")
        if start_bar is not None and end_bar is not None:
            lines.append(f"| Consumption | {start_bar - end_bar:.1f} bar |")
        if sac_rate is not None:
            lines.append(f"| SAC Rate | {sac_rate:.1f} L/min |")
            lines.append(f"| Tank Volume | {tank_volume} L (assumed) |")

        lines.append("")

    # Decompression
    lines.append("## Decompression")
    lines.append("")
    deco_parts = []
    if samples["was_deco"]:
        deco_parts.append("Decompression dive -- ceiling obligations encountered.")
    else:
        deco_parts.append("No-decompression dive.")

    end_cns = fit_data.get("end_cns") if fit_data else None
    o2_tox = fit_data.get("o2_toxicity") if fit_data else None
    if end_cns is not None:
        deco_parts.append(f"CNS: {end_cns}%.")
    if o2_tox is not None:
        deco_parts.append(f"OTU: {o2_tox}.")

    gf_low = fit_data.get("gf_low") if fit_data else None
    gf_high = fit_data.get("gf_high") if fit_data else None
    if gf_low is not None and gf_high is not None:
        deco_parts.append(f"GF: {gf_low}/{gf_high}.")

    lines.append(" ".join(deco_parts))
    lines.append("")

    # Alarms
    if samples["alarms"]:
        lines.append("## Alarms")
        lines.append("")
        for elapsed, alarm_type in samples["alarms"]:
            lines.append(f"- {alarm_type} at {format_duration(elapsed)}")
        lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(samples, header, output_path):
    """Create and save a dive profile chart as PNG."""
    if not samples["depth_profile"]:
        print("  No depth data for chart", file=sys.stderr)
        return

    times = [t / 60 for t, _ in samples["depth_profile"]]
    depths = [d for _, d in samples["depth_profile"]]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0a1628")
    ax1.set_facecolor("#0a1628")

    # Depth line
    ax1.plot(times, depths, color="#4fc3f7", linewidth=1.5, zorder=3)

    # Gradient fill
    ax1.fill_between(times, 0, depths, alpha=0.3, color="#0288d1", zorder=2)

    # Invert y-axis (depth increases downward)
    max_depth = max(depths)
    ax1.invert_yaxis()
    ax1.set_ylim(max_depth * 1.15, -0.5)

    # Annotate max depth
    max_idx = depths.index(max_depth)
    max_time = times[max_idx]
    # Place annotation to the right if room, otherwise to the left
    text_offset_x = 1 if max_time < times[-1] * 0.75 else -2
    ax1.annotate(
        f"{max_depth:.1f} m",
        xy=(max_time, max_depth),
        xytext=(max_time + text_offset_x, max_depth * 1.08),
        color="white",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        zorder=5,
    )

    # Tank pressure on secondary axis
    if samples["tank_pressure"]:
        ax2 = ax1.twinx()
        tp_times = [t / 60 for t, _ in samples["tank_pressure"]]
        tp_vals = [p for _, p in samples["tank_pressure"]]
        ax2.plot(tp_times, tp_vals, color="#ffb74d", linewidth=1.2,
                 alpha=0.7, linestyle="--", zorder=2)
        ax2.set_ylabel("Tank Pressure (bar)", color="#ffb74d", fontsize=11)
        ax2.tick_params(axis="y", colors="#ffb74d")
        ax2.spines["right"].set_color("#ffb74d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        # Pad the pressure range a little
        p_min, p_max = min(tp_vals), max(tp_vals)
        p_range = p_max - p_min if p_max > p_min else 10
        ax2.set_ylim(p_min - p_range * 0.1, p_max + p_range * 0.1)

    # Styling
    ax1.set_xlabel("Time (min)", color="white", fontsize=11)
    ax1.set_ylabel("Depth (m)", color="#4fc3f7", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#ffffff40")
    ax1.spines["left"].set_color("#4fc3f7")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", color="#ffffff15", linestyle="-", linewidth=0.5)

    # Title
    dt = header.get("datetime")
    if dt:
        title = f"Dive Profile -- {dt.strftime('%d %b %Y, %H:%M')}"
    else:
        title = "Dive Profile"
    ax1.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Suunto dive computer exports to Obsidian markdown dive logs."
    )
    parser.add_argument("input", help="Path to Suunto JSON export (or directory with --batch)")
    parser.add_argument("--gas", default=None,
                        help="Override gas label (default: auto-detect from FIT, fallback Air)")
    parser.add_argument("--tank-volume", type=float, default=12.0,
                        help="Tank volume in litres for SAC calculation (default: 12)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input file)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all .json files in the input directory")
    return parser.parse_args()


def process_file(input_path, args):
    """Full pipeline for one JSON file."""
    input_path = Path(input_path)
    print(f"\nProcessing: {input_path.name}", file=sys.stderr)

    # Load JSON
    data = load_dive(input_path)
    header = extract_header(data)
    samples = extract_samples(data)

    # Try paired FIT file
    fit_path = input_path.with_suffix(".fit")
    fit_data = {}
    if fit_path.exists():
        print(f"  Found FIT file: {fit_path.name}", file=sys.stderr)
        fit_data = extract_fit_data(fit_path)
    else:
        print(f"  No paired FIT file found", file=sys.stderr)

    # Determine gas label
    gas_label = args.gas  # CLI override takes priority
    if gas_label is None:
        gas_label = fit_data.get("gas_label")
    if gas_label is None:
        gas_label = "Air (21% O2)"

    # Calculate SAC rate
    start_bar, end_bar = get_dive_pressures(samples)
    sac_rate = calc_sac_rate(
        header.get("avg_depth"),
        header.get("dive_time"),
        start_bar, end_bar,
        args.tank_volume,
    )

    # Output paths
    dt = header.get("datetime")
    if dt:
        stem = dt.strftime("%Y-%m-%d_%H%M")
    else:
        stem = input_path.stem

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    chart_path = out_dir / f"{stem}_profile.png"
    chart_filename = f"{stem}_profile.png"

    # Generate chart
    generate_chart(samples, header, chart_path)
    print(f"  Chart: {chart_path}", file=sys.stderr)

    # Generate markdown
    md = generate_markdown(header, samples, fit_data, gas_label, sac_rate,
                           args.tank_volume, chart_filename)
    md_path.write_text(md)
    print(f"  Markdown: {md_path}", file=sys.stderr)

    # Summary
    print(f"  Depth: {header.get('max_depth', '?')}m max, "
          f"{header.get('avg_depth', '?')}m avg", file=sys.stderr)
    print(f"  Time: {format_duration(header.get('dive_time'))}", file=sys.stderr)
    print(f"  Gas: {gas_label}", file=sys.stderr)
    if sac_rate:
        print(f"  SAC: {sac_rate:.1f} L/min", file=sys.stderr)


def main():
    args = parse_args()

    if args.batch:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: {args.input} is not a directory", file=sys.stderr)
            sys.exit(1)
        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            print(f"No .json files found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        for f in json_files:
            try:
                process_file(f, args)
            except Exception as e:
                print(f"  Error processing {f.name}: {e}", file=sys.stderr)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            sys.exit(1)
        process_file(input_path, args)


if __name__ == "__main__":
    main()
