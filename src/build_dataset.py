import json
import time
from pathlib import Path

import pandas as pd
import requests

# Project directories
ROOT = Path(__file__).resolve().parent.parent   # repo root (one level above src/)
CACHE = ROOT / "cache"
DATA = ROOT / "data"

# Create directories if they don't already exist
CACHE.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"
YEARS = [2023, 2024, 2025]
PAUSE = 0.5   # seconds to wait before every real request


def get(endpoint, **params):
    """Fetch one OpenF1 endpoint as a list of dicts, with an on-disk cache."""

    # Create a unique cache filename based on endpoint + parameters
    key = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = CACHE / f"{key}.json"

    # Use cached response if we have already downloaded this data
    if path.exists():
        return json.loads(path.read_text())

    # Otherwise query the API (up to 5 attempts)
    for attempt in range(5):
        time.sleep(PAUSE)

        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)

        # Successful response
        if r.status_code == 200:
            data = r.json()

            # Save response for future runs
            path.write_text(json.dumps(data))
            return data

        if r.status_code == 404:          # "no results found"
            return []

        # Wait longer after each failed attempt
        time.sleep(2 * 2 ** attempt)

    raise RuntimeError(f"{endpoint} {params}: HTTP {r.status_code}")


def expand_stints(stints):
    """Turn one-row-per-stint into one-row-per-lap with compound and tyre age."""

    rows = []

    # Ignore stints that do not have valid lap boundaries
    for st in stints.dropna(subset=["lap_start", "lap_end"]).itertuples():

        # Create one row for every lap in the stint
        for lap in range(int(st.lap_start), int(st.lap_end) + 1):
            rows.append({
                "driver_number": st.driver_number,
                "lap_number": lap,
                "stint_number": st.stint_number,
                "compound": st.compound,

                # Tyre age increases by one every lap
                "tyre_age": st.tyre_age_at_start + (lap - st.lap_start),
            })

    out = pd.DataFrame(rows)

    # Ensure only one record exists per driver/lap combination
    return out.drop_duplicates(["driver_number", "lap_number"], keep="last")


# Columns we want to keep from lap data
LAP_COLS = [
    "driver_number",
    "lap_number",
    "lap_duration",
    "is_pit_out_lap",
    "date_start",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3"
]

# Columns we want to keep from race control messages
RC_COLS = [
    "date",
    "lap_number",
    "category",
    "flag",
    "scope",
    "message"
]

# One DataFrame per race will be stored here
lap_frames, rc_frames = [], []

for year in YEARS:

    # dataframe of meetings from that year (only meeting_key and meeting_name columns)
    meetings = pd.DataFrame(get("meetings", year=year))[["meeting_key", "meeting_name"]]

    # all sessions from that year
    sessions = get("sessions", year=year, session_name="Race")

    # Process every race session separately
    for s in sessions:

        if s.get("is_cancelled"):
            continue

        sk = s["session_key"]

        try:
            # Download all required datasets for this session
            laps = pd.DataFrame(get("laps", session_key=sk))
            stints = pd.DataFrame(get("stints", session_key=sk))
            drivers = pd.DataFrame(get("drivers", session_key=sk))
            rc = pd.DataFrame(get("race_control", session_key=sk))

            if laps.empty or stints.empty:
                raise ValueError("no laps or stints returned")

            # add the data from expand_stints
            # Result: lap data + tyre compound/stint information
            df = laps[LAP_COLS].merge(
                expand_stints(stints),
                on=["driver_number", "lap_number"],
                how="left"
            )

            # Add driver abbreviations and team names to every lap
            df = df.merge(
                drivers[["driver_number", "name_acronym", "team_name"]]
                .drop_duplicates("driver_number"),
                on="driver_number",
                how="left"
            )

            # Add race/session metadata
            df["session_key"] = sk
            df["meeting_key"] = s["meeting_key"]
            df["year"] = year
            df["circuit"] = s["circuit_short_name"]
            df["country"] = s["country_name"]
            df["race_date"] = s["date_start"]

            # Add human-readable race name
            df = df.merge(meetings, on="meeting_key", how="left")

            # Keep only race-control columns we care about
            rc = rc.reindex(columns=RC_COLS)

            # Preserve which session each message belongs to
            rc["session_key"] = sk

        except Exception as e:
            print(f"SKIPPED {year} session {sk} ({s.get('country_name')}): {e}")
            continue

        # Store race data for concatenation later
        lap_frames.append(df)
        rc_frames.append(rc)

        print(f"done {year} {s['country_name']} (session {sk})")

# Combine all races into two final DataFrames
laps_all = pd.concat(lap_frames, ignore_index=True)
rc_all = pd.concat(rc_frames, ignore_index=True)

# Save datasets in a compact columnar format
laps_all.to_parquet(DATA / "laps_2023_2025.parquet")
rc_all.to_parquet(DATA / "race_control_2023_2025.parquet")

print(laps_all.shape, rc_all.shape)