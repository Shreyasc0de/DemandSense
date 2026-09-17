"""Download daily London weather from Open-Meteo for the span of the sales data.

Open-Meteo's historical reanalysis is free, needs no API key, and is published
under CC BY 4.0, so the CSV it writes can be committed alongside the code and
anyone cloning this repo reproduces the same numbers.

Run:  python scripts/fetch_weather.py
"""

from __future__ import annotations

import argparse
import sys
import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from demandsense.data import LONDON_LAT, LONDON_LON  # noqa: E402

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"


def parse_csv(body: str) -> pd.DataFrame:
    """Pull the daily table out of an Open-Meteo CSV response.

    The response opens with a metadata block (latitude, elevation, timezone)
    before the table, and the column headers carry their units in parentheses.
    Split out so it can be tested without a network call.
    """
    lines = body.splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    except StopIteration:
        raise ValueError("no daily table in the Open-Meteo response") from None
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])))
    if frame.shape[1] != 3:
        raise ValueError(f"expected 3 daily columns, got {list(frame.columns)}")
    frame.columns = ["ds", "temp_c", "precip_mm"]
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame


def fetch(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Europe/London",
        "format": "csv",
    })
    url = f"{ENDPOINT}?{query}"
    print(f"[weather] GET {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    return parse_csv(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2009-12-01")
    parser.add_argument("--end", default="2011-12-31")
    parser.add_argument("--lat", type=float, default=LONDON_LAT)
    parser.add_argument("--lon", type=float, default=LONDON_LON)
    parser.add_argument("--out", default=str(ROOT / "data" / "weather_london.csv"))
    args = parser.parse_args()

    frame = fetch(args.start, args.end, args.lat, args.lon)
    if frame[["temp_c", "precip_mm"]].isna().any().any():
        print("[weather] warning: the response contains missing values", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[weather] wrote {out}: {len(frame)} days, "
          f"{frame['ds'].min().date()} to {frame['ds'].max().date()}, "
          f"mean {frame['temp_c'].mean():.1f} C")
    print("[weather] source: Open-Meteo historical reanalysis, CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
