import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import cdsapi
import xarray as xr


def poland_bbox_from_points(lats, lons, pad_deg=0.8):
    """
    Build a CDS bbox [north, west, south, east] with padding.
    """
    north = float(np.nanmax(lats)) + pad_deg
    south = float(np.nanmin(lats)) - pad_deg
    west = float(np.nanmin(lons)) - pad_deg
    east = float(np.nanmax(lons)) + pad_deg
    return [north, west, south, east]


def download_era5_land_monthly(year: int, area, out_nc: Path):
    """
    Download ERA5-Land monthly means for the bbox and year.
    Dataset: reanalysis-era5-land-monthly-means (monthly_averaged_reanalysis)
    """
    if out_nc.exists() and out_nc.stat().st_size > 0:
        print(f"Using cached NetCDF: {out_nc}")
        return out_nc

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    # Variables: keep it rail-relevant and simple.
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "total_precipitation",
            "snowfall",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "format": "netcdf",
        "area": area,  # [N, W, S, E]
    }

    print("Requesting ERA5-Land monthly means from CDS...")
    print("Area [N,W,S,E]:", area)
    c.retrieve("reanalysis-era5-land-monthly-means", request, str(out_nc))
    print("Downloaded:", out_nc)
    return out_nc


def extract_station_month_weather(nc_path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """
    For each station lat/lon, extract nearest grid cell for each month.

    ERA5-Land monthly means has time dimension (monthly) and lat/lon grid.
    """
    ds = xr.open_dataset(nc_path)

    # Normalize coordinate names (ERA5 commonly uses latitude/longitude)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    # Convert longitudes if needed (ERA5 usually is -180..180, but sometimes 0..360)
    # We'll align station lons to dataset convention.
    ds_lons = ds[lon_name].values
    lon_0_360 = np.nanmin(ds_lons) >= 0 and np.nanmax(ds_lons) > 180

    stations = stations.copy()
    if lon_0_360:
        stations["lon_ds"] = (stations["lon"] % 360.0)
    else:
        stations["lon_ds"] = stations["lon"]

    # Build results
    rows = []

    # Convert time to month int
    time_month = pd.to_datetime(ds["time"].values).month

    # Variables (units):
    # t2m: Kelvin -> Celsius
    # tp: meters of water -> mm
    # sf: meters of water equivalent (often) -> mm
    # u10/v10: m/s -> wind speed m/s
    for _, r in stations.iterrows():
        st = r["station"]
        lat = float(r["lat"])
        lon = float(r["lon_ds"])

        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

        t2m_c = (point["t2m"].values - 273.15).astype(float)
        tp_mm = (point["tp"].values * 1000.0).astype(float)
        sf_mm = (point["sf"].values * 1000.0).astype(float)
        u10 = point["u10"].values.astype(float)
        v10 = point["v10"].values.astype(float)
        wind_ms = np.sqrt(u10 ** 2 + v10 ** 2)

        for idx in range(len(time_month)):
            rows.append(
                {
                    "station": st,
                    "month": int(time_month[idx]),
                    "t2m_c": float(t2m_c[idx]),
                    "precip_mm": float(tp_mm[idx]),
                    "snowfall_mm": float(sf_mm[idx]),
                    "wind_ms": float(wind_ms[idx]),
                }
            )

    out = pd.DataFrame(rows)

    # Add “event-style” features for easier modeling
    out["freezing"] = (out["t2m_c"] <= 0).astype(int)
    out["heavy_rain"] = (out["precip_mm"] >= 50).astype(int)     # tweak thresholds later
    out["windy"] = (out["wind_ms"] >= 10).astype(int)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay_csv", default="outputs/train_delay_risk_dataset_2024.csv",
                    help="Your delay dataset (must include station, lat, lon).")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out_weather_csv", default="outputs/weather_era5_land_monthly_2024_by_station.csv")
    ap.add_argument("--cache_nc", default="cache/era5_land_monthly_2024_poland.nc")
    args = ap.parse_args()

    delay_path = Path(args.delay_csv)
    out_weather = Path(args.out_weather_csv)
    cache_nc = Path(args.cache_nc)

    df = pd.read_csv(delay_path)

    # unique stations with valid coords
    stations = (
        df[["station", "lat", "lon"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if stations.empty:
        raise SystemExit("No station lat/lon found. Ensure geocoding ran and delay CSV includes lat/lon.")

    area = poland_bbox_from_points(stations["lat"].values, stations["lon"].values, pad_deg=0.8)

    nc_path = download_era5_land_monthly(args.year, area, cache_nc)
    weather = extract_station_month_weather(nc_path, stations)

    out_weather.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_weather, index=False)
    print("Wrote:", out_weather)

    # quick sanity print
    print("\nWeather rows:", len(weather), "| Stations:", weather["station"].nunique(), "| Months:", weather["month"].nunique())
    print(weather.groupby("month")[["t2m_c", "precip_mm", "snowfall_mm", "wind_ms"]].mean().round(2))


if __name__ == "__main__":
    main()
