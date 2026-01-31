from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def extract_station_month_weather(nc_path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)

    # ERA5 usually uses these coord names
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    # Handle 0..360 lon grids if they appear
    ds_lons = ds[lon_name].values
    lon_0_360 = float(np.nanmin(ds_lons)) >= 0 and float(np.nanmax(ds_lons)) > 180

    st = stations.copy()
    if lon_0_360:
        st["lon_ds"] = st["lon"] % 360.0
    else:
        st["lon_ds"] = st["lon"]

    # ERA5 files sometimes use "valid_time" instead of "time"
    time_key = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
    if time_key is None:
        raise KeyError(f"No time coordinate found. coords={list(ds.coords)} vars={list(ds.variables)}")

    months = pd.to_datetime(ds[time_key].values).month

    rows = []
    for _, r in st.iterrows():
        station = r["station"]
        lat = float(r["lat"])
        lon = float(r["lon_ds"])

        # nearest grid cell for this station
        p = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

        # Variable names in ERA5-Land monthly file
        # t2m in K, tp/sf in meters (water equivalent), u10/v10 in m/s
        t2m_c = (p["t2m"].values - 273.15).astype(float)
        tp_mm = (p["tp"].values * 1000.0).astype(float)
        sf_mm = (p["sf"].values * 1000.0).astype(float)
        u10 = p["u10"].values.astype(float)
        v10 = p["v10"].values.astype(float)
        wind_ms = np.sqrt(u10**2 + v10**2)

        for i, m in enumerate(months):
            rows.append(
                {
                    "station": station,
                    "month": int(m),
                    "t2m_c": float(t2m_c[i]),
                    "precip_mm": float(tp_mm[i]),
                    "snowfall_mm": float(sf_mm[i]),
                    "wind_ms": float(wind_ms[i]),
                }
            )

    out = pd.DataFrame(rows)

    # Simple “event flags” (tune later)
    out["freezing"] = (out["t2m_c"] <= 0).astype(int)
    out["heavy_rain"] = (out["precip_mm"] >= 50).astype(int)
    out["windy"] = (out["wind_ms"] >= 10).astype(int)

    return out


def main():
    nc_path = Path("cache/era5_land_monthly_2024_poland.nc")
    delay_csv = Path("outputs/train_delay_risk_dataset_2024.csv")
    out_csv = Path("outputs/weather_era5_land_monthly_2024_by_station.csv")

    if not nc_path.exists():
        raise SystemExit(f"Missing NetCDF: {nc_path}")

    df = pd.read_csv(delay_csv)
    stations = (
        df[["station", "lat", "lon"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if stations.empty:
        raise SystemExit("No station lat/lon found in delay CSV. Make sure geocoding ran.")

    weather = extract_station_month_weather(nc_path, stations)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)
    print("Rows:", len(weather), "| Stations:", weather["station"].nunique(), "| Months:", weather["month"].nunique())
    print(weather.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
