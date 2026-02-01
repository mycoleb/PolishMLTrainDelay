import csv
import re


import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ML
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# Mapping
import folium

# Geocoding (OSM Nominatim) - be polite: cache + rate limit
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUT_DIR = BASE_DIR / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# These are direct file endpoints for 2024 station-level tables
# (If an endpoint changes, update the IDs. Keep the structure.)
URL_TOTAL_STOPS_2024 = "https://api.dane.gov.pl/resources/67808%2Cliczba-zatrzyman-pociagow-pasazerskich-w-2024-r/file"
URL_DELAY_MINUTES_2024 = "https://api.dane.gov.pl/resources/67803%2Cczas-opoznien-na-stacjach-pociagow-pasazerskich-w-minutach-w-2024-r/file"
URL_DELAYED_STOPS_2024 = "https://api.dane.gov.pl/resources/67804%2Cliczba-opoznionych-zatrzyman-pociagow-w-2024-r-przewozy-pasazerskie-i-przewozy-towarowe/file"

MONTHS_PL = [
    "styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
    "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"
]


@dataclass
class TableSpec:
    url: str
    value_name: str  # column name after melt


def _download_csv(url: str, dst: Path) -> Path:
    """
    Download a CSV once and reuse cached file.
    Handles Polish encodings (cp1250 / ISO-8859-2).
    """
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    print(f"Downloading: {url}")

    encodings_to_try = ["utf-8", "cp1250", "iso-8859-2"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                url,
                sep=";",
                encoding=enc,
                engine="python",
                dtype=str,
            )
            print(f"  Loaded with encoding: {enc}")
            df.to_csv(dst, index=False)
            return dst
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "Could not decode CSV with known Polish encodings",
        b"",
        0,
        1,
        str(last_error),
    )




def _sniff_header_row(path: Path, encoding: str = "utf-8", sep: str = ";", max_lines: int = 60) -> int:
    """
    Return the row index (0-based) that looks like a real header.
    We search for a row containing something like 'stacja' and/or 'przewoźnik'
    OR month columns.
    """
    wanted = ["stacja", "przew", "przewoź", "przewoz"]  # carrier variations
    month_tokens = ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
                    "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"]

    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        for i, row in enumerate(reader):
            if i >= max_lines:
                break
            row_l = [str(x).strip().lower() for x in row if str(x).strip() != ""]
            if not row_l:
                continue

            joined = " ".join(row_l)

            # header if it contains station/carrier keywords
            if any(k in joined for k in wanted):
                return i

            # or header if it contains at least 3 month names
            month_hits = sum(1 for m in month_tokens if m in joined)
            if month_hits >= 3:
                return i

    return 0  # fallback


def _normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\ufeff", "")  # BOM
    # strip Polish diacritics (so "Przewoźnik" matches "przewoznik")
    s = (s.replace("ł", "l").replace("ó", "o").replace("ś", "s").replace("ń", "n")
           .replace("ż", "z").replace("ź", "z").replace("ć", "c").replace("ę", "e").replace("ą", "a"))
    s = re.sub(r"\s+", " ", s)
    return s
df["station"] = df["station"].replace({
    "Jagodin": "Jagodzin",
})

def _read_station_month_table(local_csv: Path, value_name: str) -> pd.DataFrame:
    """
    Robust reader for dane.gov.pl rail punctuality tables that may include title rows.
    Reads cached CSV (already saved locally by _download_csv).
    """
    header_row = _sniff_header_row(local_csv, encoding="utf-8", sep=";", max_lines=80)

    df = pd.read_csv(local_csv, dtype=str, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop "Unnamed" columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]

    # Map normalized name -> original
    norm_map = {_normalize_colname(c): c for c in df.columns}

    # Find station + carrier columns
    station_candidates = ["stacja", "nazwa stacji", "nazwa", "stacja handlowa"]
    carrier_candidates = ["przewoznik", "przewoznik kolejowy", "przewoźnik", "operator", "spolka"]

    station_col = None
    for k in norm_map.keys():
        if any(sc == k or sc in k for sc in station_candidates):
            station_col = norm_map[k]
            break

    carrier_col = None
    for k in norm_map.keys():
        if any(cc == k or cc in k for cc in carrier_candidates):
            carrier_col = norm_map[k]
            break

    if station_col is None or carrier_col is None:
        raise ValueError(
            f"Could not find station/carrier columns in {local_csv.name}. "
            f"Columns: {df.columns.tolist()}"
        )

    # Month columns: Polish names OR numeric 1..12 / 01..12
    month_cols = [m for m in MONTHS_PL if m in df.columns]
    if not month_cols:
        month_cols = [c for c in df.columns if re.fullmatch(r"(0?[1-9]|1[0-2])", str(c).strip())]
        if len(month_cols) < 6:
            raise ValueError(
                f"No month columns detected in {local_csv.name}. "
                f"Columns: {df.columns.tolist()}"
            )

    out = df[[station_col, carrier_col] + month_cols].copy()
    out = out.melt(
        id_vars=[station_col, carrier_col],
        value_vars=month_cols,
        var_name="month_raw",
        value_name=value_name,
    )
    out.rename(columns={station_col: "station", carrier_col: "carrier"}, inplace=True)

    # Parse month number
    if all(m in MONTHS_PL for m in month_cols):
        month_map = {m: i + 1 for i, m in enumerate(MONTHS_PL)}
        out["month"] = out["month_raw"].map(month_map).astype(int)
    else:
        out["month"] = (
            out["month_raw"].astype(str).str.strip().str.lstrip("0").replace("", "0")
        )
        out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(0).astype(int)

    # Clean numeric values
    out[value_name] = (
        out[value_name]
        .fillna("0")
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce").fillna(0.0)

    # Clean strings
    out["station"] = out["station"].astype(str).str.strip()
    out["carrier"] = out["carrier"].astype(str).str.strip()

    return out.drop(columns=["month_raw"])



def build_dataset_2024() -> pd.DataFrame:
    specs = [
        TableSpec(URL_TOTAL_STOPS_2024, "total_stops"),
        TableSpec(URL_DELAYED_STOPS_2024, "delayed_stops"),
        TableSpec(URL_DELAY_MINUTES_2024, "delay_minutes"),
    ]

    frames = []
    for spec in specs:
        local = CACHE_DIR / (spec.value_name + "_2024.csv")
        _download_csv(spec.url, local)
        frames.append(_read_station_month_table(local, spec.value_name))

    # Merge on (station, carrier, month)
    df = frames[0]
    for nxt in frames[1:]:
        df = df.merge(nxt, on=["station", "carrier", "month"], how="outer")

    # Fill missing numeric values with 0
    for c in ["total_stops", "delayed_stops", "delay_minutes"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # Targets / derived metrics
    df["delay_risk"] = np.where(df["total_stops"] > 0, df["delayed_stops"] / df["total_stops"], 0.0)
    df["delay_severity_min"] = np.where(df["delayed_stops"] > 0, df["delay_minutes"] / df["delayed_stops"], 0.0)

    # Clip risk to [0, 1] just in case
    df["delay_risk"] = df["delay_risk"].clip(0, 1)

    return df


def geocode_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode unique station names -> lat/lon using OSM Nominatim, cached locally.
    This is the slowest step the first time.
    """
    cache_file = CACHE_DIR / "station_geocode_cache.csv"
    cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    if cache_file.exists():
        cdf = pd.read_csv(cache_file)
        for _, r in cdf.iterrows():
            cache[str(r["station"])] = (float(r["lat"]) if pd.notna(r["lat"]) else None,
                                        float(r["lon"]) if pd.notna(r["lon"]) else None)

    geolocator = Nominatim(user_agent="polish-delay-risk-map")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)  # be polite

    stations = sorted(df["station"].dropna().unique().tolist())
    new_rows = []

    for s in stations:
        if s in cache:
            continue

        query = f"{s}, Polska"
        loc = None
        try:
            loc = geocode(query)
        except Exception:
            loc = None

        lat = float(loc.latitude) if loc else None
        lon = float(loc.longitude) if loc else None
        cache[s] = (lat, lon)
        new_rows.append({"station": s, "lat": lat, "lon": lon})
        print(f"Geocoded: {s} -> {lat}, {lon}")
        # RateLimiter already sleeps; extra sleep not necessary

    # Write updated cache
    out_cache = pd.DataFrame([{"station": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()])
    out_cache.to_csv(cache_file, index=False)

    geo = out_cache
    out = df.merge(geo, on="station", how="left")

    # Drop stations we couldn't geocode (optional: keep but they won't map)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    return out


def train_risk_model(df: pd.DataFrame, use_weather: bool = False) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train a model to predict delay_risk from features.
    If use_weather=True, include ERA5 station-month weather features.
    """
    base_features = ["month", "carrier", "station", "lat", "lon", "total_stops"]
    weather_features = ["t2m_c", "precip_mm", "snowfall_mm", "wind_ms", "freezing", "heavy_rain", "windy"]

    df_model = df.dropna(subset=["delay_risk"]).copy()

    # Fill missing coords (needed for both models)
    for c in ["lat", "lon"]:
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")
        df_model[c] = df_model[c].fillna(df_model[c].median())

    features = base_features
    numeric = ["month", "lat", "lon", "total_stops"]
    categorical = ["carrier", "station"]

    if use_weather:
        # Ensure weather columns exist; if not, fail loudly with a helpful message
        missing = [c for c in weather_features if c not in df_model.columns]
        if missing:
            raise KeyError(f"use_weather=True but missing weather columns: {missing}")

        # Fill missing weather values
        for c in weather_features:
            df_model[c] = pd.to_numeric(df_model[c], errors="coerce")
            df_model[c] = df_model[c].fillna(df_model[c].median())

        features = base_features + weather_features
        numeric = numeric + weather_features

    X = df_model[features]
    y = df_model["delay_risk"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense output overall
    )

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=42
    )

    pipe = Pipeline([("prep", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test).clip(0, 1)

    print("\nModel quality (holdout):")
    print("  MAE:", round(mean_absolute_error(y_test, preds), 4))
    print("  R2 :", round(r2_score(y_test, preds), 4))

    return pipe, df_model


def print_delay_risk_summary(df_model: pd.DataFrame):
    print("\n" + "=" * 70)
    print("POLISH TRAIN DELAY RISK — SUMMARY")
    print("=" * 70)

    # Overall stats
    print("\nOVERALL DELAY RISK (all stations, all months)")
    print("-" * 70)
    print(df_model["delay_risk"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

    # Worst stations (by average risk)
    print("\nTOP 10 WORST STATIONS (highest avg delay risk)")
    print("-" * 70)
    worst = (
        df_model.groupby("station")["delay_risk"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print(worst.round(3))

    # Best stations
    print("\nTOP 10 BEST STATIONS (lowest avg delay risk)")
    print("-" * 70)
    best = (
        df_model.groupby("station")["delay_risk"]
        .mean()
        .sort_values()
        .head(10)
    )
    print(best.round(3))

    # By carrier
    print("\nAVERAGE DELAY RISK BY CARRIER")
    print("-" * 70)
    carrier = (
        df_model.groupby("carrier")["delay_risk"]
        .mean()
        .sort_values(ascending=False)
    )
    print(carrier.round(3))

    print("\nINTERPRETATION GUIDE")
    print("-" * 70)
    print("0.00–0.05  → very reliable")
    print("0.05–0.15  → mostly reliable")
    print("0.15–0.30  → noticeable delays")
    print("0.30+      → high delay risk")

def make_risk_map(df_model: pd.DataFrame, pipe: Pipeline, predict_month: int = 12) -> Path:
    """
    Predict risk for a chosen month and plot stations.
    """
    # Choose a month to "forecast" using same stations/carriers; set month=predict_month
    df_pred = (
        df_model.groupby(["station", "carrier"], as_index=False)
        .agg({"lat": "median", "lon": "median", "total_stops": "mean"})
    )
    df_pred["month"] = int(predict_month)
    # Add month-level weather from df_model (median per station for the chosen month)
    wx_cols = ["t2m_c","precip_mm","snowfall_mm","wind_ms","freezing","heavy_rain","windy"]
    wx_month = (
        df_model[df_model["month"] == int(predict_month)]
        .groupby("station", as_index=False)[wx_cols]
        .median()
    )

    df_pred = df_pred.merge(wx_month, on="station", how="left")

    # Fill any missing weather
    # Fill any missing weather in df_pred using df_model medians
    for c in wx_cols:
        df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")
        df_pred[c] = df_pred[c].fillna(df_model[c].median())


    feature_cols = [
        "month","carrier","station","lat","lon","total_stops",
        "t2m_c","precip_mm","snowfall_mm","wind_ms",
        "freezing","heavy_rain","windy",
    ]
    preds = pipe.predict(df_pred[feature_cols]).clip(0, 1)
    df_pred["pred_delay_risk"] = preds

    # Center map on Poland
    m = folium.Map(location=[52.1, 19.4], zoom_start=6, tiles="cartodbpositron")

    def color_for_risk(r: float) -> str:
        # Simple discrete palette (no external libs)
        if r < 0.05: return "#2c7bb6"
        if r < 0.10: return "#00a6ca"
        if r < 0.20: return "#00ccbc"
        if r < 0.30: return "#90eb9d"
        if r < 0.40: return "#ffff8c"
        if r < 0.50: return "#f9d057"
        if r < 0.60: return "#f29e2e"
        if r < 0.70: return "#e76818"
        if r < 0.80: return "#d7191c"
        return "#8b0000"

    # Plot only rows with valid coordinates
    for c in ["t2m_c","precip_mm","snowfall_mm","wind_ms","freezing","heavy_rain","windy"]:
        df_model[c] = df_model[c].fillna(df_model[c].median())


    df_plot = df_pred.dropna(subset=["lat", "lon"]).copy()

    for _, r in df_plot.iterrows():
        risk = float(r["pred_delay_risk"])
        radius = 3 + 12 * risk  # scale bubble size
        popup = folium.Popup(
            f"<b>{r['station']}</b><br>"
            f"Carrier: {r['carrier']}<br>"
            f"Pred. delay risk: {risk:.1%}<br>"
            f"(month={predict_month})",
            max_width=320
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=radius,
            color=color_for_risk(risk),
            fill=True,
            fill_opacity=0.75,
            popup=popup,
        ).add_to(m)

    out_html = OUT_DIR / "risk_map.html"
    m.save(str(out_html))
    print(f"\nWrote map: {out_html}")
    return out_html

def print_summary_of_findings(df_model: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUMMARY OF FINDINGS — POLISH TRAIN DELAY RISK (2024)")
    print("=" * 72)

    overall = df_model["delay_risk"]

    print("\nOVERALL RELIABILITY")
    print("-" * 72)
    print(f"Average delay risk: {overall.mean():.2%}")
    print(f"Median delay risk : {overall.median():.2%}")
    print(f"90th percentile   : {overall.quantile(0.9):.2%}")

    print("\nINTERPRETATION:")
    print("• Most stations delay ~1 in 10 trains")
    print("• Worst 10% of stations delay ~1 in 4 trains or more")

    print("\nCARRIERS WITH HIGHEST AVERAGE DELAY RISK")
    print("-" * 72)
    print(
        df_model.groupby("carrier")["delay_risk"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(3)
    )

    print("\nSTATIONS WITH HIGHEST AVERAGE DELAY RISK")
    print("-" * 72)
    print(
        df_model.groupby("station")["delay_risk"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(3)
    )

    print("\nBOTTOM LINE")
    print("-" * 72)
    print(
        "Delay risk is unevenly distributed across Poland.\n"
        "Station location, carrier, and traffic volume together explain\n"
        "a substantial share of delay behavior, indicating structural\n"
        "rather than random causes."
    )

def main():
    print("Building dataset (2024) from dane.gov.pl API files...")
    df = build_dataset_2024()
    print("Rows:", len(df), "| Stations:", df["station"].nunique(), "| Carriers:", df["carrier"].nunique())

    print("\nGeocoding stations (cached)...")
    df_geo = geocode_stations(df)

    # --- load & merge weather ---
    weather_path = OUT_DIR / "weather_era5_land_monthly_2024_by_station.csv"
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        df_full = df_geo.merge(weather, on=["station", "month"], how="left")
        print("Weather match rate:", df_full["t2m_c"].notna().mean())
    else:
        print("WARNING: weather file not found:", weather_path)
        df_full = df_geo

    # --- baseline model (no weather) ---
    print("\nTraining baseline model (no weather)...")
    pipe_base, df_model_base = train_risk_model(df_geo, use_weather=False)

    # --- weather model (with weather) ---
    print("\nTraining weather model (with weather)...")
    pipe_wx, df_model_wx = train_risk_model(df_full, use_weather=True)

    print("\nGenerating map (weather model)...")
    make_risk_map(df_model_wx, pipe_wx, predict_month=12)

    # Summaries for the weather model
    print_delay_risk_summary(df_model_wx)
    print_summary_of_findings(df_model_wx)

    # Save merged modeling table for your repo
    out_csv = OUT_DIR / "train_delay_risk_dataset_2024.csv"
    df_full.to_csv(out_csv, index=False)
    print("Wrote dataset:", out_csv)

if __name__ == "__main__":
    main()
