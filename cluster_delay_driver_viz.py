"""
cluster_delay_driver_viz.py

Goal:
Visualize which weather/operational/geographic factors most differentiate delay behavior
using K-means clustering (unsupervised).

What you get (outputs/):
- station_clusters_driver_map.html
  A map of stations colored by cluster. Popups include delay + key features.

- cluster_factor_importance.png
  Bar chart ranking features by:
    (A) between-cluster separation (effect size-ish)
    (B) correlation with delay_mean
  (combined score)

- cluster_profiles_parallel.png
  Parallel coordinates plot: cluster "profiles" across standardized features.

- cluster_driver_report.html
  A simple HTML report that embeds the PNGs and links the map.

Inputs:
- outputs/train_delay_risk_dataset_2024.csv   (from p.py pipeline)

Notes:
- Clustering does NOT "prove causation".
- But it’s great for discovering station archetypes and seeing which features move together
  with high/low delay clusters.
"""

from __future__ import annotations

from pathlib import Path
import json
from time import sleep

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import folium
from geopy.geocoders import Nominatim


# ============================================================
# SECTION 0 — Configuration (edit these first)
# ============================================================

IN = Path("outputs/train_delay_risk_dataset_2024.csv")

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MAP = Path("outputs/station_kmeans_clusters_map.html")
# Add this line:
out_parallel = OUT_DIR / "cluster_profiles_parallel.png"

CACHE_PATH = Path("cache/station_geocode_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Try K = 4..7; if you see a singleton cluster, reduce K a bit.
K = 5

# Be polite to Nominatim.
GEOCODE_SLEEP_SEC = 1.0

# If a station name is ambiguous or cross-border, override the query string:
ALIASES = {
    "Kralovec": "542 03 Královec, Czechia",
    "Mikulovice": "790 84 Mikulovice u Jeseníku 1, Czechia",
    "Plavec": "065 44 Plaveč, Slovakia",
    "Jagodin": "Jagodzin, Kieszków, Poland",
}

# If you know exact lat/lon for a few tricky stations, put them here:
MANUAL_COORDS: dict[str, tuple[float, float]] = {
    # "Jagodin": (52.4, 20.7),  # example (replace with exact if you want)
}


# ============================================================
# SECTION 1 — Load geocode cache + create geocoder
# ============================================================

if CACHE_PATH.exists():
    GEO_CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded geocode cache with {len(GEO_CACHE)} entries")
else:
    GEO_CACHE = {}
    print("No geocode cache found; starting new cache.")

geolocator = Nominatim(user_agent="polish_train_delay_driver_viz")


def geocode_station(name: str) -> tuple[float | None, float | None]:
    """
    Cached geocoding:
    0) manual override
    1) cache
    2) alias query (if present)
    3) try Poland-biased (if not aliased), then fallback to raw
    """
    if name in MANUAL_COORDS:
        lat, lon = MANUAL_COORDS[name]
        GEO_CACHE[name] = [lat, lon]
        return lat, lon

    if name in GEO_CACHE:
        lat, lon = GEO_CACHE[name]
        return lat, lon

    query_name = ALIASES.get(name, name)
    queries: list[str] = []

    if name not in ALIASES:
        queries.append(f"{query_name}, Poland")
    queries.append(query_name)

    for q in queries:
        try:
            loc = geolocator.geocode(q, timeout=10)
            if loc:
                lat, lon = float(loc.latitude), float(loc.longitude)
                GEO_CACHE[name] = [lat, lon]
                return lat, lon
        except Exception:
            pass

    GEO_CACHE[name] = [None, None]
    return None, None


# ============================================================
# SECTION 2 — Load station-month dataset and aggregate to station-level
# ============================================================

df = pd.read_csv(IN)

# Station-level features: one row per station
station = (
    df.groupby("station")
      .agg(
          # Delay behavior
          delay_mean=("delay_risk", "mean"),
          delay_p90=("delay_risk", lambda x: x.quantile(0.9)),

          # Operational proxy
          traffic=("total_stops", "mean"),

          # Coordinates
          lat=("lat", "first"),
          lon=("lon", "first"),

          # Weather summaries (already station-month in df; we average across months)
          t2m_mean=("t2m_c", "mean"),
          precip_mean=("precip_mm", "mean"),
          snowfall_mean=("snowfall_mm", "mean"),
          wind_mean=("wind_ms", "mean"),

          # Flag rates: share of months flagged (0..1)
          freezing_rate=("freezing", "mean"),
          heavy_rain_rate=("heavy_rain", "mean"),
          windy_rate=("windy", "mean"),
      )
      .reset_index()
)

# Ensure coords are numeric if they came in as strings
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")


# ============================================================
# SECTION 3 — Fix missing coordinates (only for missing rows)
# ============================================================

missing_mask = station[["lat", "lon"]].isna().any(axis=1)
missing_names = station.loc[missing_mask, "station"]

print(f"Stations missing coords: {len(missing_names)}")
if len(missing_names) > 0:
    print(missing_names.head(10).to_string(index=False))
    for idx in station.index[missing_mask]:
        name = station.at[idx, "station"]
        lat, lon = geocode_station(name)
        if lat is not None and lon is not None:
            station.at[idx, "lat"] = lat
            station.at[idx, "lon"] = lon
            print(f"Fixed: {name} -> {lat:.4f}, {lon:.4f}")
        sleep(GEOCODE_SLEEP_SEC)

# Drop any stations still missing coords (Folium can’t plot NaNs)
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")

before = len(station)
station = station.dropna(subset=["lat", "lon"]).copy()
after = len(station)
print(f"Plottable stations: {after}/{before} (dropped {before-after})")


# ============================================================
# SECTION 4 — K-means clustering
# ============================================================

# These are the features that define station "type"
cluster_features = [
    "delay_mean",
    "delay_p90",
    "traffic",
    "t2m_mean",
    "precip_mean",
    "snowfall_mean",
    "wind_mean",
    "freezing_rate",
    "heavy_rain_rate",
    "windy_rate",
]

X = station[cluster_features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
station["cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster counts:")
print(station["cluster"].value_counts().sort_index())

# ... existing kmeans code ...
station["cluster"] = kmeans.fit_predict(X_scaled)

# NEW: Save the cluster labels to the CSV so o.py can find them
# We merge the clusters back to the original full dataset
df_with_clusters = df.merge(station[['station', 'cluster']], on='station', how='left')
df_with_clusters.to_csv(IN, index=False) 
print(f"Updated {IN} with cluster labels.")
# ============================================================
# SECTION 5 — "Which factors matter?" scoring (cluster separation + delay correlation)
# ============================================================

# A) Between-cluster separation score:
#    For each feature, compute variance of cluster means / overall variance.
#    Higher means feature strongly differentiates clusters.
cluster_means = station.groupby("cluster")[cluster_features].mean()
overall_var = station[cluster_features].var(ddof=0).replace(0, np.nan)
between_var = cluster_means.var(ddof=0)

sep_score = (between_var / overall_var).fillna(0.0)

# B) Correlation with delay_mean (absolute value)
corr = station[cluster_features].corr(numeric_only=True)["delay_mean"].abs().fillna(0.0)

# Combine (normalize both to 0..1 then average)
def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return s * 0.0
    return (s - mn) / (mx - mn)

sep_n = minmax(sep_score)
corr_n = minmax(corr)
combined = 0.5 * sep_n + 0.5 * corr_n

rank = pd.DataFrame({
    "feature": cluster_features,
    "sep_score": sep_score.values,
    "abs_corr_with_delay_mean": corr.values,
    "combined_score": combined.values
}).sort_values("combined_score", ascending=False)

# Print a readable ranking to terminal (not a big table, just top lines)
print("\nTop features by combined clustering+delay score:")
for _, row in rank.head(10).iterrows():
    print(
        f" - {row['feature']}: combined={row['combined_score']:.3f} "
        f"(sep={row['sep_score']:.3f}, |corr|={row['abs_corr_with_delay_mean']:.3f})"
    )


# ============================================================
# SECTION 6 — Visual 1: Feature importance bar chart
# ============================================================

out_bar = OUT_DIR / "cluster_factor_importance.png"

plt.figure()
plt.bar(rank["feature"], rank["combined_score"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Combined score (cluster separation + delay correlation)")
plt.title("Which factors most differentiate station delay behavior? (K-means)")
plt.tight_layout()
plt.savefig(out_bar, dpi=200)
plt.close()
print("Wrote:", out_bar)

# ============================================================
# SECTION 7 — Visual 2: Parallel coordinates (cluster profiles)
# ============================================================

# Build standardized feature frame AND add cluster BEFORE groupby
Z = pd.DataFrame(X_scaled, columns=cluster_features)
Z["cluster"] = station["cluster"].astype(int).to_numpy()

sample_per_cluster = 150
Zs = (
    Z.groupby("cluster", group_keys=False)
     .apply(lambda g: g.sample(n=min(sample_per_cluster, len(g)), random_state=42))
     .reset_index(drop=True)
)

out_parallel = OUT_DIR / "cluster_profiles_parallel.png"

plt.figure(figsize=(12, 6))
parallel_coordinates(Zs, "cluster", alpha=0.25)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Standardized value (z-score)")
plt.title(f"Station Cluster Profiles (Standardized Features, K={K})")
plt.tight_layout()
plt.savefig(out_parallel, dpi=200)
plt.close()
print("Wrote:", out_parallel)

# ============================================================
# SECTION 8 — Visual 3: Cluster map (Folium HTML)
# ============================================================

out_map = OUT_DIR / "station_clusters_driver_map.html"
m = folium.Map(location=[52, 19], zoom_start=6, tiles="cartodbpositron")

colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

# Show top 3 features in popup so you can "see drivers"
top3 = rank["feature"].head(3).tolist()

for _, r in station.iterrows():
    popup_bits = [
        f"<b>{r.station}</b>",
        f"Cluster: {int(r.cluster)}",
        f"Avg delay risk: {r.delay_mean:.3f}",
        f"90th pct delay: {r.delay_p90:.3f}",
        f"Traffic (mean stops): {r.traffic:.1f}",
        "<hr>",
        f"Top drivers shown: {', '.join(top3)}",
    ]
    for f in top3:
        popup_bits.append(f"{f}: {float(r[f]):.3f}")

    folium.CircleMarker(
        location=[float(r.lat), float(r.lon)],
        radius=4,
        color=colors[int(r.cluster) % len(colors)],
        fill=True,
        fill_opacity=0.75,
        popup="<br>".join(popup_bits),
    ).add_to(m)

m.save(out_map)
print("Wrote:", out_map)


# ============================================================
# SECTION 9 — Save cache + a simple HTML report
# ============================================================

CACHE_PATH.write_text(json.dumps(GEO_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved geocode cache with {len(GEO_CACHE)} entries")

out_report = OUT_DIR / "cluster_driver_report.html"

report_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Polish Train Delays — K-means Driver Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    img {{ max-width: 1100px; width: 100%; height: auto; border: 1px solid #ddd; padding: 6px; }}
    .small {{ color: #555; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Polish Train Delays — “What factors matter?” via K-means</h1>
  <p class="small">
    This is an exploratory (unsupervised) view. Clusters show station archetypes, and the charts show
    which features most differentiate those archetypes and align with higher delay risk.
  </p>

  <h2>1) Feature importance (cluster separation + delay correlation)</h2>
  <img src="{out_bar.name}" alt="Feature importance"/>

  <h2>2) Cluster profiles (parallel coordinates, standardized)</h2>
  <img src="{out_parallel.name}" alt="Parallel coordinates"/>

  <h2>3) Cluster map</h2>
  <p>Open the interactive map: <code>{out_map.name}</code></p>

  <h2>Top features (text)</h2>
  <pre>{rank.head(12).to_string(index=False)}</pre>
</body>
</html>
"""

out_report.write_text(report_html, encoding="utf-8")
print("Wrote:", out_report)

print("\nDone. Open these:")
print(" -", out_report)
print(" -", out_map)


if __name__ == "__main__":
    # This file is intended to be run directly:
    #   python cluster_delay_driver_viz.py
    pass
