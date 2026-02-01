"""
build_station_cluster_table.py

Goal:
- Build a K-means clustering visualization of Polish (and cross-border) rail stations.
- Uses your existing outputs/train_delay_risk_dataset_2024.csv, which already includes:
  - delay_risk
  - total_stops
  - station lat/lon (from your geocoding pipeline)
  - monthly weather features (t2m_c, precip_mm, wind_ms, etc.)

What this script does:
1) Load the station-month dataset from your pipeline
2) Aggregate to ONE ROW PER STATION (features for clustering)
3) Fix missing coordinates by geocoding (cached to JSON)
4) Drop any stations still missing coords (Folium can't plot NaNs)
5) Run KMeans clustering on standardized station features
6) Render a Folium map with cluster colors

Outputs:
- outputs/station_kmeans_clusters.html
- cache/station_geocode_cache.json  (persistent cache)
"""

from pathlib import Path
import json
from time import sleep

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from geopy.geocoders import Nominatim
MANUAL_COORDS = {
    "Jagodin": (52.4, 20.7),  
}
# If a station name is ambiguous or cross-border, override the query string
ALIASES = {
    "Kralovec": "542 03 Královec, Czechia",
    "Mikulovice": "790 84 Mikulovice u Jeseníku 1, Czechia",
    "Plavec": "065 44 Plaveč, Slovakia",
    "Jagodin": "Jagodzin, Kieszków, Poland",  # from earlier
}

# =================================================
# SECTION 0 — Paths / Configuration
# =================================================

IN = Path("outputs/train_delay_risk_dataset_2024.csv")
OUT_MAP = Path("outputs/station_kmeans_clusters_map.html")

CACHE_PATH = Path("cache/station_geocode_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# K-means parameter (try 4–7 for different looks)
K = 5

# Be polite to Nominatim. 1 sec is safe.
GEOCODE_SLEEP_SEC = 1.0


# =================================================
# SECTION 1 — Load geocode cache
# =================================================

if CACHE_PATH.exists():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        GEO_CACHE = json.load(f)
    print(f"Loaded geocode cache with {len(GEO_CACHE)} entries")
else:
    GEO_CACHE = {}
    print("No geocode cache found; starting new cache.")


# =================================================
# SECTION 2 — Geocoder + cached geocoding function
# =================================================

geolocator = Nominatim(user_agent="polish_train_delay_geocoder")

def geocode_station(name: str):
    # 1) Manual lat/lon override (if you add it later)
    if name in MANUAL_COORDS:
        lat, lon = MANUAL_COORDS[name]
        GEO_CACHE[name] = [lat, lon]
        return lat, lon

    # 2) Cache
    if name in GEO_CACHE:
        lat, lon = GEO_CACHE[name]
        return lat, lon

    # 3) Alias -> better query
    query_name = ALIASES.get(name, name)

    # 4) Try Poland bias first ONLY if no alias was used
    queries = []
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



# =================================================
# SECTION 3 — Load input dataset
# =================================================

df = pd.read_csv(IN)
# df is station-month level (~92k rows)
# We will reduce to station-level (~3k rows) for clustering.


# =================================================
# SECTION 4 — Aggregate to station-level features
# =================================================

station = (
    df.groupby("station")
      .agg(
          # delay behavior
          delay_mean=("delay_risk", "mean"),
          delay_p90=("delay_risk", lambda x: x.quantile(0.9)),

          # operational scale proxy
          traffic=("total_stops", "mean"),

          # location (from your main pipeline)
          lat=("lat", "first"),
          lon=("lon", "first"),

          # weather summaries (monthly averaged -> station averaged)
          t2m_mean=("t2m_c", "mean"),
          precip_mean=("precip_mm", "mean"),
          wind_mean=("wind_ms", "mean"),

          # event-rate proxies (share of months flagged)
          freezing_rate=("freezing", "mean"),
          heavy_rain_rate=("heavy_rain", "mean"),
          windy_rate=("windy", "mean"),
      )
      .reset_index()
)

# Coerce lat/lon to numeric in case they came in as strings
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")


# =================================================
# SECTION 5 — Fix missing coordinates (before dropping!)
# =================================================

missing_mask = station[["lat", "lon"]].isna().any(axis=1)
missing_names = station.loc[missing_mask, "station"]

print(f"Stations missing coords: {len(missing_names)}")
if len(missing_names) > 0:
    print(missing_names.head(10))

    # IMPORTANT: iterate over the *missing subset of station*, not a dropped version.
    for idx in station.index[missing_mask]:
        name = station.at[idx, "station"]
        lat, lon = geocode_station(name)
        if lat is not None and lon is not None:
            station.at[idx, "lat"] = lat
            station.at[idx, "lon"] = lon
            print(f"Fixed: {name} -> {lat:.4f}, {lon:.4f}")
        sleep(GEOCODE_SLEEP_SEC)

# Recompute missing after the geocode attempt
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")
missing_after = station[["lat", "lon"]].isna().any(axis=1).sum()
print(f"Stations still missing coords after fix: {missing_after}")


# =================================================
# SECTION 6 — Drop unplottable stations (Folium can't handle NaNs)
# =================================================

before = len(station)
station = station.dropna(subset=["lat", "lon"]).copy()
after = len(station)
print(f"Plottable stations: {after}/{before} (dropped {before-after})")


# =================================================
# SECTION 7 — K-means clustering
# =================================================

features = [
    "delay_mean",
    "delay_p90",
    "traffic",
    "t2m_mean",
    "precip_mean",
    "wind_mean",
    "freezing_rate",
    "heavy_rain_rate",
    "windy_rate",
]

X = station[features].fillna(station[features].median())
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
station["cluster"] = kmeans.fit_predict(X_scaled)

print("Stations:", len(station))
print("Cluster counts:")
print(station["cluster"].value_counts().sort_index())


# =================================================
# SECTION 8 — Render Folium map (the visualization you want)
# =================================================

m = folium.Map(location=[52, 19], zoom_start=6, tiles="cartodbpositron")

colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

for _, r in station.iterrows():
    folium.CircleMarker(
        location=[float(r.lat), float(r.lon)],
        radius=4,
        color=colors[int(r.cluster) % len(colors)],
        fill=True,
        fill_opacity=0.75,
        popup=(
            f"<b>{r.station}</b><br>"
            f"Cluster: {int(r.cluster)}<br>"
            f"Avg delay risk: {r.delay_mean:.2f}<br>"
            f"Traffic (mean stops): {r.traffic:.1f}"
        ),
    ).add_to(m)


# =================================================
# SECTION 9 — Save cache + map outputs
# =================================================

with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(GEO_CACHE, f, ensure_ascii=False, indent=2)
print(f"Saved geocode cache with {len(GEO_CACHE)} entries")

OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
m.save(OUT_MAP)
print("Wrote map:", OUT_MAP)
