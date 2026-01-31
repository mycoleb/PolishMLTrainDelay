from pathlib import Path
import cdsapi

def main():
    dataset = "reanalysis-era5-land-monthly-means"

    request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "total_precipitation",
        "snowfall",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "year": ["2024"],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [55, 14, 49, 25],   # [N, W, S, E]
}



    out_path = Path("cache/era5_land_monthly_2024_poland.nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Using cached file: {out_path}")
        return

    print("Requesting ERA5-Land monthly means...")
    print("Output:", out_path)

    client = cdsapi.Client()
    client.retrieve(dataset, request, str(out_path))

    print("Done:", out_path)

if __name__ == "__main__":
    main()
