import datetime
import tempfile
from typing import Tuple

# Not used directly, but used via xarray
import cfgrib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import haversine_distances
import xarray as xr


def create_url_and_get_xr(day: datetime.date, cycle: int, leadtime: int):
    # Constants for creating the full URL
    blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
    sector = "conus"
    product = "wrfsfcf"  # 2D surface levels

    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{leadtime:02}.grib2"
    url = f"{blob_container}/hrrr.{day:%Y%m%d}/{sector}/{file_path}"
    r = requests.get(f"{url}.idx")
    idx = r.text.splitlines()

    smk_line = [l for l in idx if "MASSDEN" in l][0].split(
        ":"
    )  # [l for l in idx if "COLMD" in l][0].split(":")
    line_num = int(smk_line[0])
    range_start = smk_line[1]
    next_line = idx[line_num].split(":") if line_num < len(idx) else None
    range_end = next_line[1] if next_line else None

    file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)

    headers = {"Range": f"bytes={range_start}-{range_end}"}
    resp = requests.get(url, headers=headers, stream=True)

    with file as f:
        f.write(resp.content)

    ds = xr.open_dataset(file.name, engine="cfgrib")

    return ds


def hrrr_data_download(date: datetime.date, cycle: int = 0) -> pd.DataFrame:
    def _download_hour(leadtime: int):
        ds = create_url_and_get_xr(date, cycle, leadtime)
        # Rescale entry of mdens to micrograms/m^2
        mdens_rescaled = ds.mdens * 1e9
        # Get the latitude and longitude values
        lats = ds.latitude.values
        lons = ds.longitude.values
        # Create a pandas dataframe with latitude, longitude, time, and the mden values
        hrrr_df = pd.DataFrame(
            {
                "Latitude": lats.ravel(),
                "Longitude": lons.ravel(),
                "ValidTime": f"{cycle + leadtime}",
                "PM25": np.round(mdens_rescaled.values.ravel(), 2),
            }
        )
        return hrrr_df

    hourly_dfs = [_download_hour(leadtime) for leadtime in range(1, 25)]
    return pd.concat(hourly_dfs)


def find_nearby_predictions(
    predictions: pd.DataFrame,
    coordinates: Tuple[float, float],
    max_distance: float = 10.0,
) -> pd.DataFrame:
    """
    Return the average hourly air quality at AQS monitors near coordinates.
    """
    pred_lats = np.radians(predictions["Latitude"])
    pred_lons = np.radians(predictions["Longitude"])
    pred_latlons = np.stack([pred_lats, pred_lons], axis=-1)
    coords = np.radians(coordinates)[None, :]
    dists = 6371 * haversine_distances(coords, pred_latlons)[0]
    inds = (dists <= max_distance).nonzero()[0]
    nearby_preds = predictions.iloc[inds]
    return nearby_preds
