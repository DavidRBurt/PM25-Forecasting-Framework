import datetime
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Union
from urllib.error import HTTPError

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def airnow_data_download(date: datetime.date, data_directory: Path) -> pd.DataFrame:
    os.makedirs(str(data_directory), exist_ok=True)
    return parse_day(data_directory, date)


def parse_day(datadir: str, date: datetime.date) -> Union[pd.DataFrame, None]:
    success = download_day(datadir, date)
    if success:
        df = open_day(datadir, date)
        clean_up(datadir, date)
    else:
        print(f"Was not able to download all hours for {date}")
        df = None
    return df


def download_day(datadir: str, date: datetime.date):
    for hour in range(4, 24):
        success = download_hour(datadir, date, hour)
        if not success:
            return False
    for hour in range(0, 18):
        tomorrow = date + datetime.timedelta(days=1)
        success = download_hour(datadir, tomorrow, hour)
        if not success:
            return False
    return True


def download_hour(datadir: str, date: datetime.date, hour: int) -> str:
    url = f"https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/{date.year:04}/{date.year:04}{date.month:02}{date.day:02}/HourlyAQObs_{date.year:04}{date.month:02}{date.day:02}{hour:02}.dat"
    filename = str(Path(datadir, f"{date}-{hour:02}.dat"))
    try:
        urllib.request.urlretrieve(url, filename)
    except HTTPError as e:
        print(f"Could not download data for {date}. Could this be in the future?")
        return False
    return True


def open_day(datadir: str, date: datetime.date):
    hour_dfs1 = [
        open_hour_df(str(Path(datadir, f"{date}-{hour:02}.dat")))
        for hour in range(4, 24)
    ]
    tomorrow = date + datetime.timedelta(days=1)
    hour_dfs2 = [
        open_hour_df(str(Path(datadir, f"{tomorrow}-{hour:02}.dat")))
        for hour in range(0, 18)
    ]
    for hour in hour_dfs2:
        h = int(hour.ValidTime.unique()[0])
        h += 24
        hour.ValidTime = h
    hour_dfs = hour_dfs1 + hour_dfs2
    day_df = pd.concat(hour_dfs)
    return day_df


def open_hour_df(filename: str):
    hour_df = pd.read_csv(filename)
    hour_df = hour_df.dropna(subset=["PM25"])
    hour_df = hour_df[
        ["AQSID", "Latitude", "Longitude", "ValidTime", "PM25", "PM25_Unit"]
    ]
    # Check unit then get rid of it to save memory
    assert len(hour_df.PM25_Unit.unique()) == 1
    assert hour_df.PM25_Unit.unique()[0] == "UG/M3"
    # Convert valid time to integer
    hour_df["ValidTime"] = hour_df["ValidTime"].str[:2].astype(int)
    hour_df = hour_df[["AQSID", "Latitude", "Longitude", "ValidTime", "PM25"]]
    # Continental US (very roughly)
    hour_df = filter_conus(hour_df)
    return hour_df


def filter_conus(df: pd.DataFrame) -> pd.DataFrame:
    # Define latitude and longitude ranges for the continental US
    lat_range = (24.396308, 49.384358)
    lon_range = (-125.0, -66.93457)

    # Filter the DataFrame based on latitude and longitude ranges
    filtered_df = df[
        (df["Latitude"] >= lat_range[0])
        & (df["Latitude"] <= lat_range[1])
        & (df["Longitude"] >= lon_range[0])
        & (df["Longitude"] <= lon_range[1])
    ]

    return filtered_df


def clean_up(datadir: str, date: datetime.date) -> None:
    today_paths = [str(Path(datadir, f"{date}-{hour:02}.dat")) for hour in range(6, 24)]
    tomorrow = date + datetime.timedelta(days=1)
    tomorrow_paths = [
        str(Path(datadir, f"{tomorrow}-{hour:02}.dat")) for hour in range(0, 12)
    ]
    for p in today_paths + tomorrow_paths:
        os.remove(p)


def find_nearby_monitors(
    aqs_readings: pd.DataFrame,
    coordinates: Tuple[float, float],
    max_distance: float,
    max_neighbors: int,
) -> pd.DataFrame:
    """
    Return the average hourly air quality at AQS monitors near coordinates.
    """
    neigh = NearestNeighbors(
        n_neighbors=max_neighbors, metric="haversine", algorithm="ball_tree"
    )
    aqs = aqs_readings.drop_duplicates(subset="AQSID", keep="first", inplace=False)
    monitor_lats = np.radians(aqs["Latitude"])
    monitor_lons = np.radians(aqs["Longitude"])
    monitor_latlons = np.stack([monitor_lats, monitor_lons], axis=-1)

    coords = np.radians(coordinates)[None, :]

    neigh_fit = neigh.fit(monitor_latlons)
    distances, indices = neigh_fit.kneighbors(coords)
    good_inds = np.where(distances * 6371 <= max_distance)
    good_inds = indices[good_inds]
    nearby_aqs = aqs_readings.loc[aqs_readings["AQSID"].isin(aqs.AQSID.iloc[good_inds])]
    return nearby_aqs
