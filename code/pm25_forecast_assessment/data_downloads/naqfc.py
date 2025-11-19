import datetime
import tempfile
from typing import Tuple

import cfgrib
import numpy as np
import pandas as pd
import requests
import xarray as xr


def get_AQM_version(day: datetime.date) -> str:
    if day.year < 2020:
        raise ValueError(f"naqfc doesn't support years < 2020. Year given: {day.year}")
    elif day < datetime.datetime(2021, 7, 20):
        return "AQMv5"
    elif day < datetime.datetime(2024, 5, 14):
        return "AQMv6"
    else:
        return "AQMv7"

def create_naqfc_url_and_get_xr(day: datetime.date, cycle: int) -> xr.Dataset:
    """
    Data source: https://registry.opendata.aws/noaa-nws-naqfc-pds/?utm_source=chatgpt.com
    Properties of data:
        Data source is split by year:
            beginning with AQMv5 in 2020, 
            transitioning to AQMv6 on 20 July 2021, 
            and to AQMv7 on 14 May 2024. 
            The length of each forecast was 48 hours prior to the implementation of AQMv6
        Only cycles 06 and 12 exist
    
    Args:
        day (datetime.date):
            The date (UTC) of the forecast run.
        cycle (int):
            The initialization hour of the forecast [6, 12 UTC].
            e.g., `cycle=12` corresponds to the 12Z model run.
    Returns:
        xr.Dataset:
            An xarray Dataset containing NAQFC forecast for pmtf.
    """
    if cycle != 6 and cycle != 12:
        raise ValueError(f"create_naqfc_url_and_get_xr: cycle must be either 6 or 12, but is {cycle}")
    
    # 1. Constants for creating the full URL
    blob_container = "https://noaa-nws-naqfc-pds.s3.amazonaws.com"
    AQM_version = get_AQM_version(day)
    product = "ave_1hr_pm25_bc"

    file_name =  f"aqm.t{cycle:02}z.{product}.{day:%Y%m%d}.227.grib2"
    url = f"{blob_container}/{AQM_version}/CS/{day:%Y%m%d}/{cycle:02}/{file_name}"

    # 2. Fetch grib data
    file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)
    resp = requests.get(url)

    with file as f:
        f.write(resp.content)

    # Open the GRIB2 file
    ds = xr.open_dataset(file.name, engine='cfgrib')

    return ds

def naqfc_data_download(date: datetime.date, cycle: int = 6) -> pd.DataFrame:
    ds = create_naqfc_url_and_get_xr(date, cycle)

    data = []

    # Iterate over the selected forecast steps and extract the relevant data
    for i in range(1, 25):  # Note: range starts at 1 and goes to 25
        pm25_values = ds['pmtf'].isel(step=i).values.flatten()
        latitudes = ds['latitude'].values.flatten()
        longitudes = ds['longitude'].values.flatten()

        # Create a DataFrame for the current time step
        df_temp = pd.DataFrame({
            'Latitude': latitudes,
            'Longitude': longitudes,
            'ValidTime': np.repeat(12+i, len(latitudes)),
            'PM25': np.round(pm25_values, 2)
        })

        data.append(df_temp)

    # Concatenate all the data into a single DataFrame
    df = pd.concat(data, ignore_index=True)

    # Drop rows with NaN values in PM25
    df = df.dropna(subset=['PM25'], ignore_index=True)

    return df