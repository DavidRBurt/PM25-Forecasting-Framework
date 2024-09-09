import datetime
import tempfile
from typing import Tuple

import cfgrib
import numpy as np
import pandas as pd
import requests
import xarray as xr

def create_naqfc_url_and_get_xr(day: datetime.date, cycle: int):
    
    # Constants for creating the full URL
    blob_container = "https://noaa-nws-naqfc-pds.s3.amazonaws.com/AQMv6"
    product = "ave_1hr_pm25_bc.227"  # 1-hour average PM2.5 concentration

    # Put it all together
    file_name = f"aqm.t{cycle:02}z.{product}.grib2"
    url = f"{blob_container}/cs.{day:%Y%m%d}/{file_name}"

    print(url)
    # Download the file
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded {file_name}")

    # Open the GRIB2 file
    ds = xr.open_dataset(file_name, engine='cfgrib')

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