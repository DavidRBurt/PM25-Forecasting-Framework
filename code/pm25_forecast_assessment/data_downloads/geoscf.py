import os
import urllib
from datetime import datetime, timedelta
import xarray as xr
from pydap.client import open_dods_url
import requests
import pandas as pd

def conus(ds: xr.Dataset) -> xr.Dataset:
    """
    Return the dataset with a bounding box around the contiguous US
    """
    return ds.sel(lat=slice(25, 50), lon=slice(-125, -65))

def download_geoscf_data(date: datetime.date) -> xr.Dataset:
    """
    Download Geos-CF data from OpenDAP server
    """
    tomorrow = date + timedelta(days=1)
    baseurl = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v1/forecast/"
    yearurl = f"{baseurl}/Y{date.year:04}"
    monthurl = f"{yearurl}/M{date.month:02}"
    dayurl = f"{monthurl}/D{date.day:02}/H12"
    rootfilename = "GEOS-CF.v01.fcst.aqc_tavg_1hr_g1440x721_v1."
    starttime = f"{date.year:04}{date.month:02}{date.day:02}_12z"
    # which hours do we want? How to deal with 12 hour offset and forecasting at half hour?
    endtimes = [f"{date.year:04}{date.month:02}{date.day:02}_{h:02}30z" for h in range(12, 24)] \
        + [f"{tomorrow.year:04}{tomorrow.month:02}{tomorrow.day:02}_{h:02}30z" for h in range(0, 24)]
    filenames = [f"{rootfilename}{starttime}+{endtime}" for endtime in endtimes]
    def download_file(filename: str):
        url = f"{dayurl}/{filename}.nc4"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, f"{filename}.nc4")
        return xr.open_dataset(f"{filename}.nc4")
    
    datasets = [download_file(filename) for filename in filenames]
    # remove the files
    for filename in filenames:
        os.remove(f"{filename}.nc4")
    ds = xr.concat(datasets, dim="time") # Concatenate the datasets along the time dimension
    # Remove unused lev dimension 
    ds = ds.squeeze("lev")

    conus_ds = conus(ds["PM25_RH35_GCC"])
    # Rename the columns to match the column names in the other datasets
    stacked_ds = conus_ds.stack(stacked_dim={"time": "ValidTime", "lat": "Latitude", "lon": "Longitude"})
    df = stacked_ds.reset_index("stacked_dim").to_dataframe(name='PM25')
    df = df.reset_index()
    # select only the columns we want
    df = df[["time", "lat", "lon", "PM25"]]
    # convert the time to an integer, add 24 if next day,
    # Get time delta in hours between the time column and date, be careful at end of months
    df["time"] = pd.to_datetime(df["time"])
    time = (df["time"] - date).dt.total_seconds() / 3600 - 0.5 # subtract 0.5 to get the time at the start of the hour
    df["time"] = time + 1
    # rename the columns
    df = df.rename(columns={"time": "ValidTime", "lat": "Latitude", "lon": "Longitude"})
    # Truncate PM2.5 to 1 decimal place
    df["PM25"] = df["PM25"].round(1)
    return df


if __name__ == "__main__":
    fp = "/Users/davidrburt/Downloads/GEOS-CF.v01.fcst.aqc_tavg_1hr_g1440x721_v1.20220101_12z+20220106_1130z.nc4"
    date = datetime(2023, 5, 31)
    ds = download_geoscf_data(date)
    print(ds)
    # CHeck if ds contains nan
    print(ds.isnull().sum())
