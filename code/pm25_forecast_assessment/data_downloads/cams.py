import cdsapi
import pygrib
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import datetime
from pathlib import Path
from calendar import monthrange
from pathlib import Path

# to be able to run the commands to get api data, see the following:
# - https://ads.atmosphere.copernicus.eu/api-how-to
# - https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS
# and this is a helper interface to get api calls:
# - https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=form

def cams_data_download(date: datetime.date, cams_path: Path, cycle: int = 0) -> None:
    """
    Download CAMS data for a specific date, actually downloads full month
    """

    month = date.month
    year = date.year

    _, month_end = monthrange(year, month)

    c = cdsapi.Client()
    c.retrieve(
        'cams-global-atmospheric-composition-forecasts',
        {   
            'variable': 'particulate_matter_2.5um',
            'date': f'{year:04}-{month:02}-01/{year:04}-{month:02}-{month_end:02}',
            'time': [
                f'{cycle:02}:00', 
            ],
            'leadtime_hour': [
                '1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30',
            ],
            'type': 'forecast',
            'format': 'grib',
        },
        f'download-{date}.grib'
        )
    
    # open the file as an xarray to filter out data
    ds = xr.open_dataset(f"download-{date}.grib", engine="cfgrib")
    df = ds.to_dataframe()
    df = df.reset_index()
    # delete the downloaded file
    os.remove(f"download-{date}.grib")

    # if longitude is more than 180, convert it to negative
    df.loc[df["longitude"] > 180, "longitude"] = df.loc[df["longitude"] > 180,"longitude"] - 360

    # filter out non CONUS data
    df_conus = df[(df["latitude"] > 20) & (df["latitude"] < 50) & (df["longitude"] > -130) & (df["longitude"] < -60)]

    # convert pm2pm5 to ug/m3
    df_conus["pm2p5"] = df_conus["pm2p5"] * (10**9)

    # round the values
    df_conus["latitude"] = df_conus["latitude"].round(4)
    df_conus["longitude"] = df_conus["longitude"].round(4)
    df_conus["pm2p5"] = df_conus["pm2p5"].round(4)
    # valid time, add a day if it is for the next day, this is broken right now because date.day doesn't correspond to time
    time_diff = df_conus["valid_time"] - df_conus["time"]
    df_conus["valid_time"] = 24 * time_diff.dt.days + time_diff.dt.seconds / 3600 + cycle
    #+ cycle + 24 * (df_conus["valid_time"].dt.day - date.day)
    # get the unique values of the time
    df_conus["time"].unique()


    # for each day in time, save a different csv file containing only valid time, lat, long, and pm2.5
    for t in df_conus["time"].unique():
        df_t = df_conus[df_conus["time"] == t]
        df_t = df_t[["valid_time", "latitude", "longitude", "pm2p5"]]
        # rename df_t columns to match other datasets
        df_t = df_t.rename(columns={"valid_time": "ValidTime", "latitude": "Latitude", "longitude": "Longitude", "pm2p5": "PM25"})
        # check if cams folder exists, otherwise create it

        if not os.path.exists(str(Path(cams_path).parent)):
            os.makedirs(str(Path(cams_path).parent))
        # save data 
        data_date = str(t).split(" ")[0]

        df_t.to_csv(f"{str(Path(cams_path).parent)}/{data_date}.csv", index=False)

if __name__ == "__main__":
    # run a quick test
    cams_data_download(datetime.date(2021, 1, 1), Path("data/cams/2021/01/01.csv"))