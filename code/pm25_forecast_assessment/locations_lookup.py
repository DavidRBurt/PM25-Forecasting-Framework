# Given a file containing a list of city names and the state abbreviation, write a function that returns the lat, lon pair using census Gazetteer files.
import requests
import zipfile
from pathlib import Path
import os
import io 
from typing import Callable
import numpy as np
import difflib
import pandas as pd

def download_gazeeter_files(dp: Callable):
    """
    Download the gazetteer files from the census website
    """
    # Check if the file is already downloaded
    if os.path.exists(dp("urban_centers/2023_Gaz_ua_national.txt")):
        return dp("urban_centers/2023_Gaz_ua_national.txt")
    # If not, download the file -- note we use urban area lookup table currently
    target_url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_ua_national.zip"
    r = requests.get(target_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dp("urban_centers"))

    return dp("urban_centers/2023_Gaz_ua_national.txt")

def load_city_names(filepath: str):
    """
    Load the city names from the file
    """
    # load the file
    with open(filepath, "r") as f:
        city_names = f.readlines()
    # Split along space to get city name and state abbreviation
    return city_names

def get_lat_lon(datadir: str, city_name: str):
    """
    Given a city name and state abbreviation, return the lat, lon pair
    """
    dp = lambda x: Path(datadir, x)
    # load the gazetteer file with pandas
    datapath = download_gazeeter_files(dp)
    # Load with pandas, name columns name, lat, lon
    ub = pd.read_csv(datapath, delimiter="\t", skiprows=1, usecols=[1, 7, 8], names=["name", "lat", "lon"])
    # find the city name and state abbreviation
    # Strip the newline character
    city_name = city_name.strip()
    lat_lon = ub[ub["name"] == city_name]
    if not lat_lon.empty:
        return list(lat_lon[["lat", "lon"]].values[0])
    else:
        print(f"{city_name} not found")
        # find closest string to city name in the gazetteer file
        # and ask if this was the intended name            
        close = difflib.get_close_matches(city_name, ub["name"].to_list(), n=1)
        print(f"Did you mean: {close}?")
        return None