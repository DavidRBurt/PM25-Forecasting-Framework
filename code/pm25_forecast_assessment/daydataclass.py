import datetime
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from abc import abstractmethod

import numpy as np
import pandas as pd

from .data_downloads.airnow import airnow_data_download, find_nearby_monitors
from .data_downloads.hrrr import hrrr_data_download, find_nearby_predictions
from .data_downloads.cams import cams_data_download
from .data_downloads.geoscf import download_geoscf_data
from .data_downloads.naqfc import naqfc_data_download
from .locations_lookup import get_lat_lon


@dataclass
class Forecast:
    _location: str
    date: datetime.date
    data_directory: Path
    _name: Optional[str] = None
    _data: Optional[pd.DataFrame] = None
    _location_data: Optional[pd.DataFrame] = None

    @property
    def location(self) -> str:
        return self._location

    @property
    def name(self) -> str:
        return self._name

    @property
    def year(self) -> int:
        return self.date.year

    @property
    def month(self) -> int:
        return self.date.month

    @property
    def day(self) -> int:
        return self.date.day

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.download()
            self._data = pd.read_csv(self.datapath)
        return self._data

    @property
    def location_data(self) -> pd.DataFrame:
        if self._location_data is None:
            self.build_location_data()
            self._location_data = pd.read_csv(self.location_datapath)
        return self._location_data

    @property
    def datapath(self) -> str:
        """
        Build the path to the data file for the given data type
        """
        return str(
            Path(
                self.data_directory,
                self.name,
                f"{self.year:04}",
                f"{self.month:02}",
                f"{self.date.strftime("%Y-%m-%d")}.csv",
            )
        )

    @property
    def location_datapath(self) -> str:
        """
        Build the path to the location data file for the given data type
        """
        return str(
            Path(
                self.data_directory,
                "location-data",
                self.location,
                self.name,
                f"{self.year:04}",
                f"{self.month:02}",
                f"{self.date.strftime("%Y-%m-%d")}.csv",
            )
        )

    def is_downloaded(self) -> bool:
        return os.path.exists(self.datapath)

    def location_built(self) -> bool:
        return os.path.exists(self.location_datapath)

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def build_location_data(self) -> None:
        pass


@dataclass
class GenericForecast(Forecast):
    _name: str

    def download_fn(self):
        raise NotImplementedError("Must implement download_fn")

    def find_neighbor_fn(self):
        raise NotImplementedError("Must implement find_neighbor_fn")

    def download(self) -> None:
        if not self.is_downloaded():
            data = self.download_fn()
            os.makedirs(Path(self.datapath).parent, exist_ok=True)
            data.to_csv(self.datapath, sep=",", index=False)

    def build_location_data(self) -> None:
        if not self.location_built():
            neighbors = self.find_neighbor_fn()
            os.makedirs(Path(self.location_datapath).parent, exist_ok=True)
            neighbors.to_csv(self.location_datapath, sep=",", index=False)


@dataclass
class AirNowForecast(GenericForecast):
    _name: str = "airnow"
    _max_distance: Optional[float] = 50.0
    _max_neighbors: Optional[int] = 10

    def download_fn(self):
        return airnow_data_download(self.date, self.data_directory)

    def find_neighbor_fn(self):
        return find_nearby_monitors(
            self.data,
            get_lat_lon(self.data_directory, self.location),
            self._max_distance,
            self._max_neighbors,
        )


@dataclass
class CAMSForecast(GenericForecast):
    _name: str = "cams"
    _max_distance: Optional[float] = 60.0

    def download_fn(self) -> None:
        return cams_data_download(self.date, self.datapath, cycle=12)

    def find_neighbor_fn(self) -> None:
        return find_nearby_predictions(
            self.data,
            get_lat_lon(self.data_directory, self.location),
            self._max_distance,
        )


@dataclass
class GEOSCFForecast(GenericForecast):
    _name: str = "geoscf"
    _max_distance: Optional[float] = 50.0

    def download_fn(self) -> None:
        return download_geoscf_data(self.date)

    def find_neighbor_fn(self) -> None:
        return find_nearby_predictions(
            self.data,
            get_lat_lon(self.data_directory, self.location),
            self._max_distance,
        )

@dataclass
class NAQFCForecast(GenericForecast):
    _name: str = "naqfc"
    _max_distance: Optional[float] = 50.0

    def download_fn(self) -> None:
        return naqfc_data_download(self.date, cycle=12)

    def find_neighbor_fn(self) -> None:
        return find_nearby_predictions(
            self.data,
            get_lat_lon(self.data_directory, self.location),
            self._max_distance,
        )

@dataclass
class HRRRForecast(Forecast):
    _name: str = "hrrr"

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.download()
            _data = pd.read_csv(self.datapath)
            self.idx_to_latlon_convert(_data)
            self._data = _data
        return self._data

    @property
    def latlon_idx_path(self) -> str:
        return str(
            Path(
                self.data_directory,
                "hrrr-latlon-idx.csv",
            )
        )

    @property
    def location_data(self) -> pd.DataFrame:
        if self._location_data is None:
            self.build_location_data()
            _location_data = pd.read_csv(self.location_datapath)
            self.idx_to_latlon_convert(_location_data)
            self._location_data = _location_data
        return self._location_data

    def download(self) -> None:
        if not self.is_downloaded():
            data = hrrr_data_download(self.date, cycle=12)
            self.latlon_to_idx_convert(data)
            os.makedirs(Path(self.datapath).parent, exist_ok=True)
            data.to_csv(self.datapath, sep=",", index=False)

    def build_location_data(self, max_distance: float = 50.0) -> None:
        if not self.location_built():
            nearby_predictions = find_nearby_predictions(
                self.data,
                get_lat_lon(self.data_directory, self.location),
                max_distance,
            )
            os.makedirs(Path(self.location_datapath).parent, exist_ok=True)
            self.latlon_to_idx_convert(nearby_predictions)
            nearby_predictions.to_csv(self.location_datapath, sep=",", index=False)

    def latlon_to_idx_convert(self, prediction: pd.DataFrame) -> None:
        if not os.path.exists(self.latlon_idx_path):
            self.build_latlon_idx(prediction)
        latlon_idx = np.loadtxt(self.latlon_idx_path, delimiter=",")
        idx_lookup = {(latlon[0], latlon[1]): i for i, latlon in enumerate(latlon_idx)}
        lats, lons = (
            prediction.Latitude.to_list(),
            prediction.Longitude.to_list(),
        )
        latlonidx = [
            idx_lookup[(float(f"{lat:.6f}"), float(f"{lon:.6f}"))]
            for lat, lon in zip(lats, lons)
        ]
        prediction["LatLonIdx"] = np.array(latlonidx)
        prediction.drop(columns=["Latitude", "Longitude"], inplace=True)

    def idx_to_latlon_convert(self, prediction: pd.DataFrame) -> None:
        if not os.path.exists(self.latlon_idx_path):
            raise FileNotFoundError(
                "Must build lookup table before indices can be mapped to (lat, lon) pairs!"
            )
        latlon_idx = np.loadtxt(self.latlon_idx_path, delimiter=",")
        latlons = latlon_idx[prediction.LatLonIdx.to_list()]
        prediction["Latitude"] = latlons[:, 0]
        prediction["Longitude"] = latlons[:, 1]
        prediction.drop(columns=["LatLonIdx"], inplace=True)

    def build_latlon_idx(self, prediction: pd.DataFrame) -> None:
        lats = prediction.Latitude.to_list()
        lons = prediction.Longitude.to_list()
        latlons = ((lat, lon) for lat, lon in zip(lats, lons))
        unique_latlons = set(latlons)
        np_unique_latlons = np.array(list(unique_latlons))
        np.savetxt(self.latlon_idx_path, np_unique_latlons, delimiter=",", fmt="%.6f")


@dataclass
class DailyData:
    date: datetime.date
    location_name: str
    data_directory: str
    _forecasts: Optional[List[str]] = field(
        default_factory=lambda: ["airnow", "geoscf", "hrrr", "cams"]
    )

    @property
    def forecasts(self) -> Dict:
        return {forecast: self.build_forecast(forecast) for forecast in self._forecasts}

    def build_forecast(self, forecast: str) -> Forecast:
        if forecast == "airnow":
            return AirNowForecast(self.location_name, self.date, self.data_directory)
        elif forecast == "hrrr":
            return HRRRForecast(self.location_name, self.date, self.data_directory)
        elif forecast == "cams":
            return CAMSForecast(self.location_name, self.date, self.data_directory)
        elif forecast == "geoscf":
            return GEOSCFForecast(self.location_name, self.date, self.data_directory)
        elif forecast == "naqfc":
            return NAQFCForecast(self.location_name, self.date, self.data_directory)
        else:
            raise ValueError(f"Unknown forecast type {forecast}")
