from dataclasses import dataclass
from abc import abstractmethod
from typing import Dict, Iterable, Union
import numpy as np
from .daydataclass import DailyData


@dataclass
class Metric:
    name: str

    @abstractmethod
    def __call__(self, day_data: DailyData) -> Dict[str, Union[float, bool]]:
        """Evaluate metric for given day"""


@dataclass
class RMSE(Metric):
    name: str = "rmse"
    hours_to_include: Iterable[int] = range(13, 36)
    persistence_hour: int = 11

    def __call__(self, day_data: DailyData) -> Dict[str, float]:
        airnow = day_data.forecasts.pop("airnow")
        monitor_readings = airnow.location_data
        observed = [
            np.mean(monitor_readings.loc[monitor_readings.ValidTime == h].PM25)
            for h in self.hours_to_include
        ]
        predictions = {
            forecast_name: [np.mean(forecast.location_data.loc[forecast.location_data.ValidTime == h].PM25) for h in self.hours_to_include]
            for forecast_name, forecast in day_data.forecasts.items()
        }
        rmses = {
            forecast_name: np.sqrt(np.mean([(o - p) ** 2 for o, p in zip(observed, predicted)]))
            for forecast_name, predicted in predictions.items()
        }
        # Add persistence RMSE
        persistence = np.mean(
            monitor_readings.loc[
                monitor_readings.ValidTime == self.persistence_hour
            ].PM25
        )
        rmses["persistence"] = np.sqrt(np.mean([(o - persistence) ** 2 for o in observed]))
        return rmses


@dataclass
class MeanExcessExposure(Metric):
    name: str = "mee"
    hours_to_include: Iterable[int] = range(13, 36)

    def __call__(self, day_data: DailyData) -> Dict[str, float]:
        airnow = day_data.forecasts.pop("airnow")
        monitor_readings = airnow.location_data
        observed = [
            np.mean(monitor_readings.loc[monitor_readings.ValidTime == h].PM25)
            for h in self.hours_to_include
        ]
        predictions = {
            forecast_name: [np.mean(forecast.location_data.loc[forecast.location_data.ValidTime == h].PM25) for h in self.hours_to_include]
            for forecast_name, forecast in day_data.forecasts.items()
        }
        # Compute best hour to go out according to each forecast
        best_predicted_hours = {
            forecast_name: np.argmin(predicted)
            for forecast_name, predicted in predictions.items()
        }
        # Excess exposure is difference between hour you go out and best hour
        excess_exposures = {
            forecast_name: observed[best_predicted_hour] - np.min(observed)
            for forecast_name, best_predicted_hour in best_predicted_hours.items()
        }
        # Add persistence excess exposure, which is difference between mean and min
        excess_exposures["persistence"] = np.mean(observed) - np.min(observed)
        return excess_exposures

@dataclass
class IsSmokeDay(Metric):
    name: str = "smokeday"
    # Using WHO 24 hour exposure guidance currently (25)
    # But we compare max exposure instead of mean.
    # Maybe this is not good. Also EPA is 35.
    threshold: float = 35.0
    hours_to_include: Iterable[int] = range(13, 36)
    persistence_hour: int = 11

    def __call__(self, day_data: DailyData) -> Dict[str, bool]:
        airnow = day_data.forecasts.pop("airnow")
        monitor_readings = airnow.location_data
        observed = [
            np.mean(monitor_readings.loc[monitor_readings.ValidTime == h].PM25)
            for h in self.hours_to_include
        ]
        predictions = {
            forecast_name: [np.mean(forecast.location_data.loc[forecast.location_data.ValidTime == h].PM25) for h in self.hours_to_include]
            for forecast_name, forecast in day_data.forecasts.items()
        }
        # Add persistence estimate exposure
        persistence = np.mean(
            monitor_readings.loc[
                monitor_readings.ValidTime == self.persistence_hour
            ].PM25
        )
        # Take max value for all monitors
        max_observed = np.max(observed)
        max_predictions = {
            forecast_name: np.max(predicted)
            for forecast_name, predicted in predictions.items()
        }
        max_persistence = np.max(persistence)
        results = {
            forecast_name: max_value > self.threshold
            for forecast_name, max_value in max_predictions.items()
        }
        results["persistence"] = max_persistence > self.threshold
        results["observed"] = max_observed > self.threshold
        return results