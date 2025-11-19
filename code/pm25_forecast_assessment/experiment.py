import os
from datetime import datetime, timedelta
from pathlib import Path

import json_tricks
import numpy as np

from pm25_forecast_assessment.daydataclass import DailyData
from pm25_forecast_assessment.metrics import Metric


class Experiment:

    def __init__(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        metrics: list[Metric],
        results_directory: str,
        figures_directory: str,
        data_directory: str,
        forecasts: list[str] | None = None,
    ) -> None:
        self.location = location
        self.start_date = start_date
        self.end_date = end_date
        self.metrics = metrics
        self.results_directory = results_directory
        self.figures_directory = figures_directory
        self.data_directory = data_directory
        self.forecasts = forecasts
        self.daily_data = self.load_data()

    def load_data(self) -> list[DailyData]:
        """
        Return a list of DailyData objects for the location and date range.
        Note this can be slow since data is (down)loaded when the DailyData object is created.
        """
        delta = self.end_date - self.start_date
        return [
            DailyData(
                self.start_date + timedelta(days=i),
                self.location,
                self.data_directory,
                _forecasts=self.forecasts,
            )
            for i in range(delta.days + 1)
        ]

    def evaluate_metrics(self) -> dict[str, dict[str, dict]]:
        return {
            day.date: {metric.name: metric(day) for metric in self.metrics}
            for day in self.daily_data
        }

    def save_results(self, results: dict[str, dict[str, dict]]) -> None:
        for date, day_results in results.items():
            directory = Path(self.results_directory, self.location)
            os.makedirs(directory, exist_ok=True)
            
            filepath = str(Path(directory, f"{date}.json"))
            try:
                json_tricks.dump(day_results, filepath)
            except ValueError:
                print(f"Missing values for {date}. Need to decide how to handle this.")

    def run(self) -> dict[str, dict[str, dict]]:
        results = self.evaluate_metrics()
        self.save_results(results)
        return results
