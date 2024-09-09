import argparse

import numpy as np
from datetime import datetime
from typing import Tuple, Dict
from pathlib import Path

from pm25_forecast_assessment.experiment import Experiment
from pm25_forecast_assessment.metrics import (
    Metric,
    RMSE,
    MeanExcessExposure,
    IsSmokeDay,
)
from pm25_forecast_assessment.plotters import plot_time_series, confusion_matrix


def parse_arguments() -> argparse.Namespace:
    """
    Add command line arguments. Currently, this is a location, year and list of months
    over which to do the analysis.
    Returns the namespace argument for use in other functions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location_file",
        type=str,
        help="Name of file. File should be a csv with columns location, start date, end data.\
        Location: Must match a location in the UA Census Gazetteer. \
        Start date for analysis in format YYYY-MM-DD \
        End date for analysis in format YYYY-MM-DD.",
    )
    # add an argument to name the figure output
    parser.add_argument(
        "--figure_name",
        type=str,
        help="Name of figure output. Must be a string.",
        default="tmp.pdf",
    )
    return parser.parse_args()


def load_file(file_name: str) -> Tuple[str, str, str]:
    """
    Load the file with the location, start date and end date.
    """
    locations = []
    start_dates = []
    end_dates = []
    with open(file_name, "r") as f:
        for line in f:
            print(line.split("\t"))
            location, start_date, end_date = line.split(";")
            # If end date has a new line character, remove it
            if end_date[-1] == "\n":
                end_date = end_date[:-1]
            locations.append(location)
            start_date = datetime(*[int(a) for a in start_date.strip().split("-")])
            end_date = datetime(*[int(a) for a in end_date.strip().split("-")])
            start_dates.append(start_date)
            end_dates.append(end_date)
    return locations, start_dates, end_dates


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    metrics = [RMSE(), MeanExcessExposure(), IsSmokeDay()]

    locations, start_dates, end_dates = load_file(args.location_file)
    figures_directory = Path(Path(__file__).parents[1], "figures")
    results_directory = Path(Path(__file__).parents[1], "results")
    data_directory = Path(Path(__file__).parents[1], "data")
    forecasts = ["hrrr", "airnow", "geoscf", "cams", "naqfc"]
    experiments = []
    for location, start_date, end_date in zip(locations, start_dates, end_dates):
        experiment = Experiment(
            location=location,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            results_directory=results_directory,
            figures_directory=figures_directory,
            data_directory=data_directory,
            forecasts=forecasts,
        )
        experiments.append(experiment)

    plot_time_series(experiments, figure_name=args.figure_name)
