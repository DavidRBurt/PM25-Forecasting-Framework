import argparse

from datetime import datetime
from pathlib import Path

from pm25_forecast_assessment.experiment import Experiment
from pm25_forecast_assessment.metrics import (
    IsSmokeDay,
    MeanExcessExposure,
    RMSE,
)
from pm25_forecast_assessment.plotters import plot_time_series


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for prediction analysis.
    
    Currently accepts location file path and optional figure output name.
    The location file should contain semicolon-separated values with location names,
    start dates, and end dates for analysis.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing location_file
            and figure_name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location_file",
        type=str,
        help=(
            "Name of file. File should contain semicolon-separated values with columns: "
            "location, start date, end date. "
            "Location: Must match a location in the UA Census Gazetteer. "
            "Start date for analysis in format YYYY-MM-DD. "
            "End date for analysis in format YYYY-MM-DD."
        ),
    )
    # add an argument to name the figure output
    parser.add_argument(
        "--figure_name",
        type=str,
        help="Name of figure output. Must be a string.",
        default="tmp.pdf",
    )
    
    # forecast types
    parser.add_argument(
        "--forecasts",
        type=str,
        nargs='+',
        help="Names of forecasts. Must be strings.",
        default="airnow",
    )
    return parser.parse_args()


def load_file(file_name: str) -> tuple[list[str], list[datetime], list[datetime]]:
    """Load location and date information from a semicolon-separated file.
    
    Args:
        file_name: Path to the file containing location and date information.
            Each line should be formatted as: location;start_date;end_date
            where dates are in YYYY-MM-DD format.
    
    Returns:
        tuple: A tuple containing three lists:
            - locations (list[str]): Location names matching UA Census Gazetteer.
            - start_dates (list[datetime]): Start dates for each location's analysis.
            - end_dates (list[datetime]): End dates for each location's analysis.
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
    forecasts = list(set(args.forecasts) | {'airnow'}) # automatically include the airnow baseline
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
        results = experiment.run()
        plot_time_series([experiment], figure_name=args.figure_name)
