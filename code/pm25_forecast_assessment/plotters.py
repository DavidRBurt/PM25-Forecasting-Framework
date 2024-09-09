from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from .experiment import Experiment

def plot_time_series(
    experiments: List[Experiment],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    figure_name: str = "tmp.pdf",
) -> None:
    
    # Enable LaTeX formatting, and set font size to 12
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=28)

    num_experiments = len(experiments)
    fig, axes = plt.subplots(1, num_experiments, figsize=((num_experiments +1)* 5, 5))
    
    # If there's only one experiment, wrap axes in a list for consistent handling
    if num_experiments == 1:
        axes = [axes]
    
    # For storing labels to create a unique legend
    handles = []
    labels = []

    # Define color-blind friendly colors
    color_blind_friendly_colors = ['blue', '#56B4E9', '#009E73', 'fuchsia', '#0072B2']
    
    # Define linestyles for the four additional forecasts
    linestyles = ['-.', ':', (0, (3, 5, 1, 5, 1, 5)), (0,(3,1,1,1))]

    for idx, experiment in enumerate(experiments):
        ax = axes[idx]
        
        # Reset start and end dates if not provided (to plot multiple experiments)
        start_date = experiment.start_date
        end_date = experiment.end_date

        start_time = 13
        end_time = start_time + 24
        persistence_time = start_time - 2

        days_to_plot = list()
        # Get days in the date range to plot
        for day in experiment.daily_data:
            if start_date is not None and day.date.date() < start_date.date():
                continue
            if end_date is not None and day.date.date() > end_date.date():
                continue
            days_to_plot.append(day)

        # Loop through all forecasts, with airnow first
        forecasts = ["airnow", "naqfc", "hrrr", "geoscf", "cams"]
        for i, forecast in enumerate(forecasts):
            all_predictions = list()
            for day in days_to_plot:
                predictions = day.forecasts[forecast].location_data
                day_predicted = [
                    np.mean(predictions.loc[predictions.ValidTime == h].PM25)
                    for h in range(start_time, end_time)
                ]
                all_predictions.append(day_predicted)
            # Flatten the list of lists into a single list
            all_predictions_flat = [hour for d in all_predictions for hour in d]
            
            # Determine linestyle and color
            if forecast == "airnow":
                linestyle = '-'
                color = 'orange'  # Airnow will have a solid line in blue color (for visibility)
                linewidth = 3
            else:
                linestyle = linestyles[(i-1) % len(linestyles)]
                color = color_blind_friendly_colors[(i-1) % len(color_blind_friendly_colors)]
                linewidth = 2
            
            line, = ax.plot(
                all_predictions_flat,
                label=forecast,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )
            
            # Store handle and label for the legend
            if idx == 0:  # Only store once for the legend
                handles.append(line)
                labels.append(forecast)
        
        # Add in persistence baseline (dashed line, separate from forecasts)
        all_predictions = list()
        for day in days_to_plot:
            predictions = day.forecasts["airnow"].location_data
            day_persistence = [
                np.mean(predictions.loc[predictions.ValidTime == persistence_time].PM25)
                for h in range(start_time, end_time)
            ]
            all_predictions.append(day_persistence)
        all_predictions_flat = [hour for d in all_predictions for hour in d]
        # plot persistence as step functions
        for day in range(len(days_to_plot)):
            line, = ax.plot(
                range(day * 24, (day + 1) * 24 + 1),
                [all_predictions_flat[day * 24]] * 25,
                color="black",
                linestyle="--",
                linewidth=2,  # Dashed line for persistence
                label="Persistence" if day == 0 else None,
            )
            if idx == 0 and day == 0:  # Only add once for the legend
                handles.append(line)
                labels.append("Persistence")

        # Set xlim to the number of hours in the plot
        ax.set_xlim(0, len(all_predictions_flat))

        ax.set_xticks(
            range(0, len(all_predictions_flat), 24),
            labels=[str(day.date)[5:10] for day in days_to_plot],
            rotation=45,
        )

        # Only show y-axis on the first subplot
        if idx == 0:
            ax.set_ylabel(r"PM2.5 ($\mu g/m^3$)")

        loc = experiment.location[:-11]
        ax.set_title(f"{loc}")

    # If only one location, create a single legend on the far right, with each label in a row and larger font size
    if len(experiments) == 1:
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2))
        plt.tight_layout()
    else:
        # Create a single legend below the plots, on one line, with larger font size
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.1))

    plt.savefig(figure_name, bbox_inches="tight")
    
def confusion_matrix(results: List[Dict]) -> None:
    """
    Plot a confusion matrix for the results.
    """
    true_positives = dict()
    false_positives = dict()
    false_negatives = dict()
    true_negatives = dict() 
    for day, day_results in results.items():
        r = day_results["smokeday"]
        for forecast, result in r.items():
            if forecast not in true_positives:
                true_positives[forecast] = 0
                false_positives[forecast] = 0
                false_negatives[forecast] = 0
                true_negatives[forecast] = 0
            if result:
                if r["observed"]:
                    true_positives[forecast] += 1
                else:
                    false_positives[forecast] += 1
            else:
                if not r["observed"]:
                    true_negatives[forecast] += 1
                else:
                    false_negatives[forecast] += 1
    return true_positives, false_positives, false_negatives, true_negatives
    # mport numpy as np



    # for day in days_to_plot:
    #     monitor_readings = day.airnow_location_data
    #     hours = [f"{h:02}:00" for h in range(13, 37)]
    #     day_observed = [
    #         np.mean(monitor_readings.loc[monitor_readings.ValidTime == h].PM25)
    #         for h in hours
    #     ]
    #     observed.append(day_observed)
    #     hrrr_predictions = day.hrrr_location_data
    #     hrrr_day_predicted = [
    #         np.mean(hrrr_predictions.loc[hrrr_predictions.ValidTime == h].PM25)
    #         for h in range(13, 37)
    #     ]
    #     hrrr_predicted.append(hrrr_day_predicted)
    #     cams_predictions = day.cams_location_data
    #     cams_day_predicted = [
    #         np.mean(cams_predictions.loc[cams_predictions.ValidTime == h].PM25)
    #         for h in range(13, 37)
    #     ]

    #     # cams_predicted.append(cams_day_predicted)
    #     # cams_day_predicted = [
    #     #     np.mean(cams_predictions.loc[cams_predictions.ValidTime == h].PM25)
    #     #     for h in range(0, 13)
    #     # ]
    #     cams_predicted.append(cams_day_predicted)

    #     geoscf_predictions = day.geoscf_location_data
    #     geoscf_day_predicted = [
    #         np.mean(geoscf_predictions.loc[geoscf_predictions.ValidTime == h].PM25)
    #         for h in range(13, 37)
    #     ]
    #     geoscf_predicted.append(geoscf_day_predicted)

    #     day_persistence = [
    #         np.mean(monitor_readings.loc[monitor_readings.ValidTime == "11:00"].PM25)
    #         for h in hours
    #     ]
    #     persistence.append(day_persistence)

    # observed = [hour for d in observed for hour in d]
    # hrrr_predicted = [hour for d in hrrr_predicted for hour in d]
    # cams_predicted = [hour for d in cams_predicted for hour in d]
    # geoscf_predicted = [hour for d in geoscf_predicted for hour in d]
    # persistence = [hour for d in persistence for hour in d]

    # import matplotlib.pyplot as plt

    # plt.plot(observed, label="Observed")
    # plt.plot(hrrr_predicted, label="HRRR")
    # plt.plot(cams_predicted, label="CAMS")
    # plt.plot(geoscf_predicted, label="GEOS-CF")
    # plt.plot(persistence, label="Persistence")
    # plt.legend()

    # plt.title(
    #     f"{self.location[:-2]} {self.location[-2:]} from {str(start_date)[:10]} to {str(end_date)[:10]}"
    # )



    # # I would like to remove the white space before and after the end of the plot
    # plt.xlim(0, len(observed))

    # plt.ylabel("PM2.5 ($\mu g/m^3$)")
    # # plt.legend(["Observed", "HRRR Predictions", "CAMS Predictions", "Persistence"], loc="upper right", fontsize=8)
    # os.makedirs(self.figures_directory, exist_ok=True)
    # plt.savefig(
    #     f"{self.figures_directory}/{self.location}-{'_'.join([str(m) for m in self.months])}-first-day-{start_date.day}-last-day-{end_date.day}.pdf",
    #     bbox_inches="tight",
    # )
