import os
import json
from collections import defaultdict
import numpy as np
from pathlib import Path

# Paths
results_dir = Path(__file__).resolve().parents[1] / "results"
tables_dir = Path(__file__).resolve().parents[1] / "latex_tables"
os.makedirs(tables_dir, exist_ok=True)

short_city_names = {    
    "Boston, MA--NH Urban Area": "boston",
    "Worcester, MA--CT Urban Area": "worcester",
    "New York--Jersey City--Newark, NY--NJ Urban Area": "newyork",
    "Philadelphia, PA--NJ--DE--MD Urban Area": "philadelphia",
    "Chicago, IL--IN Urban Area": "chicago",
    "Detroit, MI Urban Area": "detroit",
    "Minneapolis--St. Paul, MN Urban Area": "minneapolis",
    "St. Louis, MO--IL Urban Area": "stlouis",
    "Washington--Arlington, DC--VA--MD Urban Area": "washington",
    "Atlanta, GA Urban Area": "atlanta",
    "Miami--Fort Lauderdale, FL Urban Area": "miami",
    "Nashville-Davidson, TN Urban Area": "nashville",
    "Memphis, TN--MS--AR Urban Area": "memphis",
    "Louisville--Jefferson County, KY--IN Urban Area": "louisville",
    "Dallas--Fort Worth--Arlington, TX Urban Area": "dallas",
    "Houston, TX Urban Area": "houston",
    "Phoenix--Mesa--Scottsdale, AZ Urban Area": "phoenix",
    "Denver--Aurora, CO Urban Area": "denver",
    "Las Vegas--Henderson--Paradise, NV Urban Area": "lasvegas",
    "Los Angeles--Long Beach--Anaheim, CA Urban Area": "losangeles",
    "San Francisco--Oakland, CA Urban Area": "sanfrancisco",
    "Riverside--San Bernardino, CA Urban Area": "riverside",
    "Seattle--Tacoma, WA Urban Area": "seattle",
    "Portland, OR--WA Urban Area": "portland",
    "Tampa--St. Petersburg, FL Urban Area": "tampa",
    "San Diego, CA Urban Area": "sandiego",
    "Baltimore, MD Urban Area": "baltimore",
    "Orlando, FL Urban Area": "orlando",
    "Charlotte, NC--SC Urban Area": "charlotte",
    "San Antonio, TX Urban Area": "sanantonio"
}

# Function to update confusion matrix
def update_confusion_matrix(matrix, predicted, observed):
    if predicted and observed:
        matrix['TP'] += 1  # True Positive
    elif predicted and not observed:
        matrix['FP'] += 1  # False Positive
    elif not predicted and observed:
        matrix['FN'] += 1  # False Negative
    elif not predicted and not observed:
        matrix['TN'] += 1  # True Negative

# Function to calculate precision and recall
def calculate_precision_recall(matrix):
    try:
        precision = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    except ZeroDivisionError:
        precision = "--"
    
    try:
        recall = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    except ZeroDivisionError:
        recall = "--"

    return precision, recall

# Iterate over each city folder in the results directory
for city in os.listdir(results_dir):
    print(f"Processing {short_city_names[city]}")
    city_path = os.path.join(results_dir, city)

    # Check if the directory is non-empty
    if os.path.isdir(city_path) and os.listdir(city_path):
        
        # Initialize confusion matrices for each model
        confusion_matrices = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
        
        # Iterate over each JSON file in the city folder
        for json_file in os.listdir(city_path):
            json_path = os.path.join(city_path, json_file)
            
            # Load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract observed and predicted values
            observed = data['smokeday'].get('observed', False)
            for model in data['smokeday']:
                if model != 'observed':
                    predicted = data['smokeday'][model]
                    update_confusion_matrix(confusion_matrices[model], predicted, observed)
        
        # Drop the 'airnow' key from the confusion matrices
        confusion_matrices.pop('airnow', None)
 
        # Calculate precision and recall for each model
        metrics = {model: calculate_precision_recall(matrix) for model, matrix in confusion_matrices.items()}
        persistence_metrics = metrics.get('persistence', (1, 1))  # Set default to 1 for comparison

        # Generate LaTeX table for the city 
        latex_table = (
            "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{lcccccc}\n"
            "\\hline\nModel & True Positive & False Positive & False Negative & True Negative & Precision & Recall\\\\ \\hline\n"
        )
        for model, matrix in confusion_matrices.items():
            total_days = sum(matrix.values())
            precision, recall = metrics[model]
            if precision == "--":
                precision_str = "--"
            elif model == 'persistence':
                precision_str = f"{precision:.2f}"
            elif persistence_metrics[0] == '--':
                precision_str = f"{precision:.2f}"
            else: 
                precision_bg = 'green' if precision >= persistence_metrics[0] else 'red'
                precision_str = f"\\cellcolor{{{precision_bg}!25}}{precision:.2f}"
            if recall == "--":
                recall_str = "--"
            elif model == 'persistence':
                recall_str = f"{recall:.2f}"
            elif persistence_metrics[1] == '--':
                recall_str = f"{recall:.2f}"
            else:
                recall_bg = 'green' if recall >= persistence_metrics[1] else 'red'
                recall_str = f"\\cellcolor{{{recall_bg}!25}}{recall:.2f}"
            # Determine background color based on comparison with persistence
            # if model != 'persistence':
            #     precision_bg = 'green' if precision >= persistence_metrics[0] else 'red'
            #     recall_bg = 'green' if recall >= persistence_metrics[1] else 'red'
            #     precision_str = f"\\cellcolor{{{precision_bg}!25}}{precision:.2f}"
            #     recall_str = f"\\cellcolor{{{recall_bg}!25}}{recall:.2f}"
            # else:
            #     precision_str = f"{precision:.2f}"
            #     recall_str = f"{recall:.2f}"
            
            latex_table += (
                f"{model} & {matrix['TP']} ({np.round(matrix['TP']/total_days*100,2)}\\%) & "
                f"{matrix['FP']} ({np.round(matrix['FP']/total_days*100,2)}\\%) & "
                f"{matrix['FN']} ({np.round(matrix['FN']/total_days*100,2)}\\%) & "
                f"{matrix['TN']} ({np.round(matrix['TN']/total_days*100,2)}\\%) & "
                f"{precision_str} & {recall_str} \\\\ \n"
            )
        latex_table += "\\hline\n\\end{tabular}\n\\caption{Confusion Matrix for " + city.replace('_', ' ') + "}\n\\end{table}\n"
        
        # Ensure the city-specific directory exists within latex_tables
        city_latex_dir = os.path.join(tables_dir, city)
        os.makedirs(city_latex_dir, exist_ok=True)
        
        # Save the LaTeX table to a file
        with open(os.path.join(city_latex_dir, f'{short_city_names[city]}_confusion.tex'), 'w') as latex_file:
            latex_file.write(latex_table)

        print(f"LaTeX table saved for {city} at {os.path.join(city_latex_dir, 'table_confusion_matrix.tex')}")
