# Airbnb Europe Dataset Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group28.git/main)

## Overview
This project aims to predict the prices of Airbnb listings in Europe using various regression models. The scientific motivation behind this project is to help both hosts and guests make more informed decisions when it comes to pricing and choosing accommodations. The analysis consists of data cleaning, exploratory data analysis, feature engineering, model building, and evaluation. 

## Dataset
The dataset used in this project is the [Airbnb Cleaned Europe Dataset](https://www.kaggle.com/dipeshkhemani/airbnb-cleaned-europe-dataset) available on Kaggle. The dataset contains listings information of Airbnb accommodations in various European cities. The dataset is cleaned and preprocessed, making it suitable for data analysis and modeling tasks.

The dataset consists of 18 columns and 322,330 rows. Each row represents a unique listing, and the columns contain various attributes related to the listing, such as the listing name, host details, location, property details, price, availability, and reviews.

To use this dataset, download the CSV file from the [Kaggle page](https://www.kaggle.com/dipeshkhemani/airbnb-cleaned-europe-dataset?resource=download) and place it in the appropriate directory within your project.

## Project Website
The project's JupyterBook website can be accessed [here](https://ucb-stat-159-s23.github.io/project-Group28/main.html)

## Repository Structure

The repository is structured as follows:

* `data/`: Contains the raw dataset and processed data files
* `figures/`: Contains the generated figures and plots
* `results/`: Contains the results obtained from model training and evaluation
* `utils/`: Contains utility functions and modules used throughout the project
* `main.ipynb`: Main project notebook, providing an overview of the analysis and results
* `data_cleaning.ipynb`: Notebook containing data cleaning and preprocessing steps
* `model_building.ipynb`: Notebook containing model training and evaluation steps
* `environment.yml`: Environment file with required packages for the project
* `Makefile`: Makefile to build JupyterBook for the repository and manage other tasks

## Setup and Installation

1. Clone this repository:
```python
git clone https://github.com/UCB-stat-159-s23/project-Group28.git
cd project-Group28
```
2. Create and activate the aemf environment:
```python
mamba env create -f environment.yml --name aemf
conda activate aemf
```
3. Install the IPython kernel with the aemf environment:
```python
python -m ipykernel install --user --name aemf --display-name "IPython - aemf"
```

## Usage

To build the JupyterBook website locally, run:

	make html

To clean up build files, run:

	make clean

To execute all notebooks, run:

	make all
	
## Package Structure

The `pipeline_utils` package is a collection of functions that facilitate the creation, evaluation, and display of results for different model pipelines. These pipelines consist of combinations of various imputers and models. The package includes the following functions:

1. `create_pipelines()`: This function creates a dictionary of pipelines with combinations of different imputers and models. It returns a dictionary with keys as imputer+model names and values as pipeline objects.

2. `create_summary(valid_errs, y_valid, ypred_valid)`: This function creates a summary table of validation errors, mean squared error (MSE), and mean absolute error (MAE) for different models. It takes a dictionary containing validation errors for different models, true target values for validation data, and a dictionary containing predicted target values for validation data from different models. It returns a summary table with model names, validation errors, MSE, and MAE.

3. `calculate_metrics(y_true, y_pred)`: This function calculates mean squared error (MSE), mean absolute error (MAE), and R-squared score for the given true and predicted target values. It takes true target values and predicted target values as input and returns a tuple containing MSE, MAE, and R-squared score.

4. `display_results(mse, mae, r2, model_name='knn_imputer+rf')`: This function creates a table to display evaluation metrics for a given model. It takes mean squared error, mean absolute error, R-squared score, and an optional model name as input. It returns a table displaying evaluation metrics for the given model.

The package uses the following external libraries:
- `sklearn`: For imputers, models, pipeline construction, and evaluation metrics.
- `pandas`: For creating and manipulating dataframes.

## Testing
To run tests, navigate to the root directory of the project and execute the following command:
	`PYTHONPATH=./ pytest`

## License

This project is licensed under the BSD 3-Clause License.

## Additional Information
For more detailed information about the analysis and results, please refer to the main narrative notebook available here [here](https://ucb-stat-159-s23.github.io/project-Group28/main.html).
