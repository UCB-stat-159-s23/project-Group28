# Airbnb Europe Dataset Analysis

## Overview
This project aims to predict the prices of Airbnb listings in Europe using various regression models. The scientific motivation behind this project is to help both hosts and guests make more informed decisions when it comes to pricing and choosing accommodations. The analysis consists of data cleaning, exploratory data analysis, feature engineering, model building, and evaluation. 

## Dataset
The dataset used in this project comes from [Kaggle](https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset?resource=download) and contains information on Airbnb listings in Europe.

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
	`git clone https://github.com/UCB-stat-159-s23/project-Group28.git`
	`cd project-Group28`
	
2. Create and activate the aemf environment:
	`mamba env create -f environment.yml --name aemf`
	`conda activate aemf`
	
3. Install the IPython kernel with the aemf environment:
	`python -m ipykernel install --user --name aemf --display-name "IPython - aemf"`

## Usage

To build the JupyterBook website locally, run:

	`make html`

To clean up build files, run:

	`make clean`

To execute all notebooks, run:

	`make all`
	
## Testing
To run tests, navigate to the root directory of the project and execute the following command:
	`pytest`

## Launch Interactive Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group28.git/main)

## License

This project is licensed under the BSD 3-Clause License.

## Additional Information
For more detailed information about the analysis and results, please refer to the main narrative notebook available here [here](https://ucb-stat-159-s23.github.io/project-Group28/main.html)