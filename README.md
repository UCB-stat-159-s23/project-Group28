# Project Group 28: Airbnb Europe Dataset Analysis

This project aims to analyze the Airbnb Europe dataset, which is available on [Kaggle](https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset?resource=download). We explore various data cleaning, preprocessing, and modeling techniques to extract insights from the dataset and make predictions.

## Project Website
The project's JupyterBook website can be accessed [here](https://ucb-stat-159-s23.github.io/project-Group28/main.html)

## Repository Structure

The repository is structured as follows:

* 'data/': Contains the raw dataset and processed data files
* 'figures/': Contains the generated figures and plots
* 'results/': Contains the results obtained from model training and evaluation
* 'utils/': Contains utility functions and modules used throughout the project
* 'main.ipynb': Main project notebook, providing an overview of the analysis and results
* 'data_cleaning.ipynb': Notebook containing data cleaning and preprocessing steps
* 'model_building.ipynb': Notebook containing model training and evaluation steps
* 'environment.yml': Environment file with required packages for the project
* 'Makefile': Makefile to build JupyterBook for the repository and manage other tasks

## Setup and Installation

1. Clone this repository:
	git clone https://github.com/UCB-stat-159-s23/project-Group28.git
	cd project-Group28
	
2. Create and activate the aemf environment:
	mamba env create -f environment.yml --name aemf
	conda activate aemf
	
3. Install the IPython kernel with the aemf environment:
	python -m ipykernel install --user --name aemf --display-name "IPython - aemf"

## Usage

To build the JupyterBook website locally, run:
	make html

To clean up build files, run:
	make clean

To execute all notebooks, run:
	make all

## Launch Interactive Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s23/project-Group28.git/main)

## License

This project is licensed under the BSD 3-Clause License.
