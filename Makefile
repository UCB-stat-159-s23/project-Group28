## Makefile to build JupyterBook for this repository

.PHONY: env
env :
	mamba env create -f environment.yml --name aemf
	bash -ic 'conda activate aemf;python -m ipykernel install --user --name aemf --display-name "IPython - aemf"'

	
all:
	bash -ic 'conda activate aemf'
	jupyter execute data_cleaning.ipynb data_visualization.ipynb model_building.ipynb main.ipynb

## - html    : Build static website for local display
.PHONY: html
html:
	jupyter-book build .
	
## - clean   : remove all build files
.PHONY: clean
clean:
	rm -f figures/*.png
	rm -f results/*.csv
	rm -rf _build/html/

