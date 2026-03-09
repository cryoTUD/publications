# LocScale 2.0

Here is a repository which contains all the scripts needed to view the figures from the paper "[Confidence-guided cryo-EM map optimisation with LocScale-2.0](https://www.biorxiv.org/content/10.1101/2025.09.11.674726v1.full)" (2025) by Alok Bharadwaj, Reinier de Bruin, and Arjen J. Jakobi.

## Run on Binder
Click the following link to run the script directly on Binder. This automatically loads the required environment from the environment.yml file. Opening this for the first time requires about ten minutes of time (depending on server load). 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cryoTUD/publications/LocScale_2025?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F)

## Run locally
You can find the Jupyter Notebooks present under "notebooks" folder. We recommend running them on VS Code. [Click here for a quick introduction](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

### Steps: 
#### 1. Install environment
We use conda environment manager to prepare an environment. Miniconda or Miniforge or Mamba works with the provided environment file. [Click here to install miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

Use the environment.yml file to install a conda environment locally. Use the command line: 
```
conda env create -f environment.yml -y
```

This installs an environment ```locscale2_scripts``` with all the required packages. 
#### 2. Run the notebooks
Select the right kernel and run the notebook. Each notebook is named after the figure it corresponds to. Click the very first cell on top which downloads the plot data from this [surfdrive link](https://edu.nl/e784n). It then downloads the folder **LocScale2.0_NComms_2026_plotData** inside the notebooks directory. The variable _PLOT_DATA_STORE_PATH_ in the notebooks should point to this folder. If download does not work as intended, you can download the folder manually from the Surfdrive link and redefine this variable in the notebooks. 


Click through all the code cells to get the figures on the screen.

That's it! 


