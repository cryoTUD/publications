# Electron scattering properties of biological macromolecules

Here is a repository which contains all the scripts needed to view the figures from the paper "[Electron scattering properties of biological macromolecules and their use for cryo-EM map sharpening](https://doi.org/10.1039/D2FD00078D)" (2022) by Alok Bharadwaj, and Arjen J. Jakobi.


## Instructions

You can easily run these scripts by clicking Binder link below: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cryoTUD/publications/ElectronScattering_2022)

Alternatively, you can install the required pacakges using the _environment.yml_ file

conda env create -f environment.yml 

This creates a python environment faraday2022. Activate this. Either run the python script print_all_figures.py or go through all the Jupyter Notebooks inside the Notebooks folder. The first notebook allows you to run through all the analysis scripts to process the raw data to generate CSV files which are used to obtain the plots. 
