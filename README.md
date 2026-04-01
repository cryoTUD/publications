# Electron scattering properties of biological macromolecules

Here is a repository which contains all the scripts needed to view the figures from the paper "[Electron scattering properties of biological macromolecules and their use for cryo-EM map sharpening](https://doi.org/10.1039/D2FD00078D)" (2022) by Alok Bharadwaj, and Arjen J. Jakobi.


## Instructions

1) Create a environment variable to point this directory
export LOCAL_FARADAY_PATH=/home/path/to/this/folder

2) Pull the latest version of LocScale with the tag "faraday_discussions_prerelease_updated"
git checkout faraday_discussions_prerelease_updated

3) Have the conda environment "locscalev2" so that it contains all the modules required

4) run the script LOCAL_FARADAY_PATH+"\faraday_discussions\figure_scripts\general\print_all_figures.py"