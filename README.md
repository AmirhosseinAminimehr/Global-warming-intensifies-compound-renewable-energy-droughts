# Global warming intensifies compound renewable energy droughts

This repository contains the code used in the study **"Global warming intensifies compound renewable energy droughts."**

**Status:** Preprint (not yet published)

## Preprint

- **DOI:** https://doi.org/10.21203/rs.3.rs-9856664/v1
- **Research Square:** https://www.researchsquare.com/article/rs-9856664/v1

## Data

The analyses use the following publicly available datasets:

- **CMIP6 climate model outputs**, available through the Earth System Grid Federation (ESGF): https://metagrid.esgf-west.org/search
- **ERA5 reanalysis data**, available through the Copernicus Climate Change Service (C3S): https://climate.copernicus.eu/climate-reanalysis

Please download the required datasets from the original sources before running the code.

## Citation

If you use this repository, please cite the corresponding preprint:

> **Global warming intensifies compound renewable energy droughts**.
> Research Square (preprint).
> DOI: https://doi.org/10.21203/rs.3.rs-9856664/v1

## How to run the code

First, run `compute_red_metrics.py`. This script performs the required preprocessing and processing of the input climate data, including wind speed, solar radiation and temperature, and generates the renewable energy drought metrics required for the subsequent analyses and figures.

After `compute_red_metrics.py` has been completed, each analysis is organized in a separate folder. Within each analysis folder, the `data.py` script generates the numerical data required for the corresponding analysis and plots, while the `plot.py` script generates the figures presented in the manuscript.

The general workflow is therefore:

1. Run `compute_red_metrics.py` to preprocess the input data and calculate the RED metrics.
2. Navigate to the folder corresponding to the analysis of interest.
3. Run `data.py` to generate the numerical data required for the analysis and figures.
4. Run `plot.py` to generate the corresponding plots.
