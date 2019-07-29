# MsTMIP-Data-Analysis
Summer Research Project on MsTMIP Data Analysis

## Overview
MsTMIP stands for the "Multi-scale Synthesis and Terrestrial Model Intercomparison Project",  and was a set of common experiments run by 15 independent land surface/ecosystem models. These models are capable of assimilating climate information, and producing a range of diagnostic outputs to help us evaluate vegetation and ecosystem responses, and their feedbacks to the surface climate globally.
Specifically, we analyze the data in East Africa protected areas (e.g. Serengeti) as there could be potential policy impact for the Great Migration and other environemental issues. 

## Data
A full description of this dataset can be found [here](https://daac.ornl.gov/NACP/guides/NACP_MsTMIP_TBMO.html).

## Research Questions
1. How well do the DGVMs/TBMs model data represent seasonal vegetation changes in East Africa protected areas (e.g. Serengeti)
2. What are the biophysical mechanisms behind these seasonal vegetation changes?

## Methods
1. Ensemble Empirical Mode Decomposition (EEMD, Wu and Huang 2009): time series trend decomposition, using noise-assisted data analysis
2. Surrogate Testing (IAAFT, Schreiber and Schmitz, 1996): iteratively refined amplitude adjusted Fourier transform method
