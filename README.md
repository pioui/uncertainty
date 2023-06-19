# Uncertainty - Under Construction
This repository contains a modified variance as an uncertainty measure, developed by Saloua & Pigi.

## Set up

### Envirioment

If you haven't created it already, please create a conda environment using the following command
```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate uncertainty
```

### Installation

To install the Python package for uncertainty, execute the following command:
```
python3 setup.py build install
```
Alternatively, if you plan to make changes under the uncertainty/uncertainty folder, install the editable version:
```
pip install -e .
```
### Configuaration
Edit the data directories and output configurations in the respective configuration files for each dataset you want to use:
```
uncertainty
│   
└───scripts
    │   trento_config.py
    │   bcss_config.py
    │   signalModulation_config.py

```
### Structure
Save the classification probability .npy files for which you want to calculate uncertainties in the following structure:
```
uncertainty
│   
└───outputs
    │   
    └───<dataset-name1>
    │   └───classification
    |       |    <dataset-name1>_<classifier_name1>.npy
    |       |    <dataset-name1>_<classifier_name2>.npy
    |
    └───<dataset-name2>
    │   └───classification
    |       |    <dataset-name2>_<classifier_name1>.npy
    |       |    <dataset-name2>_<classifier_name2>.npy

```
## Uncertainty evaluation

To calculate and save uncertainties for all the ```outputs/<dataset-name>/classification/*.npy``` files, run the following command:
```
python3 scripts/calculate_uncertainties.py
```

### Output structure
The uncertainties should be saved in the following structure:
```
uncertainty
│   
└───outputs
    │   
    └───<dataset-name1>
    │   └───uncertainties
    |       └───<classifier_name1>
    |       |    |     <dataset-name1>_<classifier_name1>_<uncertainty-measurement1>.npy
    |       |    |     <dataset-name1>_<classifier_name1>_<uncertainty-measurement2>.npy
    |       |
    |       └───<classifier_name2>
    |            |     <dataset-name1>_<classifier_name2>_<uncertainty-measurement1>.npy
    |            |     <dataset-name1>_<classifier_name2>_<uncertainty-measurement2>.npy

```

### Analysis 
To run analysis for Trento, signal modulation, and BCSS, execute the following commands:
```
python3 scripts trento_analysis.py
python3 scripts signalModulation_analysis.py
python3 scripts bcss_analysis.py
```
The output plots and maps will be saved under ```outputs/<dataset-name>/images/*.npy```.
### To do
 - Logging and documentation
 - Investigate more efficient way to calculate maximum variance (dynamic programming?)
 - scikit-learn  Decision Trees modification for uncertainty loss


