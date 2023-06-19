# Uncertainty - Under Construction
A repository for modified variance as uncertainty measure
by Saloua & Pigi

## Set up

### Envirioment

If you haven't created it already, create conda envirioment 
```
conda env create -f environment.yml
```
Activate envirioment
```
conda activate uncertainty
```

### Installation

Install python package for uncertainty:
```
python3 setup.py build install
```
Or if you plan to change anthing under uncertainty/uncertainty folder install editable version
```
pip install -e .
```
### Configuaration
Edit data directories and output at the configurations files for each dataset you want to use:
```
uncertainty
│   
└───scripts
    │   trento_config.py
    │   bcss_config.py
    │   signalModulation_config.py

```
### Structure
Save classification probabilities .npy files for which you want to calculate uncertainties in the following structure
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

Calculate and Save Uncertainty for all the ``` outputs/<dataset-name>/classification/*.npy``` files
```
python3 scripts/calculate_uncertainties.py
```

### Output structure
The uncertainties are should be saved in the following structure

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

Run analysis for trento, signal modulation and bcss

```
python3 scripts trento_analysis.py
python3 scripts signalModulation_analysis.py
python3 scripts bcss_analysis.py
```
Output plots and maps are saved under ``` outputs/<dataset-name>/images/*.npy ```

### To do
 - Logging and documentation
 - Investigate more efficient way to calculate maximum variance (dynamic programming?)
 - scikit-learn  Decision Trees modification for uncertainty loss


