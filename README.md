# Uncertainty - Under Construction
A repository for modified variance as uncertainty measure
by Saloua & Pigi

#### Create conda envirioment
```
conda env create -f environment.yml
conda activate uncertainty
```

#### Install
```
python3 setup.py build install
```
#### Install - editable version
```
pip install -e .
```

#### Edit data directories and output at the configurations files :

```
uncertainty
│   
└───scripts
    │   trento_config.py
    │   bcss_config.py
    │   signalModulation_config.py

```

#### Save classification probabilities .npy files in the following structure
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

#### Calculate and Save Uncertainty for all the ``` outputs/<dataset-name>/classification/*.npy``` files
```
python3 scripts/calculate_uncertainties.py
```
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

#### Analysis for trento, signal modulation and bcss

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


