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

#### Calculate and Save Uncertainty for all the outputs/<dataset-name>/classification/*.npy files
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
    |             |    <dataset-name1>_<classifier_name1>_<uncertainty-measurement1>.npy
    |             |    <dataset-name1>_<classifier_name1>_<uncertainty-measurement2>.npy
    |       └───<classifier_name2>
    |             |    <dataset-name1>_<classifier_name2>_<uncertainty-measurement1>.npy
    |             |    <dataset-name1>_<classifier_name2>_<uncertainty-measurement2>.npy

```

### Output .npy, logs and uncertainty images files are saved at :

```
uncertainty
│   
└───outputs
    │   
    └───trento
    │   │   trento.logs
    │   │   trento.npy
    │   └───images
    │   
    └───bcss
    |   │   bcss.logs
    |   │   bcss.npy
    |   └───images
    |
    └───signalModulation
        │   ignal_modulation.logs
        │   ignal_modulation.npy
        └───images
```

### To do
 - Logging and documentation
 - Investigate more efficient way to calculate maximum variance (dynamic programming?)
 - scikit-learn  Decision Trees modification for uncertainty loss

#### outputs directory:

```
outputs/dataset-name/classification/dataset-name_classifier-name.npy
outputs/dataset-name/uncertainties/dataset-name_classifier-name_uncertainty-name.npy
outputs/dataset-name/images/ddataset-name_classifier-name_uncertainty-name_map.eps
outputs/dataset-name/images/dataset-name_classifier-name_plots.eps
```
#### scripts structure:
``` classification.py ``` -> not neccessary to include all classfication scripts (since they diverge a lot and it not straight forward to access the dataset to run them) we can make available our classification's predictions.

``` calculate_uncertainties.py ```-> reads a file (in outputs/classification) with .npy predictions' probabilities and saves a new .npy in the outputs/uncertainties folder (or somewhere else) - no plots or maps since this depends on the dataset.

``` dataset1_config.py ``` -> configurations for dataset 1

``` dataset1_analysis.py ``` -> analysis for dataset 1

``` dataset1_config.py ``` -> configurations for dataset 2

``` dataset1_analysis.py ``` -> analysis for dataset 2




