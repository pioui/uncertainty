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
    │   singal_modulation_config.py

```



#### Run SVM & RF Classification
```
python3 scripts/classification.py -d trento
python3 scripts/classification.py -d bcss
python3 scripts/classification.py -d sm
```

#### Calculate and Save Uncertainty maps for all the output/*.npy files
```
python3 scripts/calculate_uncertainties.py
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
    └───signal_modulation
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
outputs/classification/dataset_classifier.npy
outputs/uncertainty/dataset_classifier.npy
outputs/images/dataset_classifier_map.eps
outputs/images/dataset_classifier_plots.eps
```
 OR
```
outputs/dataset/classification/dataset_classifier.npy
outputs/dataset/uncertainty/dataset_classifier.npy
outputs/dataset/images/dataset_classifier_map.eps
outputs/dataset/images/dataset_classifier_plots.eps
```
#### scripts structure:
``` classification.py ``` -> not neccessary to include all classfication scripts (since they diverge a lot and it not straight forward to access the dataset to run them) we can make available our classification's predictions.

``` calculate_uncertainties.py ```-> reads a file (in outputs/classification) with .npy predictions' probabilities and saves a new .npy in the outputs/uncertainties folder (or somewhere else) - no plots or maps since this depends on the dataset.

``` dataset1_config.py ``` -> configurations for dataset 1

``` dataset1_analysis.py ``` -> analysis for dataset 1

``` dataset1_config.py ``` -> configurations for dataset 2

``` dataset1_analysis.py ``` -> analysis for dataset 2




