# On Measures Of Uncertainty In Classification
Geometry-based and homophily-based measures to assess the uncertainty of a pre-trained classifier.

## Repository Structure

The ```uncertainty/```  folder contains three subfolders: ```uncertainty/uncertainty/```, ```uncertainty/scripts/```, and ```uncertainty/outputs/```. The ```uncertainty/uncertainty/``` folder includes core functionality to calculate different measures of uncertainty for a defined dataset. The ```uncertainty/scripts/``` folder contains scripts used to calculate uncertainty on different datasets and perform analysis. Finally, the ```uncertainty/outputs/``` folder includes all classification classification outputs, uncertainty measurements, images and figures.

```
uncertainty
│   
└───uncertainty
│    │   
│    └───datasets
|
└───scripts
│   
└───outputs
```


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
## Data preparation 

This repository is made to run on the following datasets:

1. <ins> Trento dataset </ins>
2. <ins> Modulation classification dataset </ins>: The data produced from [this script](https://www.mathworks.com/help/deeplearning/ug/modulation-classification-with-deep-learning.html) with noise levels set to 15dB and 50dB.
3. <ins> BCSS dataset </ins>: Download the image TCGA-D8-A1JG-DX1_xmin15677_ymin69205_MPP-0.2500 and the corresponding mask from the [BCSS repository](https://github.com/PathologyDataScience/BCSS).

The repository is designed to run on these specific datasets. However, it is possible to adapt the code to work with other datasets. To do so, you can create a new dataset loading file under ```uncertainty/uncertainty/datasets```, create a configuration file under ```uncertainty/scripts/```, and import it in ```uncertainty/scripts/calculate_uncertainties.py```. This will allow you to use the code with the new dataset.

## Classification Outputs:

You have two options for obtaining classification outputs:

1. Download pre-generated outputs: You can download pre-computed classification outputs for the datasets mentioned above from [this Google Drive folder](https://drive.google.com/drive/u/0/folders/1XHM36H289swJfjZJSK0gbZiN9w_JTqfY). Copy ```outputs/``` folder inside ```uncertainty/``` repository. 
2. Use your own classification outputs: If you have your own classification outputs, ensure they have a similar structure as the pre-generated outputs.

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
### Classification structure
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

### Calculated uncertainties structure
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

## References 

