# Uncertainty
A repository for modified variance as uncertainty measure
by Saloua & Pigi

#### Create conda envirioment
```
conda env create -f environment.yml
conda activate ame

```

#### Install
```
python3 setup.py build install
```
#### Install - editable version
```
pip install -e .
```


#### Run SVM & RF Classification
```
python3 scripts/classification -d trento

python3 scripts/classification -d bcss
```

#### Calculate and Save Uncertainty maps for all the output/*.npy files
```
python3 scripts/calculate_uncertainties.py
```

Output .npy, logs and uncertainty images files are saved at :

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
        │   bcss.logs
        │   bcss.npy
        └───images
```

#### To do
 - Find maximum value for semantic_based_uncertainty and make an efficient calculation
 - investigate "distance" measurements 