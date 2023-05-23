'''
This script runs the classification and uncertainty evaluation of BCSS dataset
'''
#python3 scripts/classification.py -d trento
python3 scripts/classification.py -d bcss

python3 scripts/class_svm.py -d bcss

#python3 scripts/calculate_uncertainties.py