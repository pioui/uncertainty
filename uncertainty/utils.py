import numpy as np
import pickle


def normalize(x):
    """
    @param x : array(channels, N)
    @return : array(L, N) normalized x at [0,1] in channels dimention
    """

    # logger.info("Normalize to 0,1")
    x_min = x.min(axis=0)[0]  # [57]
    x_max = x.max(axis=0)[0]  # [57]
    xn = (x - x_min) / (x_max - x_min)
    assert np.unique(xn.min(axis=0)[0] == 0.0)
    assert np.unique(xn.max(axis=0)[0] == 1.0)
    return xn

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def print_latex_pm_table(mean, var):
    i=0
    for class_mean,class_var in zip(mean,var):
        print(f"& $c_{i}$ & ", end = '')
        for _mean, _var in zip(class_mean, class_var):
            print(f"{np.around(_mean*100, decimals=2)}±{np.around(_var*100, decimals=2)} & ", end = '')
        print("\\\\")
        i += 1
    print("")