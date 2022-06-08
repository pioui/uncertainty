import numpy as np

def normalize(x):
    """
    normalize at [0,1] in channels dimention

    x : array (channels, N)

    """
    # logger.info("Normalize to 0,1")
    x_min = x.min(axis=0)[0] # [57]
    x_max = x.max(axis=0)[0] # [57]
    xn = (x- x_min)/(x_max-x_min)
    assert np.unique(xn.min(axis=0)[0] == 0.)
    assert np.unique(xn.max(axis=0)[0] == 1.)
    return xn