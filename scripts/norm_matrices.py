import numpy as np
#import print_options

np.set_printoptions(precision=2)

A_trento = [[0 ,  3.28 , 2.46 , 3.86 , 1.21 , 3.54],
            [3.28 , 0 , 3.4 , 3.18 , 2.81 , 3.13],
            [2.46 , 3.4 ,  0 , 1.94 , 2.11 , 2.65],
            [3.86 , 3.18 , 1.94 , 0 , 2.74 , 2.47],
            [1.21 , 2.81 , 2.11 , 2.74 , 0 , 3.64],
            [3.54 , 3.13 , 2.65 , 2.47 , 3.64 ,0]]

A_bcc = [
        [0, 2, 2, 2, 3],
        [2, 0, 1, 1, 3],
        [2, 1, 0, 1, 3],
        [2, 1, 1, 0, 3],
        [3, 3, 3, 3, 0],
    ]

print(A_bcc/np.amax(A_bcc))