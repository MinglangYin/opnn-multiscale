import numpy as np
import matplotlib.pyplot as plt

"""
data shape: [11, 11, 9118, 16]
11, 11: boundary force

"""

def read_npy(path):
    data = np.load(path)
    num_x, num_y = data.shape[0], data.shape[1]
    stress_11 = []
    stress_22 = []
    stress_12 = []
    disp_11 = []
    disp_22 = []
    disp_12 = []
    
    for id_x in range(0, num_x):
        for id_y in range(0, num_y):
            stress_11.append(data[id_x, id_y, :, ])

    dataset = {}
    # dataset = {
    #     "node_id": data[:, :, ]
    # }
    return dataset


if __name__ == "__main__":
    path = "VoidPlasticity_1.npy"
    dataset = read_npy(path)
    
