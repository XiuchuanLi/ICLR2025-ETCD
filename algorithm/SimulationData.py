import numpy as np
import pandas as pd
from itertools import permutations

def ToBij():
    if np.random.randn(1) >= 0:
        signed = 1
    else:
        signed = -1
    return signed * np.random.uniform(.5, 2.0)


def Toa():
    return np.random.uniform(.5, 1)


def Data(Num=2000,seed=0):
    # Case 4 in Figure 9
    np.random.seed(seed)
    noise = np.load(f'../noise_{Num}.npy')[np.random.choice(np.arange(25), 11, replace=False)]
    while True:
        edge = [ToBij() for _ in range(13)] 
        adjacency = np.zeros([11, 11])
        adjacency[0, 2] = edge[0]
        adjacency[1, 2] = edge[2]
        adjacency[3, 2] = edge[1]
        adjacency[4, 0] = edge[4]
        adjacency[4, 2] = edge[3]
        adjacency[5, 0] = edge[5]
        adjacency[6, 0] = edge[6]
        adjacency[7, 0] = edge[7]
        adjacency[7, 1] = edge[8]
        adjacency[7, 4] = edge[9]
        adjacency[8, 1] = edge[10]
        adjacency[9, 1] = edge[11]
        adjacency[10, 5] = edge[12]
        mixing = np.linalg.inv(np.eye(11) - adjacency)
        if np.all(np.abs(mixing[np.abs(mixing) > 1e-6]) > 0.25): # faithfulness
            break
    x1=noise[0]*Toa()
    L1=noise[1]*Toa()+edge[0]*x1
    x2=noise[2]*Toa()+edge[1]*x1
    L2=noise[3]*Toa()+edge[2]*x1
    x3=noise[4]*Toa()+edge[3]*x1+edge[4]*L1
    x4=noise[5]*Toa()+edge[5]*L1
    x5=noise[6]*Toa()+edge[6]*L1
    x6=noise[7]*Toa()+edge[7]*L1+edge[8]*L2+edge[9]*x3
    x7=noise[8]*Toa()+edge[10]*L2
    x8=noise[9]*Toa()+edge[11]*L2
    x9=noise[10]*Toa()+edge[12]*x4

    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9'])
    return data, adjacency



def performance(gt, pre, num_observed):
    num_latent = len(gt) - num_observed
    num_pre_latent = len(pre) - num_observed
    result = (np.abs(num_latent - num_pre_latent), 0, 0, 0, 0)
    if len(pre) < num_observed:
        return result
    total_edge = np.sum(np.abs(gt) > 1e-6)
    total_order = np.sum(np.abs(np.linalg.inv(np.eye(len(gt)) - gt)) > 1e-6) - len(gt)
    total_pre_edge = np.sum(np.abs(pre) > 1e-6)
    # total_pre_order = np.sum(np.abs(np.linalg.inv(np.eye(len(pre)) - pre)) > 1e-6) - len(pre)
    if len(pre) < len(gt):
        temp = np.zeros([len(gt), len(gt)])
        temp[:len(pre), :len(pre)] = pre
        pre = temp
    gt_adjacency = gt
    gt_mixing = np.linalg.inv(np.eye(len(gt_adjacency)) - gt_adjacency)
    max_correct_edge, max_correct_order = 1e-6, 1e-6
    for latent_order in permutations(list(range(num_observed, len(pre))), num_latent):
        order = list(latent_order) + list(range(num_observed))
        pre_adjacency = pre[order, :][:, order]
        pre_mixing = np.linalg.inv(np.eye(len(pre_adjacency)) - pre_adjacency)
        correct_edge = np.sum(np.abs(pre_adjacency) * np.abs(gt_adjacency) > 1e-6)
        correct_order = np.sum(np.abs(pre_mixing) * np.abs(gt_mixing) > 1e-6) - len(gt_mixing)
        if (correct_edge > max_correct_edge) or (correct_edge == max_correct_edge and correct_order > max_correct_order):
            max_correct_edge, max_correct_order = correct_edge, correct_order
            result = (np.abs(num_latent - num_pre_latent), max_correct_order / total_order,
                       2 / (total_pre_edge / max_correct_edge + total_edge / max_correct_edge))
    return result
