import numpy as np
from ours.utils import independence
from itertools import combinations


def SelectPdf(Num,data_type="exp-non-gaussian"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 7

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":
        noise = np.random.exponential(scale=1.0, size=Num)

    else: #gauss
        noise = np.random.normal(0, 1, size=Num)

    return noise


def normalize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


for Num in [2000,5000,10000]:
    noises = []
    for i in range(25):
        print(i)
        while True:
            new_noise = normalize(SelectPdf(Num))
            if np.all(np.array([independence(new_noise, noise, 0.2)[0] for noise in noises])):
                noises.append(new_noise)
                break
    noises = np.stack(noises, axis=0)
    np.save(f'noise_{Num}.npy', noises)
