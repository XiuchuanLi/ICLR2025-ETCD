import SimulationData as SD
from Stage1 import Stage1
from Stage2 import Stage2
import time
import numpy as np
import pandas as pd


def integrate(result1, result2):
    for i in result1:
        for j in result1[i]:
            result2[j, i] = 1
    return result2


for seed in range(10):
    print(seed)
    data, A = SD.Data(2000, seed)
    data = data.values.T
    start = time.perf_counter()
    S1 = Stage1(data)
    result1 = S1.run()
    S2 = Stage2(S1.Vc, S1.V, S1.Vc2V, S1.pointer, error=True)
    # If you want a trustworthy algorithm, let error=True
    # If you want the algorithm to always output a causal graph even if the assumption is invalid, let error=False
    result2 = S2.run()
    end = time.perf_counter()
    hatA = integrate(result1, result2)
    num_observed = len(data)
    num_latent = len(hatA) - num_observed
    adjacency = pd.DataFrame(hatA.astype(np.int8), 
                             columns = [f'o{i}' for i in range(1, num_observed+1)] + [f'l{i}' for i in range(1, num_latent+1)],
                             index = [f'o{i}' for i in range(1, num_observed+1)] + [f'l{i}' for i in range(1, num_latent+1)],)
    print('adjacency matrix:')
    print(adjacency)
    result = SD.performance(A, hatA, len(data))
    print(f'Error in Latent Variable: {result[0]:.1f}')
    print(f'Correct-Ordering Rate: {result[1]:.2f}')
    print(f'F1-Score: {result[2]:.2f}')
    print(f'Running Time: {end - start:.2f}')
    print('\n')
