import numpy as np
from scipy.stats import pearsonr
from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from itertools import combinations


def correlation(x, y, alpha=0.01):
    rho, p_value = pearsonr(x, y)
    if p_value < alpha:
        return True, p_value
    else:
        return False, p_value

        
def independence(x, y, alpha=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelY = GaussianKernel(float(1.0))
    kernelX=GaussianKernel(float(1.0))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=30, num_rfy=30, num_nullsims=1000)
    p_value = myspectralobject.compute_pvalue(x, y)

    if p_value > alpha:
        return True, p_value
    else:
        return False, p_value


def pr(x1,x2,x3):
    return x1 - np.cov(x1,x3)[0,1] / np.cov(x2,x3)[0,1] * x2


def quintuple(x1, x2, x3, x4, x5):
    A = np.array([[np.cov(x3, x5)[0, 1], np.cov(x4, x5)[0, 1]], [np.cov(x3, x2)[0, 1], np.cov(x4, x2)[0, 1]]])
    b = -np.array([[np.cov(x1,x5)[0, 1]], [np.cov(x1,x2)[0, 1]]])
    coefficient = (np.linalg.inv(A) @ b).flatten().tolist()
    return independence(x1 + coefficient[0] * x3 + coefficient[1] * x4, x2, 0.05)[0]


def Mergelist(L2):

    def bfs(graph, start):
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(graph[vertex] - visited)
        return visited

    def connected_components(G):
        seen = set()
        for v in G:
            if v not in seen:
                c = set(bfs(G, v))
                yield c
                seen.update(c)

    def graph(edge_list):
        result = {}
        for source, target in edge_list:
            result.setdefault(source, set()).add(target)
            result.setdefault(target, set()).add(source)
        return result

    l=L2.copy()
    edges = []
    s = list(map(set, l))
    for i, j in combinations(range(len(s)), r=2):
        if s[i].intersection(s[j]):
            edges.append((i, j))
    G = graph(edges)
    result = []
    unassigned = list(range(len(s)))
    for component in connected_components(G):
        union = set().union(*(s[i] for i in component))
        result.append(sorted(union))
        unassigned = [i for i in unassigned if i not in component]
    result.extend(map(sorted, (s[i] for i in unassigned)))
    return result



