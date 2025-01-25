from utils import correlation, independence, pr
from itertools import combinations, permutations
import numpy as np

class Stage2():

    def __init__(self, Vc, V, Vc2V, num_variables, error=True):
        self.Vc = Vc
        self.X1 = {i: V[Vc2V[i][0]] for i in self.Vc}
        self.X2 = {i: V[Vc2V[i][1]] for i in self.Vc}
        self.n = num_variables
        self.error = error
        
        self.C2 = np.zeros([self.n, ])
        self.C4M2, self.sgnM = np.zeros([self.n, self.n]), np.zeros([self.n, self.n])
        self.order = []

    def FindRoot(self):
        min_p_value = {i:1.0 for i in self.Vc}
        for i in self.Vc:
            for j in self.Vc:
                if i == j or not correlation(self.X1[i], self.X2[j])[0]:
                    continue
                p_value = independence(pr(self.X1[j], self.X1[i], self.X2[i]), self.X2[i])[1]
                if p_value < min_p_value[i]:
                    min_p_value[i] = p_value
        root = max(min_p_value, key=min_p_value.get)
        if self.error and min_p_value[root] < 0.001:
            raise ValueError("Assumption might be invalid")
        self.order.append(root)
        return root
        
    def Update(self, root):
        self.C2[root] = np.cov(self.X1[root], self.X2[root])[0,1]
        for i in self.Vc:
            if root == i or not correlation(self.X1[root], self.X2[i])[0]:
                continue
            self.C4M2[i, root] = np.cov(self.X1[root], self.X2[i])[0,1] * np.cov(self.X2[root], self.X1[i])[0,1]
            self.sgnM[i, root] = np.sign(np.cov(self.X1[root], self.X2[i])[0,1] / np.cov(self.X1[i], self.X2[i])[0,1])
        
        for i in self.Vc:
            if root == i or not correlation(self.X1[root], self.X2[i])[0]:
                continue
            self.X1[i] = pr(self.X1[i], self.X1[root], self.X2[root])

        self.Vc.remove(root)
        del self.X1[root]
        del self.X2[root]

    def run(self):
        while len(self.Vc) > 0:
            root = self.FindRoot()
            self.Update(root)

        self.M = np.eye(self.n)
        for i in self.order:
            for j in self.order:
                if i == j:
                    continue
                self.M[i, j] = self.sgnM[i, j] * np.abs((self.C4M2[i, j] / (self.C2[i] * self.C2[j]))) ** 0.5
        A = np.eye(self.n) - np.linalg.inv(self.M)
        return np.float64(np.abs(A) > 0.25)
    