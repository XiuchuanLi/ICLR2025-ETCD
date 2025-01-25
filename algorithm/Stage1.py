from utils import independence, pr, Mergelist, quintuple
from itertools import combinations, permutations


class Stage1():

    def __init__(self, data):
        self.V = data
        self.Vc = list(range(len(self.V)))
        self.Vc2V = {i:(i, i) for i in range(len(self.V))} # two created children are two copies of itself
        self.V_children = {i:[] for i in range(len(self.V))}
        self.pointer = len(self.V)
    
    def FindIP(self):
        V1, V2 = {}, {}
        for i in self.Vc:
            V1[i] = self.V[self.Vc2V[i][0]]
            V2[i] = self.V[self.Vc2V[i][1]]
        IP = []
        if len(self.Vc) < 3:
            return IP
        for (i, j) in combinations(self.Vc, 2):
            for k in self.Vc:
                if k != i and k != j:
                    break
            flag = 1
            for l in self.Vc:
                if l == i or l == j:
                    continue
                if not independence(pr(V1[i], V1[j], V1[k]), V1[l], 0.001)[0]:
                    flag = 0
                    break
            if flag:
                IP.append([i, j])
        return IP

    def ClassifyIP(self, IP):
        V1, V2 = {}, {}
        for i in self.Vc:
            V1[i] = self.V[self.Vc2V[i][0]]
            V2[i] = self.V[self.Vc2V[i][1]]
        IP_I, IP_II, IP_III = [], [], []

        def is_IP_I(i, j):
            return independence(pr(V1[i], V1[j], V2[i]), V2[i])[0]
        
        def is_IP_III(i, j):
            Vc_other = self.Vc.copy()
            Vc_other.remove(i)
            Vc_other.remove(j)
            for k in Vc_other:
                if quintuple(V1[i], V2[i], V1[j], V1[k], V2[k]):
                    return True
            for (k, l) in permutations(Vc_other, 2):
                if quintuple(V1[i], V2[i], V1[j], V1[k], V1[l]):
                    return True
            return False

        # Find IP_I
        for (i, j) in IP:
            if is_IP_I(i, j):
                IP_I.append([i, j])
            elif is_IP_I(j, i):
                IP_I.append([j, i])
        
        # Find IP_II
        fre = {i:0 for i in self.Vc}
        for (i, j) in IP:
            fre[i] += 1
            fre[j] += 1
        for (i, j) in IP:
            if [i, j] not in IP_I and [j, i] not in IP_I:
                if fre[i] > 1 or fre[j] > 1:
                    IP_II.append([i, j])

        # Find IP_III
        for (i, j) in IP:
            if [i, j] not in IP_I + IP_II and [j, i] not in IP_I + IP_II:
                if is_IP_III(i, j):
                    IP_III.append([i, j])
                elif is_IP_III(j, i):
                    IP_III.append([j, i])
                else:
                    IP_II.append([i, j])
        
        IP_II = Mergelist(IP_II)
        return IP_I, IP_II, IP_III

    def update(self, IP_I, IP_II):
        for ip_i in IP_I:
            self.V_children[ip_i[0]].append(ip_i[1])
            if ip_i[1] in self.Vc:
                self.Vc.remove(ip_i[1])

        IP_I_flatten = []
        for ip_i in IP_I:
            IP_I_flatten += ip_i
        for ip_ii in IP_II:
            if set(IP_I_flatten) & set(ip_ii):
                continue
            for i in ip_ii:
                self.Vc.remove(i)
            self.Vc.append(self.pointer)
            self.Vc2V[self.pointer] = (self.Vc2V[ip_ii[0]][0], self.Vc2V[ip_ii[1]][0])
            self.V_children[self.pointer] = ip_ii
            self.pointer += 1

    def run(self):
        while True:
            IP = self.FindIP()
            IP_I, IP_II, IP_III = self.ClassifyIP(IP)
            if len(IP_I) == 0 and len(IP_II) == 0:
                break
            self.update(IP_I, IP_II)
        return self.V_children