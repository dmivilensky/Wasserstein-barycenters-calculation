import numpy as np

class WassersteinBarycenterCalculator:
    def __init__(self, gamma, epsilon, n):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.n       = n        

    def gradient(self, q, c, _lambda):
        return (q.repeat(self.n).reshape(-1, self.n) *\
                np.exp((-c + _lambda.repeat(self.n).reshape(-1, self.n) / self.gamma)))
    