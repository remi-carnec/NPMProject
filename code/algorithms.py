import numpy as np
from utils import *
from scipy.optimize import minimize


class Algorithm:
    def __init__(self, name):
        if name == 'point2point' or name == 'point2plane' or name == 'plane2plane':
            self.name = name
        else:
            self.name = None
            print("Error in initialization of algorithm: unknown name")

    def findBestTransform(self, data, ref, dataCloud):
        if self.name == 'point2point':
            R, T = best_rigid_transform(data, ref)
        elif self.name == 'point2plane':
            projMatrices = ProjMatrix(data.T, ref.T, dataCloud, 0.012, k = 10)
            def loss(args):
                R, T = RotMatrix(args[:3]), args[3:]
                diff = ref.T - (data.T @ R.T + T[None,:])
                return np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', projMatrices, diff))
            args = minimize(loss, np.zeros(6), method = 'CG').x
            R, T = RotMatrix(args[:3]), args[3:].reshape(-1,1)
        elif self.name == 'plane2plane':
            def loss(args):
                R, T = RotMatrix(args[:3]), args[3:]
                diff = ref.T - (data.T @ R.T + T[None,:])
                return np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', projMatrices, diff))
            args = minimize(loss, np.zeros(6), method = 'CG').x
            R, T = RotMatrix(args[:3]), args[3:].reshape(-1,1)
        else:
            R, T = None
        return R, T