import numpy as np
from utils import *
from scipy.optimize import minimize, fmin_cg
from time import time


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
            #cov_data, cov_ref = computeCovTest(data.T), computeCovTest(ref.T)
            cov_data, cov_ref = computeCovMat(data.T), computeCovMat(ref.T)
            last_min = np.inf
            cpt = 0
            n_iter_max = 50
            x = np.zeros(6)
            tol = 1e-6
            n = cov_data.shape[0]
            # def loss(args):
            #     R, T = RotMatrix(args[:3]), args[3:]
            #     M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])
            #     residual = ref.T - data.T @ R.T - T[None,:] # shape n*d
            #     tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
            #     loss_ = np.sum(residual * tmp)
            #     #losses = np.array([diff[i] @ np.linalg.solve(Cov_B[i] + R @ Cov_A[i] @ R.T, diff[i]) for i in range(Cov_A.shape[0])])
            #     print(loss_)
            #     return loss_
            # f = loss
            # x = minimize(f, np.zeros(6), method = 'CG').x
            #df = lambda x: grad_loss(x,data.T,ref.T,M)
            #out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)
            #x = out[0]

            while True:
                cpt = cpt+1
                R = RotMatrix(x[:3])
                M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])
                def loss(args):
                    R, T = RotMatrix(args[:3]), args[3:]
                    residual = ref.T - data.T @ R.T - T[None,:] # shape n*d
                    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
                    loss = np.sum(residual * tmp)
                    return loss

                f = loss

                df = lambda x: grad_loss(x,data.T,ref.T,M)
                #x = minimize(f, x, method = 'CG').x
                #f_min = f(x)
                out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)

                x = out[0]
                f_min = out[1]

                if last_min - f_min < tol:
                    print("1")
                    break
                elif cpt >= n_iter_max:
                    print("2")
                    break
                else:
                    last_min = f_min
            T = x[3:].reshape(-1,1)
            R = RotMatrix(x[:3])

        elif False:
            #Cov_B, Cov_A = computeCovMat(data.T), computeCovMat(ref.T)
            cov_data, cov_ref = computeCovTest(data.T), computeCovTest(ref.T)
            n = cov_data.shape[0]
            Cov_B, Cov_A = cov_data, cov_ref
            def loss(args):
                start = time()
                R, T = RotMatrix(args[:3]), args[3:]
                diff = ref.T - (data.T @ R.T + T[None,:])
                #M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])
                #center_matrix = np.array([np.linalg.inv(Cov_B[i] + R @ Cov_A[i] @ R.T) for i in range(Cov_A.shape[0])])
                #residual = ref.T - data.T @ R.T - T[None,:] # shape n*d
                #tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
                #loss = np.sum(residual * tmp)


                #loss_ = np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', center_matrix , diff))
                losses = np.array([diff[i] @ np.linalg.solve(Cov_B[i] + R @ Cov_A[i] @ R.T, diff[i]) for i in range(Cov_A.shape[0])])
                loss_ = np.sum(losses)
                print("loss = {}".format(loss_))
                #print("ended loss computation in {}s".format(time()-start))
                return loss_
            print("--- Minimizing ---")
            #grad = lambda args: grad_loss(args,a,b,M)
            args = minimize(loss, np.zeros(6)).x#, method = 'CG').x
            print("--- Solution found ---")
            R, T = RotMatrix(args[:3]), args[3:].reshape(-1,1)
        else:
            R, T = None
        return R, T