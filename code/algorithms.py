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

    def findBestTransform(self, data, ref, indices_data, indices_ref, args):
        # Standard ICP
        if self.name == 'point2point':
            R, T = best_rigid_transform(data[:,indices_data], ref[:,indices_ref])

        # Point to Plane
        elif self.name == 'point2plane':
            projMatrices = args['projMatrices'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T
            def loss(x):
                R, T = RotMatrix(x[:3]), x[3:]
                diff = ref_0 - (data_0 @ R.T + T[None,:])
                return np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', projMatrices, diff))
            opt = 1
            if opt == 0:
                x = minimize(loss, np.zeros(6), method = 'CG').x
            else:
                df = lambda x: grad_2(x, ref_0, data_0, projMatrices)
                #out = fmin_cg(f = loss, x0 = np.zeros(6), fprime = df, disp = False, full_output = True)#, maxiter=20)
                #x = out[0]
                x = minimize(loss, jac = df, x0 = np.zeros(6), method = 'CG').x
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)

        # Generalized ICP
        elif self.name == 'plane2plane' and False:
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
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
            ref_0, data_0 = ref.T[indices_ref], data.T[indices_data]
            while True:
                cpt = cpt+1
                R = RotMatrix(x[:3])
                M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])
                def loss(x):
                    R, T = RotMatrix(x[:3]), x[3:]
                    residual = ref_0 - data_0 @ R.T - T[None,:] # shape n*d
                    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
                    loss = np.sum(residual * tmp)
                    return loss

                f = loss

                df = lambda x: grad_loss(x,data_0,ref_0,M)
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

        elif self.name == "plane2plane":
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T
            def loss(x):
                R, T = RotMatrix(x[:3]), x[3:]
                diff = ref_0 - (data_0 @ R.T + T[None,:])
                center_matrix = np.linalg.inv(cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T))
                loss_ = np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', center_matrix , diff))
                print("loss = {}".format(loss_))
                return loss_
            #print("--- Minimizing ---")
            df = lambda x: grad(x, a = data_0, b = ref_0, cov_a = cov_data, cov_b = cov_ref)
            out = fmin_cg(f = loss, x0 = np.zeros(6), fprime = df, disp = False, full_output = True, maxiter=20)
            x = out[0]
            #x = minimize(loss, x0 = np.zeros(6), method = 'CG').x
            #print("--- Solution found ---")
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)
        else:
            R, T = None
        return R, T

# test_loss = lambda t: (ref_0[idx] - t - R @ data_0[idx]) @ center_matrix[idx] @ (ref_0[idx] - t - R @ data_0[idx])
# test_grad_loss = lambda t: - 2 * center_matrix[idx] @ (ref_0[idx] - t - R @ data_0[idx])
#a,b,cov_a,cov_b = data_0, ref_0, cov_data, cov_ref