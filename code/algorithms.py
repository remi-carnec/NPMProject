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

    def findBestTransform(self, data, ref, indices_data, indices_ref, x0, args):
        # Standard ICP
        if self.name == 'point2point':
            x = None
            R, T = best_rigid_transform(data[:,indices_data], ref[:,indices_ref])

        # Point to Plane
        elif self.name == 'point2plane':
            projMatrices = args['projMatrices'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T
            #def loss(x):
            #    R, T = RotMatrix(x[:3]), x[3:]
            #    diff = ref_0 - (data_0 @ R.T + T[None,:])
            #    return np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', projMatrices, diff))
            loss_f = lambda x: loss(x, ref_0, data_0, projMatrices)
            opt = 1
            if opt == 0:
                x = minimize(loss_f, np.zeros(6), method = 'CG').x
            else:
                grad_f = lambda x: grad_2(x, ref_0, data_0, projMatrices)
                x = minimize(loss_f, jac = grad_f, x0 =x0, method = 'CG').x
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)

        # Generalized ICP
        elif self.name == 'plane2plane' and True:
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
            x = x0
            tol = 1e-6
            ref_0, data_0 = ref.T[indices_ref], data.T[indices_data]
            prev_min_loss = np.inf#, loss(x, ref_0, data_0, None, True)
            iter, max_iter = 0, 20

            while True:
                iter = iter+1
                R = RotMatrix(x[:3])
                center_matrix = np.linalg.inv(cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T))

                loss_f = lambda x: loss(x, ref_0, data_0, center_matrix)
                grad_loss = lambda x: grad_relaxed(x, data_0, ref_0, center_matrix)
                res = minimize(loss_f, x0, jac = grad_loss, method = 'CG')
                x, min_loss = res.x, res.fun

                if abs(prev_min_loss - min_loss) < tol:
                    print("1")
                    break
                elif iter >= max_iter:
                    print("2")
                    break
                else:
                    prev_min_loss = min_loss
            T = x[3:].reshape(-1,1)
            R = RotMatrix(x[:3])

        elif self.name == "plane2plane" and False:
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T
            #def loss_pl2pl(x):
            #    R, T = RotMatrix(x[:3]), x[3:]
            #    diff = ref_0 - (data_0 @ R.T + T[None,:])
            #    center_matrix = np.linalg.inv(cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T))
            #    loss_ = np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', center_matrix , diff))
            #    print("loss = {}".format(loss_))
            #    return loss_
            #print("--- Minimizing ---")
            loss_f = lambda x: loss(x, ref_0, data_0, None, cov_ref, cov_data)
            df = lambda x: grad(x, a = data_0, b = ref_0, cov_a = cov_data, cov_b = cov_ref)
            out = fmin_cg(f = loss_f, x0 = x0, fprime = df, disp = False, full_output = True, maxiter=20)
            x = out[0]
            #x = minimize(loss, jac = df, x0 = x0, method = 'CG', options={'maxiter':20}).x
            #print("--- Solution found ---")
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)
        else:
            R, T = None
        return R, T, x

# test_loss = lambda t: (ref_0[idx] - t - R @ data_0[idx]) @ center_matrix[idx] @ (ref_0[idx] - t - R @ data_0[idx])
# test_grad_loss = lambda t: - 2 * center_matrix[idx] @ (ref_0[idx] - t - R @ data_0[idx])
#a,b,cov_a,cov_b = data_0, ref_0, cov_data, cov_ref