from utils import *
from scipy.optimize import minimize

class Algorithm:
    def __init__(self, name, config_algo = dict()):
        if name == 'point-to-point' or name == 'point-to-plane' or name == 'plane-to-plane':
            self.name = name
            self.config_algo = config_algo
            if name == 'plane-to-plane':
                if 'relaxed_gradient' not in config_algo.keys():
                    self.config_algo['relaxed_gradient'] = False
                if 'max_iter' not in config_algo.keys():
                    self.config_algo['relaxed_gradient'] = 10
        else:
            raise ValueError("Error in initialization of algorithm: unknown name \"" + name + "\"")

    def findBestTransform(self, data, ref, indices_data, indices_ref, args, x0):
        # Standard ICP
        if self.name == 'point-to-point':
            x = None
            R, T = best_rigid_transform(data[:,indices_data], ref[:,indices_ref])

        # Point to Plane
        elif self.name == 'point-to-plane':
            # Initialize data
            projMatrices = args['projMatrices'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T

            # Find argmin using CG
            loss_f = lambda x: loss(x, ref_0, data_0, projMatrices)
            grad_f = lambda x: grad(x, ref_0, data_0, projMatrices)
            x = minimize(loss_f, jac = grad_f, x0 = np.zeros(6), method = 'CG').x
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)

        # Generalized ICP - Exact gradient
        elif self.name == "plane-to-plane" and not self.config_algo['relaxed_gradient']:
            # Store data
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
            ref_0, data_0 = ref[:,indices_ref].T, data[:,indices_data].T

            # Find argmin using exact CG
            loss_f = lambda x: loss(x, ref_0, data_0, None, cov_ref, cov_data)
            grad_loss = lambda x: grad(x, ref_0, data_0, None, cov_ref, cov_data)
            x = minimize(fun = loss_f, jac = grad_loss, x0 = x0, method = 'CG', options={'maxiter': self.config_algo['max_iter']}).x
            R, T = RotMatrix(x[:3]), x[3:].reshape(-1,1)

        # Generalized ICP - Relaxed gradient
        elif self.name == 'plane-to-plane' and self.config_algo['relaxed_gradient']:
            # Store data
            cov_data, cov_ref = args['cov_data'][indices_data], args['cov_ref'][indices_ref]
            ref_0, data_0 = ref.T[indices_ref], data.T[indices_data]

            # Initialize
            tol = 1e-6
            iter, max_iter = 0, 20
            x = np.zeros(6)

            # Find argmin with relaxed gradient method
            while True:
                # Fix central term
                iter = iter+1
                R = RotMatrix(x[:3])
                center_matrix = np.linalg.inv(cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T))

                # Perform optimization using relaxed CG
                loss_f = lambda x: loss(x, ref_0, data_0, center_matrix)
                grad_loss = lambda x: grad(x, ref_0, data_0, center_matrix)#grad_relaxed(x,data_0,ref_0,center_matrix)
                res = minimize(loss_f, x0, jac = grad_loss, method = 'CG', options={'maxiter': self.config_algo['max_iter']})
                x, min_loss = res.x, res.fun
                prev_min_loss = min_loss

                # Stopping rule: convergence or max_iter
                if abs(prev_min_loss - min_loss) < tol or iter >= max_iter:
                    break
            T = x[3:].reshape(-1,1)
            R = RotMatrix(x[:3])

        # Error
        else:
            raise ValueError('Error with name of algorithm')
        return R, T, x