# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from tqdm import tqdm

# For Point-to-plane and Plane-to-plane
def compute_eigen_data(data, ref, algo, k = 10, eps = 1e-3):
    # Compute eigenvectors, projection matrices
    tree = KDTree(ref)
    dist, indices = tree.query(ref, k = k)
    all_eigenvectors = PCA(ref, indices)
    projMatrices = np.einsum('ij,ik->ijk', all_eigenvectors[:,0], all_eigenvectors[:,0])
    args = {'eigenvectors': all_eigenvectors, 'projMatrices': projMatrices}

    # Covariance matrices
    if algo.name == 'plane-to-plane':
        args['cov_data'] = computeCovMat(data, None, k = k, eps = eps)
        args['cov_ref'] = computeCovMat(ref, all_eigenvectors, k = k, eps = eps)

    return tree, args

def computeCovMat(cloud, all_eigenvectors = None, k = 10, eps = 1e-3):
    # Compute neighborhoods
    tree = KDTree(cloud)
    dist, indices = tree.query(cloud, k = k)

    # Compute eigenvectors if not provided
    if all_eigenvectors is None:
        all_eigenvectors = PCA(cloud, indices)

    # Compute rotated covariance matrix
    I_eps = np.diag([eps, 1, 1])
    Cov = np.einsum('ijk,kli->ijl', np.einsum('ijk,kl->ijl', all_eigenvectors, I_eps), all_eigenvectors.T)
    return Cov

def PCA(cloud, indices):
    neighborhoods = cloud[indices]
    centered_points = neighborhoods - np.mean(neighborhoods, axis = 1)[:,None,:]
    cov = np.einsum('ijk,ikl->ijl',centered_points.swapaxes(1,2), centered_points)/centered_points.shape[1]
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))
    for k in tqdm(range(cloud.shape[0]), desc = 'Eigenvectors'):
        all_eigenvectors[k] = np.linalg.eigh(cov[k])[1]
    return all_eigenvectors

def RotDecompo(thetas):
    R_x = np.array([[1,0,0], [0, np.cos(thetas[0]), -np.sin(thetas[0])], [0, np.sin(thetas[0]), np.cos(thetas[0])]])
    R_y = np.array([[np.cos(thetas[1]),0,np.sin(thetas[1])], [0, 1, 0], [-np.sin(thetas[1]),0,np.cos(thetas[1])]])
    R_z = np.array([[np.cos(thetas[2]),-np.sin(thetas[2]), 0], [np.sin(thetas[2]),np.cos(thetas[2]), 0], [0, 0, 1]])
    return R_x, R_y, R_z

def RotMatrix(thetas):
    R_x, R_y, R_z = RotDecompo(thetas)
    return R_z @ R_y @ R_x

def computeGradRot(thetas):
    grad = np.zeros((3,3,3))
    R_x, R_y, R_z = RotDecompo(thetas)
    grad_x = np.array([[0,0,0], [0, -np.sin(thetas[0]), -np.cos(thetas[0])], [0, np.cos(thetas[0]), -np.sin(thetas[0])]])
    grad_y = np.array([[-np.sin(thetas[1]),0,np.cos(thetas[1])], [0, 0, 0], [-np.cos(thetas[1]),0,-np.sin(thetas[1])]])
    grad_z = np.array([[-np.sin(thetas[2]),-np.cos(thetas[2]), 0], [np.cos(thetas[2]),-np.sin(thetas[2]), 0], [0, 0, 0]])
    grad[0], grad[1], grad[2] = R_z @ R_y @ grad_x, R_z @ grad_y @ R_x, grad_z @ R_y @ R_x
    return grad

def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    p = np.mean(ref, axis = 1)
    p_prime = np.mean(data, axis = 1)
    Q = ref - p[:,None]
    Q_prime = data - p_prime[:,None]
    H = Q_prime @ Q.T
    U, S, V = np.linalg.svd(H)

    R = V.T @ U.T
    T = p - R @ p_prime
    T = T.reshape(-1,1)
    return R, T

def loss(x, ref_0, data_0, center_matrix = None, cov_ref = None, cov_data = None):
    '''
    Value of the loss for parameter x = (theta, t)
    loss(x) = sum_i d_i^T center_matrix d_i
    '''
    R, T = RotMatrix(x[:3]), x[3:]
    diff = ref_0 - (data_0 @ R.T + T[None,:])
    if center_matrix is None:
        inv_center_matrix = cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T)
        loss_ = np.einsum('ij,ij', diff, np.linalg.solve(inv_center_matrix, diff))
    else:
        loss_ = np.einsum('ij,ij', diff, np.einsum('ijk,ik -> ij', center_matrix , diff))
    return loss_

def grad(x, ref, data, center_matrix = None, cov_ref = None, cov_data = None):
    """
    Gradient of the loss for parameter x = (theta, t).
    The user can choose to fix the center matrix (e.g. Point-to-plane)
    By default, is G-ICP central matrix
    """
    # Initialize transformation and gradient
    t = x[3:]
    R = RotMatrix(x[:3])
    g = np.zeros(6)

    # Calculate error
    diff = ref - data @ R.T -t[None,:]

    # Compute center_matrix^{-1} * diff
    if center_matrix is None:
        inv_center_matrix = cov_ref + np.einsum('ik,jkl,lm->jim', R, cov_data, R.T)
        tmp = np.linalg.solve(inv_center_matrix, diff)
    else:
        tmp = np.einsum('ijk,ik->ij', center_matrix, diff)

    # Compute gradient of l(R,t)
    g[3:] = - 2 * np.einsum('ij->j',tmp)
    grad_R = - 2 * np.einsum('ij,il->jl', tmp, data)
    if cov_data is not None:
        outer = np.einsum('ij,il->ijl', tmp, tmp)
        grad_R -= 2 * np.einsum('ijk,kl,ilm->jm', cov_data, R.T, outer).T

    # Apply chain rule (euler angles)
    grad_angles = computeGradRot(x[:3])
    g[:3] = np.sum(grad_R[None,:,:] * grad_angles, axis = (1,2))
    return g