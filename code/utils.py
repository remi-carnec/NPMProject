# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


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

def PCA(points):
    # Center points
    centered_points = points - np.mean(points, axis = 0)

    # Compute Cov matrix
    N = points.shape[0]
    cov = (centered_points.T @ centered_points)/N

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors

def compute_eigen_data(data, ref, algo, k = 10):
    # Compute eigenvectors, projection matrices
    tree = KDTree(ref)
    dist, indices = tree.query(ref, k = k)
    all_eigenvalues = np.zeros((ref.shape[0], 3))
    all_eigenvectors = np.zeros((ref.shape[0], 3, 3))
    for k in tqdm(range(ref.shape[0]), desc = 'Eigenvectors'):
        eigenvalues, eigenvectors = PCA(ref[indices[k]])
        all_eigenvalues[k] = eigenvalues
        all_eigenvectors[k] = eigenvectors
    projMatrices = np.einsum('ij,ik->ijk', all_eigenvectors[:,0], all_eigenvectors[:,0])
    args = {'eigenvalues': all_eigenvalues, 'eigenvectors': all_eigenvectors, 'projMatrices': projMatrices}

    # Covariance matrices
    if algo.name == 'plane2plane':
        normal = False
        if normal:
            args['cov_data'] = computeCovMat(data)
            args['cov_ref'] = computeCovMat(ref)
        else:
            args['cov_data'] = computeCovTest(data)
            args['cov_ref'] = computeCovTest(ref)

    return tree, args


def compute_local_PCA(query_points, cloud_points, dataCloud, radius, k = None, showProgress = True):
    # Compute neighbors using radius or N-nearest neighbors
    tree = KDTree(cloud_points)
    if k == None:
        indices = tree.query_radius(query_points, r = radius)
    else:
        dist, indices = tree.query(query_points, k = k)

    # Compute eigenvectors and eigenvalues for each query point
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    range_ = tqdm(range(query_points.shape[0])) if showProgress else range(query_points.shape[0])
    for k in range_:
        eigenvalues, eigenvectors = PCA(cloud_points[indices[k]])
        all_eigenvalues[k] = eigenvalues
        all_eigenvectors[k] = eigenvectors

    return all_eigenvalues, all_eigenvectors


def ProjMatrix(indices_ref, eigen_data):
    all_eigenvalues, all_eigenvectors = eigen_data[0][indices_ref], eigen_data[1][indices_ref]
    projMatrices = np.einsum('ij,ik->ijk', all_eigenvectors[:,0], all_eigenvectors[:,0])
    return projMatrices

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

def computeRotations(eigenvectors):
    epsilon = 1e-2
    cov = np.diag([epsilon, 1, 1])
    rotations = np.einsum('ijk,kli->ijl', np.einsum('ijk,kl->ijl', eigenvectors, cov), eigenvectors.T)
    return rotations

def computeRotationFromVectors(normals):
    e1 = np.array([1, 0, 0])
    v = np.apply_along_axis(lambda x: np.cross(e1, x), axis=1, arr = normals)
    #s = np.linalg.norm(v, axis = 1)
    c = normals[:,0]
    sk = skew(v)
    R = np.eye(3) + sk + np.einsum('ijk,ikl->ijl', sk, sk)/(1+c[:,None,None])
    return R

def skew(normals):
    sk = np.zeros((normals.shape[0], normals.shape[1], normals.shape[1]))
    sk[:,0,1], sk[:,0,2], sk[:,1,2] = - normals[:,2], normals[:,1], - normals[:,0]
    sk = sk - sk.transpose(0,2,1)
    return sk

def computeCovMat(cloud, eps = 1e-3):
    n = cloud.shape[0]
    tree = KDTree(cloud)
    dist, indices = tree.query(cloud, k = 10)
    covMat, all_eigenvectors = np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    for k in tqdm(range(n), desc = 'Covariance'):
        _, eigenvectors = PCA(cloud[indices[k]])
        all_eigenvectors[k] = eigenvectors
    R = computeRotationFromVectors(all_eigenvectors[:,0])
    I_eps = np.diag([eps, 1, 1])
    Cov = np.einsum('ijk,kli->ijl', np.einsum('ijk,kl->ijl', R, I_eps), R.T)
    return Cov

def computeCovTest(cloud):
    d = 3
    new_n = cloud.shape[0]
    cov_mat = np.zeros((new_n,d,d))

    tree = KDTree(cloud)
    dist, indices = tree.query(cloud, k = 10)
    all_eigenvectors = np.zeros((new_n, 3, 3))
    for k in tqdm(range(new_n), desc = 'Covariance'):
        eigenvalues, eigenvectors = PCA(cloud[indices[k]])
        all_eigenvectors[k] = eigenvectors

    dz_cov_mat = np.eye(d)
    dz_cov_mat[0,0] = 1e-2
    for i in range(new_n):
        U = all_eigenvectors[i]
        cov_mat[i,:,:] = U @ dz_cov_mat @ U.T

    return cov_mat

def show_ICP(data, ref, R_list, T_list, neighbors_list):
    '''
    Show a succession of transformation obtained by ICP.
    Inputs :
                  data = (d x N_data) matrix where "N_data" is the number of point and "d" the dimension
                   ref = (d x N_ref) matrix where "N_ref" is the number of point and "d" the dimension
                R_list = list of the (d x d) rotation matrices found at each iteration
                T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = list of the neighbors of data in ref at each iteration

    This function works if R_i and T_i represent the tranformation of the original cloud at iteration i, such
    that data_(i) = R_i * data + T_i.
    If you saved incremental transformations such that data_(i) = R_i * data_(i-1) + T_i, you will need to
    modify your R_list and T_list in your ICP implementation.
    '''

    # Get the number of iteration
    max_iter = len(R_list)

    # Get data dimension
    dim = data.shape[0]

    # Insert identity as first transformation
    R_list.insert(0, np.eye(dim))
    T_list.insert(0, np.zeros((dim, 1)))

    # Create global variable for the graph plot
    global iteration, show_neighbors
    iteration = 0
    show_neighbors = 0

    # Define the function drawing the points
    def draw_event():
        data_aligned = R_list[iteration].dot(data) + T_list[iteration]
        plt.cla()
        if dim == 2:
            ax.plot(ref[0], ref[1], '.')
            ax.plot(data_aligned[0], data_aligned[1], '.')
            if show_neighbors and iteration < max_iter:
                lines = [[data_aligned[:, ind1], ref[:, ind2]] for ind1, ind2 in enumerate(neighbors_list[iteration])]
                lc = mc.LineCollection(lines, colors=[0, 1, 0, 0.5], linewidths=1)
                ax.add_collection(lc)
            plt.axis('equal')
        if dim == 3:
            ax.plot(ref[0], ref[1], ref[2], '.')
            ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
            plt.axis('equal')
        if show_neighbors and iteration < max_iter:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors ON ===> Press n to change (only in 2D)'.format(iteration))
        else:
            ax.set_title('Iteration {:d} ===> press right / left to change\nNeighbors OFF ===> Press n to change (only in 2D)'.format(iteration))

        plt.draw()

    # Define the function getting keyborad inputs
    def press(event):
        global iteration, show_neighbors
        if event.key == 'right':
            if iteration < max_iter:
                iteration += 1
        if event.key == 'left':
            if iteration > 0:
                iteration -= 1
        if event.key == 'n':
            if dim < 3:
                show_neighbors = 1 - show_neighbors
        draw_event()

    # Create figure
    fig = plt.figure()

    # Intitiate graph for 3D data
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        print('wrong data dimension')

    # Connect keyboard function to the figure
    fig.canvas.mpl_connect('key_press_event', press)

    # Plot first iteration
    draw_event()

    # Start figure
    plt.show()

def grad_loss(x,a,b,M):
    """
    Gradient of the loss loss for parameter x
    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)
    returns:
        Value of the gradient of the loss function
    """
    t = x[3:]
    R = RotMatrix(x[:3])
    g = np.zeros(6)
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d

    g[3:] = - 2*np.sum(tmp, axis = 0)

    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = computeGradRot(x[:3]) # shape 3*d*d
    g[:3] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule
    return g

def grad(x,a,b,cov_a,cov_b):
    """
    Gradient of the loss loss for parameter x
    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)
    returns:
        Value of the gradient of the loss function
    """
    t = x[3:]
    R = RotMatrix(x[:3])
    M = np.linalg.inv(cov_b + np.einsum('ik,jkl,lm->jim', R, cov_a, R.T))

    g = np.zeros(6)
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.einsum('ijk,ik->ij', M, residual) # shape n*d

    g[3:] = - 2*np.sum(tmp, axis = 0)
    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = computeGradRot(x[:3]) # shape 3*d*d
    g[:3] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule
    return g

