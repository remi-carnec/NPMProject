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


def ProjMatrix(data, ref, dataCloud, radius, k = None):
    if dataCloud == None:
        all_eigenvalues, all_eigenvectors = compute_local_PCA(data, ref, dataCloud, radius, k)
    else:
        all_eigenvalues, all_eigenvectors = dataCloud
    projMatrices = np.einsum('ij,ik->ijk', all_eigenvectors[:,0], all_eigenvectors[:,0])
    #res = np.einsum('ijk,ijl->ijkl', all_eigenvectors, all_eigenvectors)
    return projMatrices

def RotMatrix(thetas):
    R_x = np.array([[1,0,0], [0, np.cos(thetas[0]), -np.sin(thetas[0])], [0, np.sin(thetas[0]), np.cos(thetas[0])]])
    R_y = np.array([[np.cos(thetas[1]),0,np.sin(thetas[1])], [0, 1, 0], [-np.sin(thetas[1]),0,np.cos(thetas[1])]])
    R_z = np.array([[np.cos(thetas[2]),-np.sin(thetas[2]), 0], [np.sin(thetas[2]),np.cos(thetas[2]), 0], [0, 0, 1]])
    #return R.from_euler('zyx', thetas)
    return R_z @ R_y @ R_x

def computeRotations(eigenvectors):
    epsilon = 1e-2
    cov = np.diag([epsilon, 1, 1])
    rotations = rotations = np.einsum('ijk,kli->ijl', np.einsum('ijk,kl->ijl', eigenvectors, cov), eigenvectors.T)
    return rotations

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
