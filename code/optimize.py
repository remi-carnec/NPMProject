# Imports
import numpy as np
from sklearn.neighbors import KDTree
from utils import compute_eigen_data

# Find best transformation
def optimize(data, ref, algo, kNeighbors = 20, max_iter = 50, dist_threshold = 0.1, RMS_threshold = 1e-6, eps = 1e-3, verbose = True):
    # Initiate lists
    RMS_list = []

    # Precalculate
    args = None
    if algo.name == "point-to-plane" or algo.name == "plane-to-plane":
        print("---------------------")
        print("Precalculating data:")
        tree, args = compute_eigen_data(data.T, ref.T, algo, k = kNeighbors, eps = eps)
        print("---------------------")
    else:
        tree = KDTree(ref.T)

    # Initialize iterating variables
    rms = RMS_threshold
    rms_min = np.inf
    data_aligned = data.copy()
    iter = 0
    x = np.zeros(6)

    # Iterate until convergence
    while iter < max_iter and rms >= RMS_threshold:
        # Find neighbors
        dist, neighbors = tree.query(data_aligned.T, k = 1)
        account = np.concatenate(dist < dist_threshold)
        indices_ref = neighbors.flatten()[account]
        indices_data = np.arange(data.shape[1])[account]

        # Compute and apply transformation
        R, T, x = algo.findBestTransform(data_aligned, ref, indices_data, indices_ref, args, x)
        data_aligned = R @ data_aligned + T

        # Update loss
        rms = np.sqrt(np.mean(np.sum(np.power(data_aligned - ref[:,neighbors.flatten()], 2), axis=0)))
        if verbose:
            print("rms = {}".format(rms))

        # Store transformation
        RMS_list.append(rms)
        iter += 1
        if rms < rms_min:
            R_min, T_min = R.copy(), T.copy()
            rms_min = rms
    if verbose:
        print("Needed {} iterations".format(iter))

    return data_aligned, RMS_list, R_min, T_min

