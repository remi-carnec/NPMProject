# Imports
import numpy as np
from sklearn.neighbors import KDTree
from utils import compute_eigen_data

# Find best transformation
def optimize(data, ref, algo, max_iter, RMS_threshold):
    # Initiate lists
    RMS_list = []

    # Precalculate
    args = None
    if algo.name == "point2plane" or algo.name == "plane2plane":
        print("Precalculating data:")
        tree, args = compute_eigen_data(data.T, ref.T, algo, k = 20)
        print("Done")
    else:
        tree = KDTree(ref.T)

    # Initialize
    rms = RMS_threshold
    dist_threshold = 0.05
    data_aligned = data.copy()
    n = data.shape[1]
    iter = 0

    # Iterate until convergence
    while iter < max_iter and rms >= RMS_threshold:
        # Find neighbors
        dist, neighbors = tree.query(data_aligned.T, k = 1)
        account = np.concatenate(dist < dist_threshold)
        indices_ref = neighbors.flatten()[account]
        indices_data = np.arange(n)[account]

        # Compute and apply transformation
        R, T = algo.findBestTransform(data_aligned, ref, indices_data, indices_ref, args)
        data_aligned = R @ data_aligned + T

        # Update loss
        rms = np.sqrt(np.mean(np.sum(np.power(data_aligned - ref[:,indices_ref], 2), axis=0)))
        print("rms = {}".format(rms))

        # Store transformation
        RMS_list.append(rms)

    return data_aligned, RMS_list

