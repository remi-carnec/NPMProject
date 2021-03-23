# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from tqdm import tqdm
from utils import PCA



def optimize(data, ref, algo, max_iter, RMS_threshold):
    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    dataCloud = None
    if algo.name == "point2plane" or algo.name == "plane2plane":
        tree = KDTree(ref.T)
        dist, indices = tree.query(ref.T, k = 10)
        all_eigenvalues = np.zeros((ref.shape[1], 3))
        all_eigenvectors = np.zeros((ref.shape[1], 3, 3))
        range_ = tqdm(range(ref.shape[1]))
        for k in range_:
            eigenvalues, eigenvectors = PCA(ref[:,indices[k]].T)
            all_eigenvalues[k] = eigenvalues
            all_eigenvectors[k] = eigenvectors
        wholeDataCloud = (all_eigenvalues, all_eigenvectors)


    # Initialize
    rms = RMS_threshold
    dist_threshold = 0.1
    tree = KDTree(ref.T, leaf_size=8)
    data_aligned = data.copy()
    i = 0

    # Iterate until convergence
    while i < max_iter and rms < RMS_threshold:

        # Find neighbors
        dst, indices = tree.query(data_aligned.T, k = 1)
        account = np.concatenate(dst < dist_threshold)
        indices = indices.reshape(indices.shape[0])
        print(np.mean(account))
        # Compute and apply transformation
        if algo.name == "point2plane" or algo.name == "plane2plane":
            dataCloud = (wholeDataCloud[0][indices[account]], wholeDataCloud[1][indices[account]])
        R, T = algo.findBestTransform(data_aligned[:,account], ref[:,indices[account]], dataCloud)
        data_aligned = R @ data_aligned + T
        rms = np.sqrt(np.mean(np.sum(np.power(data_aligned - ref[:,indices], 2), axis=0)))
        print("rms = {}".format(rms))

        # Store transformation
        if len(R_list) == 0:
            R_list.append(R.copy())
            T_list.append(T.copy())
        else:
            R_list.append(R @ R_list[-1])
            T_list.append(R@T_list[-1] + T)
        neighbors_list.append(indices.copy())
        RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list

