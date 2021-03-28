# Imports
from ply import write_ply, read_ply
import numpy as np
from time import time
from matplotlib import pyplot as plt
from algorithms import Algorithm
from optimize import optimize
from utils import *
#from plyfile import PlyData, PlyElement

if __name__ == '__main__':
    # Transformation estimation
    # *************************
    #
    # Cloud paths
    bunny_o_path = '../data/bunny_original.ply'
    bunny_p_path = '../data/bunny_perturbed.ply'
    dragon_o_path = '../data/dragon_original.ply'
    dragon_p_path = '../data/dragon_perturbed.ply'

    # Load clouds
    UseBunny = True
    if UseBunny:
        # Load Bunny point cloud
        cloud_o_ply = read_ply(bunny_o_path)
        cloud_p_ply = read_ply(bunny_p_path)
    else:
        # Load Dragon point cloud
        cloud_o_ply = read_ply(dragon_o_path)
        cloud_p_ply = read_ply(dragon_p_path)
    cloud_o = np.vstack((cloud_o_ply['x'], cloud_o_ply['y'], cloud_o_ply['z']))
    cloud_p = np.vstack((cloud_p_ply['x'], cloud_p_ply['y'], cloud_p_ply['z']))

    # Random transformation
    apply_random_transfo = False
    if apply_random_transfo:
        np.random.seed(42)
        t = np.random.randn(3)*0.05
        thetas = np.pi * np.random.rand(3)
        R = RotMatrix(thetas)
        cloud_p = t[:,None] + R @ cloud_o
        write_ply('../data/test_perturbed', [cloud_p.T], ['x', 'y', 'z'])

    # Optimization parameters
    max_iter = 50 # Maximum iterations in main loop
    dist_threshold = 0.1 # d_max
    RMS_threshold = 1e-5 # Convergence threshold
    kNeighbors = 20 # To compute PCA
    eps = 1e-3 # Covariance matrix parameter (for Generalized-ICP only)

    # Transformation estimation
    Comparison = 0
    if Comparison == 0:
        # Run and compare all methods
        plt.title("Convergence of the different methods")
        algos = [Algorithm('plane-to-plane'), Algorithm('point-to-plane'), Algorithm('point-to-point')]
        for algo in algos:
            start = time()
            cloud_p_opt, RMS_list, R, T = optimize(cloud_p, cloud_o, algo, kNeighbors, max_iter, dist_threshold, RMS_threshold, eps)
            print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
            plt.plot(RMS_list, label = algo.name)
    elif Comparison == 1:
        # Compare Generalized-ICP with exact and relaxed gradients
        # Exact Gradient
        start = time()
        algo = Algorithm(name = 'plane-to-plane', relaxed_gradient = False, max_iter = 10)
        cloud_p_opt, RMS_list, R, T = optimize(cloud_p, cloud_o, algo, kNeighbors, max_iter, dist_threshold, RMS_threshold, eps)
        print("Optimization for exact gradient lasted: {}s".format(round(time()-start,2)))
        plt.plot(RMS_list, label = algo.name + ", exact gradient")

        # Relaxed Gradient
        start = time()
        algo = Algorithm(name = 'plane-to-plane', relaxed_gradient = True, max_iter = 10)
        cloud_p_opt, RMS_list, R, T = optimize(cloud_p, cloud_o, algo, kNeighbors, max_iter, dist_threshold, RMS_threshold, eps)
        print("Optimization for relaxed gradient lasted: {}s".format(round(time()-start,2)))
        plt.plot(RMS_list, label = algo.name + ", relaxed gradient")
        plt.title("Influence of Exact/Relaxed gradient on convergence")
    elif Comparison == 2:
        # Run Generalized-ICP
        start = time()
        relaxed_gradient = False
        algo = Algorithm(name = 'plane-to-plane', relaxed_gradient = relaxed_gradient, max_iter = 10)
        cloud_p_opt, RMS_list, R, T = optimize(cloud_p, cloud_o, algo, kNeighbors, max_iter, dist_threshold, RMS_threshold, eps)
        print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
        plt.plot(RMS_list, label = algo.name + ", " + ("relaxed" if relaxed_gradient else "exact") + " gradient")
        plt.title("Convergence of " + algo.name)
    else:
        # Not implemented
        pass

    # Plot RMS
    plt.xlabel('iterations')
    plt.ylabel('RMS')
    plt.legend()
    plt.show()
