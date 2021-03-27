# Imports
from ply import write_ply, read_ply
import numpy as np
from time import time
from matplotlib import pyplot as plt
from algorithms import Algorithm
from optimize import optimize

if __name__ == '__main__':

    # Transformation estimation
    # *************************
    #
    # Cloud paths
    bunny_o_path = '../data/bunny_original.ply'
    bunny_p_path = '../data/bunny_perturbed.ply'
    #bunny_o_path = '../model_bunny.ply'
    #bunny_p_path = '../transformed_point_cloud.ply'

    bunny_o_path = '../bunny_half.ply'
    bunny_p_path = '../bunny_new_perturbed.ply'

    # Load clouds
    bunny_o_ply = read_ply(bunny_o_path)
    bunny_p_ply = read_ply(bunny_p_path)
    bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
    bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

    #bunny_o = bunny_o[:,np.random.choice(bunny_o.shape[1], size = int(bunny_o.shape[1]/2))]
    #bunny_p = bunny_p[:,np.random.choice(bunny_p.shape[1], size = int(bunny_p.shape[1]/2))]
    #write_ply('../bunny_o_sub', [bunny_o.T], ['x', 'y', 'z'])
    #write_ply('../bunny_p_sub', [bunny_p.T], ['x', 'y', 'z'])
    # Random
    #thetas = 0.5 * np.random.randn(3)
    #t = np.random.randn(3)*0.03
    #R = RotMatrix(thetas)
    #new_perturbed = t[:,None] + R @ bunny_p
    #write_ply('../bunny_new_perturbed', [new_perturbed.T], ['x', 'y', 'z'])
    #bunny_p = new_perturbed

    # Apply ICP
    config_opti = {'max_iter': 100, 'dist_threshold': 0.05, 'RMS_threshold': 1e-5, 'kNeighbors': 20}
    comparison = False
    if comparison:
        plt.title("Convergence of the different methods")
        algos = [Algorithm('plane-to-plane'), Algorithm('point-to-plane'), Algorithm('point-to-point')]
        for algo in algos:
            start = time()
            bunny_p_opt, RMS_list = optimize(bunny_p, bunny_o, algo, config_opti)
            print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
            plt.plot(RMS_list, label = algo.name)
            #if algo.name == 'plane-to-plane':
            #    write_ply('../bunny_new_perturbed_solution', [bunny_p_opt.T], ['x', 'y', 'z'])
    else:
        #algo = Algorithm(name = 'plane-to-plane', config_algo = {'relaxed_gradient': False, 'max_iter': 10})
        #start = time()
        #bunny_p_opt, RMS_list, R, T = optimize(bunny_p, bunny_o, algo, config_opti)
        #print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
        #plt.plot(RMS_list, label = algo.name + ", exact gradient")
        #print("R = {}".format(R))
        #print("T = {}".format(T))
        algo = Algorithm(name = 'plane-to-plane', config_algo = {'relaxed_gradient': True, 'max_iter': 10})
        start = time()
        bunny_p_opt, RMS_list, R, T = optimize(bunny_p, bunny_o, algo, config_opti)
        print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
        print("R = {}".format(R))
        print("T = {}".format(T))
        plt.plot(RMS_list, label = "relaxed gradient, " + str(algo.config_algo['max_iter']) + " iterations")
        algo = Algorithm(name = 'plane-to-plane', config_algo = {'relaxed_gradient': True, 'max_iter': 20})
        start = time()
        bunny_p_opt, RMS_list, R, T = optimize(bunny_p, bunny_o, algo, config_opti)
        print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
        print("R = {}".format(R))
        print("T = {}".format(T))
        plt.plot(RMS_list, label = "relaxed gradient, " + str(algo.config_algo['max_iter']) + " iterations")
        algo = Algorithm(name = 'plane-to-plane', config_algo = {'relaxed_gradient': True, 'max_iter': 30})
        start = time()
        bunny_p_opt, RMS_list, R, T = optimize(bunny_p, bunny_o, algo, config_opti)
        print("Optimization for " + algo.name + " lasted: {}s".format(round(time()-start,2)))
        print("R = {}".format(R))
        print("T = {}".format(T))
        plt.plot(RMS_list, label = "relaxed gradient, " + str(algo.config_algo['max_iter']) + " iterations")
        plt.title("Influence of relaxation")
    # Show ICP
    #show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

    # Save cloud
    #write_ply('../bunny_r_opt', [bunny_p_opt.T], ['x', 'y', 'z'])

    # Plot RMS
    #plt.plot(RMS_list)
    #plt.title(algo.name)
    plt.xlabel('iterations')
    plt.ylabel('RMS')
    plt.legend()
    #plt.savefig('../report/img/comparison_relaxed_1.png')
    plt.show()

if False:
    thetas = np.random.randn(3)
    t = np.random.randn(3)*0.05
    R = RotMatrix(thetas)
    new_perturbed = t[:,None] + R @ bunny_o
    write_ply('../bunny_new_perturbed', [new_perturbed.T], ['x', 'y', 'z'])
    bunny_p = new_perturbed

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    maxiter = [3, 5,10,15,20,30,50]
    iterations = [40, 14,8,7,5,6,5]
    time = [90.2, 54.47,53.86,63.8,59.8,97.82,142.45]
    ax.plot(maxiter, iterations, color = 'red', marker = 'o')
    ax.set_ylabel('Overall iterations', color = 'red')
    ax.set_xlabel('Maximum optimizer iterations')
    ax2 = ax.twinx()
    ax2.plot(maxiter, time, color = 'blue', marker = 'o')
    ax2.set_ylabel('Total convergence time (s)', color = 'blue')
    plt.savefig('maxiter.png')
    plt.show()

    m = np.mean(bunny_o, axis = 1)
    shift = -0.027 * np.ones(3)
    m += shift
    v = np.ones(3)
    account = v @ (bunny_o - m[:,None]) > 0
    write_ply('../bunny_half', [bunny_o[:,account].T], ['x', 'y', 'z'])

    m = np.mean(bunny_o, axis = 1)
    shift = -0.0 * np.ones(3)
    m += shift
    v = np.ones(3)
    account = v @ (bunny_o - m[:,None]) < 0
    write_ply('../bunny_half_bis', [bunny_o[:,account].T], ['x', 'y', 'z'])
