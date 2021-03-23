# Import functions to read and write ply files
from ply import write_ply, read_ply
from utils import show_ICP
import sys
import numpy as np

# Import library to plot in python
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

    # Load clouds
    bunny_o_ply = read_ply(bunny_o_path)
    bunny_p_ply = read_ply(bunny_p_path)
    bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
    bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

    # Apply ICP
    algo = Algorithm('plane2plane')
    bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = optimize(bunny_p, bunny_o, algo, 40, 1e-4)

    # Show ICP
    #show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

    # Save cloud
    write_ply('../bunny_r_opt', [bunny_p_opt.T], ['x', 'y', 'z'])

    # Plot RMS
    plt.plot(RMS_list)
    plt.title(algo.name)
    plt.show()

