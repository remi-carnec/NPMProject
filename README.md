# NPMProject
Course Project | NPM3D | MVA Master (2020/2021)

## Recap
This is my final project of the course NPM3D that focused on the *Generalized-ICP* method. You will find in this repository my report summarizing the different experiments I ran on this method, as well as my implementation of *Standard ICP*, *Point-to-plane* and *Generalized-ICP* in Python.

## Code
A starting code is provided in ```main_py```, where the user can play around with the different methods on two datasets that you will find in the `data` folder: the *Stanford Bunny* and a lighter version of the *Asian Dragon* that I downloaded from the Standord 3D Scanning Repository (https://graphics.stanford.edu/data/3Dscanrep/). The implementation relies on three main files:
- ```optimize.py```: contains the main structure of the optimization algorithm (see *Generalized-ICP*, Segal et al.).
- ```algorithms.py```: used to minimize the loss at each iteration for the different methods (using a closed form solution for *Standard ICP* and the Conjugate Gradient for *Point-to-plane* and *Generalized-ICP*).
- ```utils.py```: contains all sorts of useful functions to perform PCA, compute rotation matrices, gradients, etc.
