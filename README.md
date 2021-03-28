# NPMProject
Course Project | NPM3D | MVA Master (2020/2021)

## Recap
This is my final project of the course NPM3D that focused on the *Generalized-ICP* method. You will find in this repository my report on the different experiments I ran on this method, as well as my implementation of *Standard ICP*, *Point-to-plane* and *Generalized-ICP*.

## Code
A starting code is provided in ```main_py```, where the user can play around with the different methods on two datasets: the *Stanford Bunny* and the *Asian Dragon* (https://graphics.stanford.edu/data/3Dscanrep/). The implementation relies on three main files:
- ```optimize.py```: contains the main structure of the optimization algorithm (see *Generalized-ICP*, Segal et al.)
- ```algorithms.py```: used to minimize the loss for each different method (Closed form solution for *Standard ICP*, Conjugate Gradient for *Point-to-plane* and *Generalized-ICP*.)
- ```utils.py```: contains all sorts of useful functions to perform PCA, compute rotation matrices, gradients, etc.
