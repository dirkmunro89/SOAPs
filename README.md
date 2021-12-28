# SOAPs
**S**equential **O**ptimization **A**pproximate sub**P**roblem**s**

THIS REPO IS UNDER CONSTRUCTION; THE CODE IS SUBJECT TO TESTING AND CLEANING UP

## Excecution

Run with

`python3 main.py | tee history.txt`

Scipy and Numpy packages required. Tested on Ubuntu 20.04 LTS.

## Problem setup

The optimization problem to be solved is defined in `prob.py`:

- in the function `init()` the problem size is set in terms of number of variables (`n`) and the number of constraints (`m`). A starting point `x_i` and global bounds on the design varialbes (`x_l` and `x_u`) is specified. Beyond this, various subproblem formulations and algorithmic parameters is set&mdash;see documentation for a description.

- in the function `simu()` the computations which yield function values `g` (objective at index 0, followed by constraints), for a given design variable array `x_p`, is set. (And of course, external analysis packages may easily be called.) If available, the computations which yield the first-order derivatives `dg` of each function to each variable is set as well&mdash;the derivative of function index `j` to variable `i` sits in row `j` and column `i`, of `dg`. Alternatively, a finite difference scheme may be activated in `init()`, at the expense of computational resources.


## Description

This repository contains a simple procedural representation of a general optimization (nonlinear programming) algorithm. In particular, it represents the established form of nonlinear programming algorithm found in the field of structural optimization, in development since at least the 1970´s [[1]](#1). It is however not incorrect to draw the equivalence with Newton's method (or the Newton-Raphson method). Apparently, as early as 1740 Thomas Simpson (of numerical integration fame) described Newton's method as an iterative method for solving systems of nonlinear equations, noting that it can be applied to optimization problems by solving for the stationary point [[2]](#2). Today, the form of nonlinear programming algorithm found in structural optimization is referred to *Sequential Approximate Optimization (SAO)*, and it is typically characterized by:

- *positive* (and bounded) *design variables*, representing physical quantities such as structural member thickness, cross-sectional area, or material density;

- *inequality constraints*, representing geometric restrictions, a restriction on material usage, or the maximum displacement and/or stress permitted in the structure;

- repeated evaluation of computionally expensive *structural analysis routines* (often external FE packages); *simulation-based*, in general, insofar as *simulation* refers to the modelling of a physical system via solution of partial differential equations (PDE);

- utilization of *intervening variables (variable transformations) and function approximation concepts* to arrive at computationally tractable yet reasonably convergent approximate subproblems;

- and *dual formulations of the subproblems* (e.g. Falk), permitting simple numerical solution by readily available bounded minimization routines (e.g. L-BFGS-B). (This is true in a traditional sense, at least; primal-dual solution methods are increasingly available, at the expense of the elegance and simplicity of the classical dual formulations.)

## Keywords (in work)

Sequential Approximate Optimization (SAO)
Method of Moving Asymptotes (MMA)
CONLIN
Sequenital Approximate Optimization based on Incomplete Taylor Series expansions (SAOi)
Quadratically Constrained Quadratic Programming
Quadratic Programming (linear constraints)
Dual method (Falk)
Bayesian Optimization Global

## References
<a id="1">[1]</a> 
Fleury, C. (1979).
Structural weight optimization by dual methods of convex programming.
International Journal for Numerical Methods in Engineering, 14(12):1761–1783.

<a id="2">[2]</a> 
Wikipedia (2002).
[Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method).
International Journal for Numerical Methods in Engineering, 14(12):1761–1783.

