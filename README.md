# SOAPs
**S**equential **O**ptimization **A**pproximate sub**P**roblem**s**

THIS REPO IS UNDER CONSTRUCTION; THE CODE IS SUBJECT TO TESTING AND CLEANING UP

## Execution

Numpy and Scipy packages required. 

Tested on Ubuntu 20.04 LTS with

`python3 main.py | tee history.txt`

and on Raspbian GNU/Linux 11 with

`python main.py | tee history.txt`

(For large or long running problems the *unbuffer* command 

`unbuffer python main.py | tee history.txt`

as provided by the Expect package, may be convenient.)

## Problem setup

The optimization problem to be solved is defined in `prob.py`:

- in the function `init()` the problem size is set in terms of number of variables (`n`) and the number of constraints (`m`). A starting point `x_i` and global bounds on the design variables (`x_l` and `x_u`) is specified. Beyond this, various subproblem formulations and algorithmic parameters is set&mdash;see documentation for a details.

- in the function `simu()` the computations which yield function values `g` (objective at index 0, followed by constraints), for a given design variable array `x_p`, is set. (And of course, external analysis packages may easily be called.) If available, the computations which yield the first-order derivatives `dg` of each function to each variable is also set&mdash;the derivative of function index `j` to variable `i` sits in row `j` and column `i`, of the 2D array `dg`. Alternatively, a finite difference scheme may be activated in `init()`, at the expense of computational resources.

## Description

This repository contains a simple procedural representation of a general optimization (nonlinear programming) algorithm. In particular, it represents the established form of nonlinear programming algorithm found in the field of structural optimization, in development since at least the 1970´s [[1]](#1). However, it is not incorrect to draw the equivalence with Newton's method (or the Newton-Raphson method); apparently, as early as 1740, Thomas Simpson (of numerical integration fame) described Newton's method as an iterative method for solving systems of nonlinear equations, noting that it can be applied to optimization problems by solving for the stationary point [[2]](#2). Today, the form of nonlinear programming algorithm found in structural optimization is referred to *Sequential Approximate Optimization (SAO)*, and it is typically characterized by:

- *positive* (and bounded) *design variables*, representing physical quantities such as *e.g.* structural member thickness, cross-sectional area, or material density;

- *inequality constraints*, representing *e.g.* design restrictions, a restriction on overall material usage (a material distribution problem), or such quantities as the maximum displacement, or stresses permitted in the structure;

- repeated evaluation of computationally expensive *structural analysis* routines (often external finite element modeling packages); that is *simulation-based*, in general, insofar as *simulation* refers to the modeling of a physical system via numerical solution of a set of partial differential equations (PDEs);

- utilization of *intervening variables (variable transformations) and function approximation concepts* to arrive at computationally tractable yet reasonably convergent approximate subproblems;

- and *dual subproblem formulations* (e.g. Falk [[3]](#3)), permitting simple numerical solution by readily available bounded function minimization routines (e.g. L-BFGS-B). (This is true in a traditional sense, at least. Primal-dual (*e.g.* interior-point) methods are increasingly available, at the expense of the elegance and the simplicity of the classic dual formulations.)

It should however be noted that the nonlinear programming paradigm represented here, is not in general restricted to: *(i)* inequality constraints [[4]](#4), *(ii)* positive (and natural) design variables (see *e.g.* simultaneous analysis and design (SAND) [[5]](#5) and Space-Time [[6]](#6) formulations), and *(iii)* as already mentioned, dual subproblem formulations [[7]](#7).

## Keywords (in work)

Sequential Approximate Optimization (SAO)
Method of Moving Asymptotes (MMA)
CONLIN
Sequential Approximate Optimization based on Incomplete Taylor Series expansions (SAOi)
Quadratically Constrained Quadratic Programming
Quadratic Programming (linear constraints)
Dual method (Falk)
Bayesian Optimization Global

## References
<a id="1">[1]</a>
Fleury, C. (1979)
Structural weight optimization by dual methods of convex programming.
International Journal for Numerical Methods in Engineering, 14(12):1761–1783.

<a id="2">[2]</a>
Wikipedia (2002).
[Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method).

<a id="3">[3]</a>
Falk, J.E. (1967)
Lagrange multipliers and nonlinear programming.
Journal of Mathematical Analysis and Applications, 19(1):141–159.

<a id="4">[4]</a>
Cronje, M. *et al.* (2019)
Brief note on equality constraints in pure dual SAO settings.
Structural and Multidisciplinary Optimization, 59(5):1853–1861.

<a id="5">[5]</a>
Haftka, R.T. (1985) 
Simultaneous analysis and design. 
American Institute of Aeronautics and Astronautics Journal, 23(7):1099–1103.

<a id="6">[6]</a>
Wu, J. (2020) 
[Space-time topology optimization for additive manufacturing](https://doi.org/10.1007/s00158-019-02420-6). 
Structural and Multidisciplinary Optimization, 61:1-18.

<a id="7">[7]</a>
Zillober, C. (2001) 
A combined convex approximation&mdash;interior point approach for large scale
nonlinear programming. 
Optimization and Engineering, 2(1):51–73.
