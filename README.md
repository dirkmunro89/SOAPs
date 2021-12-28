# SOAPs
Sequential Optimization Approximate subProblems

This repository contains a simple, procedural representation of a general optimization (nonlinear programming) algorithm. In particular, it represents the established form of nonlinear programming algorithm found in the field of structural optimization, in development since at least the 1970´s. It is however not incorrect to draw the equivalence with Newton's Method (or the Newton-Raphson Method). In fact, as early as 1740 Thomas Simpson (of numerical integration fame) described Newton's Method as an iterative method for solving systems of nonlinear equations, noting that it can be used to solve optimization problems by letting the system represent the stationary point of the problem. Today, the form of nonlinear programming algorithm found in structural optimization is referred to *Sequential Approximate Optimization (SAO)*, and it is typically characterized by:
- positive (and bounded) design variables, representing physical quantities such as structural member thickness, cross-sectional area, or material density;
- inequality constraints, representing restrictions on dimensions, the amount of material, or the maximum displacements and/or stresses permitted in the structural design;
- utilization of intervening variables (variable transformations) and approximation concepts to arrive at computationally tractable yet reasonably convergent approximate subproblems;
- and dual formulations (e.g. Falk) of the subproblems, permitting simple numerical solution by readily available bounded minimization routines (e.g. L-BFGS-B). (This is true in a traditional sense, at least; primal-dual solution methods are increasingly available, at the expense of the elegance and simplicity of the classic dual formulations.)

THIS REPO IS UNDER CONSTRUCTION; THE CODE IS SUBJECT TO TESTING AND CLEANING UP

Run with 

`python3 main.py | tee history.txt`

Tested on Ubuntu 20 with Scipy and Numpy packages installed.


Sequential Approximate Optimization (SAO)
Method of Moving Asymptotes (MMA)
CONLIN
Sequenital Approximate Optimization based on Incomplete Taylor Series expansions (SAOi)
Quadratically Constrained Quadratic Programming
Quadratic Programming (linear constraints)
Dual method (Falk)
Bayesian Global Optimization

Test problems

asdasdasd



References

"...the **go to** statement should be abolished..." [[1]](#1).

## References
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.
