# Finite Volume for Conservation Laws
This repository contains several finite volume methods for solving a hyperbolic conservation law of the form

$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0.$$

The package inludes several different numerical fluxes and some specific equations. Some examples are given in the scripts/ folder.
Currently, only scalar equations in one space dimension are supported.

## Author
The code is currently maintained by Joshua Lampert (joshua.lampert@tuhh.de) and work in progress.

## Requirements
* [numpy](https://github.com/numpy/numpy)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [scipy](https://github.com/scipy/scipy)
* [sympy](https://github.com/sympy/sympy)
