photogrammetry-resection
========================

A simple single-photo resection program written in python.

Solution technique uses a performance function based on the collinearity equation that is minimized with scipy.optimize.minimize().

Two input files are needed: camera calibration and control points. Sample data is from: 
    Introduction to Modern Photogrammetry by Mikhail, Bethel, McGlone
    John Wiley & Sons, Inc. 2001

Usage: python resection.py cam.inp resect.inp

The Jupyter notebook compares three different solvers: scipy.minimize(), scipy.leastsq() and scipy.leastsq() with first derivative.
