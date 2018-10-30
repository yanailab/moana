# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Miscellaneous functions."""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from ..core import ExpMatrix

def add_dither(matrix: ExpMatrix, dither: float = 0.01, seed: int = 0,
               inplace: bool = False):
    np.random.seed(seed)
    X = matrix.values
    if not inplace:
        X = X.copy()
    d, n = X.shape
    if dither > 0:
        for l in range(d):
            ptp = np.ptp(X[l, :])
            dx = (np.random.rand(n)-0.5)*ptp*dither
            X[l, :] = X[l, :] + dx
    result = ExpMatrix(genes=matrix.genes, cells=matrix.cells, X=X)
    return result
