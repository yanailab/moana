# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions implementing k-nearest neighbor smoothing."""

import time
import sys
import logging
from math import log, ceil
from typing import Union

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from ..core import ExpMatrix
from . import median_normalize, freeman_tukey_transform

_LOGGER = logging.getLogger(__name__)

#_UINT32_MIN = np.iinfo(np.uint32).min
#_UINT32_MAX = np.iinfo(np.uint32).max


def _calculate_pc_scores(
        X: np.ndarray, num_components: int, seed: int = 0) -> np.ndarray:
    """Calculates the first d principal components and returns PC scores."""

    # median-normalize
    T = median_normalize(X)

    # Freeman-Tukey transform
    T = freeman_tukey_transform(T)

    # perform PCA
    pca_model = PCA(n_components=num_components,
                    svd_solver='randomized', random_state=seed)
    t0 = time.time()
    Y = pca_model.fit_transform(T.T).T
    t1 = time.time()
    var_explained = np.sum(pca_model.explained_variance_ratio_)
    _LOGGER.info('PCA took %.1f s.', t1-t0)
    _LOGGER.info('The fraction of variance explained by the top %d PCs is '
                 '%.1f %%.', num_components, 100*var_explained)
    sys.stdout.flush()

    return Y


def knn_smoothing(
        matrix: ExpMatrix, k: int, d: int = 10,
        dither: Union[float, int] = 0,
        seed: int = 0,
        k_init: int = 2, step_factor: int = 2,
        start_step: int = 1,
        smoothed_matrix: ExpMatrix = None) -> ExpMatrix:
    """Denoise scRNA-Seq data using k-nearest neighbor smoothing."""

    t0_total = time.time()

    np.random.seed(seed)

    if k < 1:
        raise ValueError('Invalid value for "k" (%s)! Must be >= 1.' % str(k))

    if k > matrix.n:
        raise ValueError('Invalid value for "k" (%s)! '
                         'Must be <= number of cells in the matrix (%d).'
                         % (str(k), matrix.n))

    if start_step < 1:
        _LOGGER.warning('Invalid value for start_step...will use '
                        'start_step=1.')
        start_step = 1

    if start_step > 1 and smoothed_matrix is None:
        raise ValueError('Need smoothed matrix for hot start!')

    num_transcripts = matrix.sum(axis=0)
    _LOGGER.info('The raw matrix has a median transcript count of %.1f.',
                 num_transcripts.median())

    # determine k values for smoothing steps
    k_values = []
    k_step = 1
    while k_step < k:
        if k_step == 1:
            k_step = min(k_init, k)
        else:
            k_step = min(k_step * step_factor, k)
        k_values.append(k_step)

    X = np.array(matrix.X, dtype=np.float64, order='F', copy=False)

    # initialize smoothed matrix
    if smoothed_matrix is None:
        S = X.copy()
    else:
        S = np.array(smoothed_matrix.X, dtype=np.float64, order='F', copy=True)

    # perform smoothing
    num_steps = len(k_values)
    for i in range(start_step-1, num_steps):
        k_step = k_values[i]
        _LOGGER.info('Step %d/%d: Smooth using k=%d',
                     i+1, num_steps, k_step)
        sys.stdout.flush()

        Y = _calculate_pc_scores(S, d, seed)

        if dither > 0:
            # add dither
            for l in range(Y.shape[0]):
                ptp = np.ptp(Y[l, :])
                dy = (np.random.rand(Y.shape[1])-0.5)*ptp*dither
                Y[l, :] = Y[l, :] + dy

        # determine cell-cell distances using PCA-transformed smoothed matrix
        t0 = time.time()
        D = pairwise_distances(Y.T, n_jobs=1, metric='euclidean')
        t1 = time.time()
        _LOGGER.info('Calculating pair-wise distance matrix took %.1f s.',
                     t1-t0)
        sys.stdout.flush()
        
        t0 = time.time()
        A = np.argsort(D, axis=1, kind='mergesort')
        for j in range(X.shape[1]):
            ind = A[j, :k_step]
            S[:, j] = np.sum(X[:, ind], axis=1)
        t1 = time.time()
        _LOGGER.info('Calculating the smoothed expression matrix took %.1f s.',
                      t1-t0)
        #_LOGGER.info('Smoothed matrix hash: %s', smoothed_matrix.hash)
        sys.stdout.flush()

    smoothed_matrix = ExpMatrix(genes=matrix.genes, cells=matrix.cells, X=S)
    t1_total = time.time()

    _LOGGER.info('Finished smoothing!')
    _LOGGER.info('Smoothing took %.1f s.', t1_total-t0_total)
    num_transcripts = smoothed_matrix.sum(axis=0)
    _LOGGER.info('The smoothed matrix has a median transcript count of %.1f.',
                 num_transcripts.median())
    sys.stdout.flush()

    return smoothed_matrix


# def knn_smoothing_neighbors(
#         matrix, k, d=10, dither=0.03, seed=0,
#         k_start=4, step_factor=2,
#         start_step=1, smoothed_matrix=None):
#     """Expects a transcript count matrix.
    
#     Also returns the neighbors for each cell."""
    
#     t0_total = time.time()

#     np.random.seed(seed)

#     if k < 1:
#         raise ValueError('Invalid value for k! Must be >= 1.')

#     if k == 1:
#         num_steps = 0
#     elif k < k_start:
#         num_steps = 1
#     else:
#         num_steps = 1 + int(ceil(log(k/k_start)/log(step_factor)))

#     if start_step < 1:
#         print('Invalid value for start_step...will use start_step=1.')
#         start_step = 1

#     if start_step > 1 and smoothed_matrix is None:
#         raise ValueError('Need smoothed matrix for hot start!')
    
#     k_step = k_start * pow(step_factor, start_step-1)
#     X = matrix.X
#     N = np.empty((matrix.shape[1], k), dtype=np.int32)
    
#     if smoothed_matrix is not None:
#         S = smoothed_matrix.X.copy()
#     else:
#         S = matrix.X.copy()
    
#     for t in range(start_step, num_steps+1):
#         k_step = min(k_step, k) 
#         print('Step %d/%d: Smooth using k=%d' % (t, num_steps, k_step)); sys.stdout.flush()
        
#         Y = _calculate_pc_scores(S, d, seed=seed)
#         if dither > 0:
#             for l in range(d):
#                 ptp = np.ptp(Y[l, :])
#                 dy = (np.random.rand(Y.shape[1])-0.5)*ptp*dither
#                 Y[l, :] = Y[l, :] + dy

#         # determine cell-cell distances using smoothed matrix
#         t0 = time.time()
#         D = _calculate_distances(Y)
#         t1 = time.time()
#         print('Calculating pair-wise distance matrix took %.1f s.' % (t1-t0)); sys.stdout.flush()
        
#         t0 = time.time()
#         A = np.argsort(D, axis=1, kind='mergesort')
#         for j in range(matrix.shape[1]):
#             ind = A[j, :k_step]
#             S[:, j] = np.sum(X[:, ind], axis=1)

#         t1 = time.time()
#         print('Calculating the smoothed expression matrix took %.1f s.' %(t1-t0)); sys.stdout.flush()
#         k_step = k_step*step_factor

#     for j in range(matrix.shape[1]):
#         ind = A[j, :k]
#         N[j, :] = ind

#     matrix = ExpMatrix(X=S, genes=matrix.genes, cells=matrix.cells)

#     t1_total = time.time()
#     print('kNN-smoothing took %.1f s.' % (t1_total-t0_total)); sys.stdout.flush()

#     return matrix, N
