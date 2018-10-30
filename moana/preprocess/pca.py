# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions for dimensionality reduction of single-cell RNA-Seq data."""

from typing import Tuple
import logging
import time
from typing import Union

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

#from .. import util
from ..core import ExpMatrix
from ..util import get_component_labels
from . import median_normalize, freeman_tukey_transform

_LOGGER = logging.getLogger(__name__)


def pca(matrix: ExpMatrix, num_components: int = 10,
        normalize: bool = True,
        svd_solver: str = 'randomized',
        include_var: bool = True,
        seed: int = 0) -> Tuple[ExpMatrix, PCA]:
    """Performs PCA on scRNA-Seq data."""

    if normalize:
        # apply median-normalization
        matrix = median_normalize(matrix)

    # FT-transform the data
    tmatrix = freeman_tukey_transform(matrix)

    # perform PCA
    pca_model = PCA(n_components=num_components, svd_solver=svd_solver,
                    random_state=seed)
    t0 = time.time()
    Y = pca_model.fit_transform(tmatrix.T).T
    t1 = time.time()

    var_explained = np.cumsum(pca_model.explained_variance_ratio_)[-1]
    _LOGGER.info('PCA took %.1f s.', t1-t0)
    _LOGGER.info('The fraction of variance explained by the top %d PCs is '
                 '%.1f %%.', num_components, 100*var_explained)

    # convert to ExpMatrix
    sel_components = list(range(num_components))
    if include_var:
        dim_labels = get_component_labels(sel_components, pca_model)
    else:
        dim_labels = get_component_labels(sel_components)
    tmatrix = ExpMatrix(genes=dim_labels, cells=tmatrix.cells, X=Y)

    return tmatrix, pca_model


def apply_pca(
        matrix: ExpMatrix,
        pca_model: PCA, transcript_count: Union[float, int]):

    # normalize matrix to target transcript count
    num_transcripts = matrix.sum(axis=0)
    tmatrix = (transcript_count / num_transcripts) * matrix

    # FT-transform the data
    tmatrix = freeman_tukey_transform(matrix)

    total_var = tmatrix.var(axis=1).sum()

    # perform PCA
    Y = pca_model.transform(tmatrix.T).T

    sel_components = list(range(Y.shape[0]))
    Y = Y[sel_components, :]

    pc_var = Y.var(axis=1)

    dim_labels = ['PC %d (%.1f %%)' % (c+1, 100*(pc_var[c] / total_var))
                  for c in sel_components]

    tmatrix = ExpMatrix(genes=dim_labels, cells=tmatrix.cells, X=Y)
    return tmatrix


def determine_num_components(
        matrix: ExpMatrix, max_components: int = 50,
        num_iterations: int = 15, seed: int = 0) -> int:
    """Automatically the number of non-trivial PCs using a permutation test.

    You generally want to make sure the matrix is median-normalized first."""

    # Freeman-Tukey transform
    tmatrix = freeman_tukey_transform(matrix)

    # perform PCA
    pca_model = PCA(n_components=max_components,
                    svd_solver='randomized', random_state=seed)
    pca_model.fit(tmatrix.T)

    # perform PCA on permuted data
    np.random.seed(seed)
    R = tmatrix.X.copy()
    ratios = np.empty(num_iterations, dtype=np.float64)
    for t in range(num_iterations):
        _LOGGER.info('Permutation %d / %d...', t+1, num_iterations)
        for i in range(R.shape[0]):
            np.random.shuffle(R[i, :])
        perm_pca_model = PCA(
            n_components=1, svd_solver='randomized', random_state=seed)
        perm_pca_model.fit(R.T)
        ratios[t] = perm_pca_model.explained_variance_ratio_[0]

    perm_var_explained = np.median(ratios)
    _LOGGER.info('Variance threshold: %.3f %%', 100*perm_var_explained)

    num_components = np.sum(
        pca_model.explained_variance_ratio_ > perm_var_explained)

    return num_components
