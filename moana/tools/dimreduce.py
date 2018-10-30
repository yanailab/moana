# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions to further reduce the dimensionality of PCA-transformed data."""

import logging
import sys
import time

from sklearn.manifold import MDS, TSNE

from ..core import ExpMatrix
from . import add_dither

_LOGGER = logging.getLogger(__name__)


def mds(tmatrix: ExpMatrix, seed: int = 0, init: ExpMatrix = None,
        **kwargs) -> ExpMatrix:
    """Perform MDS on PCA-transformed data."""

    if init is not None:
        if not init.cells.equals(tmatrix.cells):
            raise ValueError('Cells do not match for initializing values.')

    # perform MDS on sampled cells
    model = MDS(random_state=seed, **kwargs)
    _LOGGER.info('Performing MDS...'); sys.stdout.flush()
    t0 = time.time()
    Y = model.fit_transform(tmatrix.T, init=init.T).T
    t1 = time.time()
    _LOGGER.info('MDS took %.1f s.' % (t1-t0))
    dim_labels = ['MDS dim. %d' % (l+1) for l in range(Y.shape[0])]
    mds_matrix = ExpMatrix(genes=dim_labels, cells=tmatrix.cells, X=Y)
    return mds_matrix


def tsne(tmatrix: ExpMatrix, seed: int = 0, dither: float = 0.0,
         perplexity: float = 30.0, **kwargs) -> ExpMatrix:
    """Perform t-SNE on PCA-transformedrandom_state data."""

    if dither > 0:
        # add dither
        tmatrix = add_dither(tmatrix, dither, seed=seed)

    # perform t-SNE on sampled cells
    model = TSNE(perplexity=perplexity, random_state=seed, **kwargs)
    _LOGGER.info('Performing t-SNE...'); sys.stdout.flush()
    t0 = time.time()
    Y = model.fit_transform(tmatrix.T).T
    t1 = time.time()
    _LOGGER.info('t-SNE took %.1f s.' % (t1-t0))
    dim_labels = ['t-SNE dim. %d' % (l+1) for l in range(Y.shape[0])]
    tsne_matrix = ExpMatrix(genes=dim_labels, cells=tmatrix.cells, X=Y)
    return tsne_matrix
