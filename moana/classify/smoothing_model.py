# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Smoothing model for Moana scRNA-Seq classifiers."""

import logging
from typing import Union, Iterable
from collections import OrderedDict
import time

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from .. import preprocess as pp
from ..core import ExpMatrix
from ..util import get_sel_components
from .util import apply_smoothing
from . import PCAModel

_LOGGER = logging.getLogger(__name__)


class SmoothingModel:
    """A smoothing model for single-cell RNA-Seq data."""

    def __init__(self, k: int = 1, d: int = 20,
                 dither: Union[float, int] = 0,
                 components: Union[Iterable[int], int] = 2,
                 min_transcript_frac: Union[float, int] = 0.9,
                 smooth_min_transcripts: Union[int, float] = 500,
                 seed: int = 0) -> None:

        self.k = k
        self.d = d
        self.dither = dither
        self.sel_components = get_sel_components(components)
        self.seed = seed
        self.min_transcript_frac = min_transcript_frac
        self.smooth_min_transcripts = smooth_min_transcripts

        self.transcript_count_ = None
        self.smooth_pca_models_ = None


    @property
    def genes_(self):
        return next(iter(self.smooth_pca_models_.values())).genes_


    def _require_trained_model(self) -> None:
        if self.smooth_pca_models_ is None:
            raise RuntimeError('You must train the model first using "fit()"!')


    def _select_pca_model(self, transcript_count: float,
                          min_transcript_frac: Union[float, int] = None):

        if min_transcript_frac is None:
            min_transcript_frac = self.min_transcript_frac

        #try:
        #    selected_count, selected_model = next(iter(smooth_pca_models.items()))
        #except StopIteration:
        #    raise ValueError('No smoothing PCA models available!')

        iter_models = iter(self.smooth_pca_models_.items())

        selected_count, selected_model = next(iter_models)

        for count, pca_model in iter_models:
            if count * min_transcript_frac <= transcript_count:
                selected_model = pca_model
                selected_count = count
            else:
                break

        return selected_model, selected_count


    def fit(self, matrix: ExpMatrix, is_smoothed: bool = False) -> None:
        """Train the smoothing model."""

        if not is_smoothed:
            # apply kNN-smoothing
            smoothed_matrix = pp.knn_smoothing(
                matrix, self.k, d=self.d, dither=self.dither, seed=self.seed)
        else:
            smoothed_matrix = matrix

        # determine median transcript count of smoothed matrix
        transcript_count = float(smoothed_matrix.sum(axis=0).median())

        if transcript_count < self.smooth_min_transcripts:
            raise ValueError(
                'The given matrix has fewer transcripts per cell than the '
                'value of the "smooth_min_transcripts" parameter. Cannot '
                'construct a smoothing model under these conditions.')

        # construct PCA models for down-scaled data
        smooth_pca_models = []
        new_transcript_count = transcript_count
        scaled_matrix = smoothed_matrix
        while new_transcript_count >= self.smooth_min_transcripts:
            scaled_matrix = scaled_matrix / 2.0  # creates a copy, unlike "/="!
            new_transcript_count /= 2
            _LOGGER.info('Generating PCA model for %.1f transcripts.',
                         new_transcript_count)

            pca_model = PCAModel(self.sel_components, self.seed)
            pca_model.fit(scaled_matrix)
            smooth_pca_models.append((new_transcript_count, pca_model))

            #_, scaled_pca_model = pp.pca(
            #    scaled_matrix, self.d, seed=self.seed)
            #smooth_pca_models.append((new_transcript_count, scaled_pca_model))

        self.transcript_count_ = transcript_count
        self.smooth_pca_models_ = OrderedDict(reversed(smooth_pca_models))


    def transform(self, matrix: ExpMatrix,
                  min_transcript_frac: Union[float, int] = None) -> ExpMatrix:
        """Apply the smoothing model."""

        self._require_trained_model()

        if min_transcript_frac is None:
            min_transcript_frac = self.min_transcript_frac

        t0_total = time.time()

        num_transcripts = matrix.sum(axis=0)
        cur_transcript_count = num_transcripts.median()
        transcript_thresh = self.transcript_count_ * min_transcript_frac
        _LOGGER.info('Median transcript count before smoothing: %.1f',
                    cur_transcript_count)
        _LOGGER.info('Transcript count threshold to be exceeded using '
                    'smoothing: %.1f', transcript_thresh)

        total_transcripts = num_transcripts.sum()
        if total_transcripts < transcript_thresh:
            raise ValueError(
                'Specified transcript count threshold is unattainable!')

        smoothed_matrix = matrix.copy().astype(np.float64, copy=False)
        if cur_transcript_count >= transcript_thresh:
            _LOGGER.info('No smoothing required!')
            return smoothed_matrix, 1

        k = 1
        X = np.array(matrix.X, dtype=np.float64, order='F', copy=False)
        while cur_transcript_count < transcript_thresh:
            k = min(k*2, matrix.n)
            _LOGGER.info('Smoothing with k=%d...', k)

            pca_model, pca_transcript_count = \
                    self._select_pca_model(cur_transcript_count)

            _LOGGER.info('Selected PCA model with transcript_count=%.1f',
                         pca_transcript_count)

            tmatrix = pca_model.transform(smoothed_matrix)

            t0 = time.time()
            D = pairwise_distances(tmatrix.T, n_jobs=1, metric='euclidean')
            t1 = time.time()
            _LOGGER.info('Calculating the pairwise distance matrix took '
                         '%.1f s.', t1-t0)

            t0 = time.time()
            A = np.argsort(D, axis=1, kind='mergesort')
            S = np.array(smoothed_matrix.values, order='F', copy=False)
            for j in range(matrix.shape[1]):
                ind = A[j, :k]
                S[:, j] = np.sum(X[:, ind], axis=1)

            t1 = time.time()
            _LOGGER.info('Calculating the smoothed expression matrix took '
                         '%.1f s.', t1-t0)

            smoothed_matrix = ExpMatrix(
                genes=matrix.genes, cells=matrix.cells, X=S)
            cur_transcript_count = smoothed_matrix.sum(axis=0).median()
            _LOGGER.info('The new transcript count is: %.1f',
                         cur_transcript_count)

        t1_total = time.time()

        _LOGGER.info('Applied smoothing with k=%d (took %.1f s)',
                     k, t1_total - t0_total)

        return smoothed_matrix, k
