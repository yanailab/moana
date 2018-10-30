# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""PCA model for Moana scRNA-Seq classifiers."""

import logging
from typing import Union, Iterable

from ..core import ExpMatrix
from .. import preprocess as pp
from ..util import get_sel_components, get_component_labels
from .util import apply_pca

_LOGGER = logging.getLogger(__name__)


class PCAModel:
    """A PCA model for single-cell RNA-Seq data."""

    def __init__(self, components: Union[Iterable[int], int] = 10,
                 seed: int = 0) -> None:

        self.sel_components = get_sel_components(components)
        self.seed = seed

        self.genes_ = None
        self.pca_model_ = None
        self.transcript_count_ = None


    def _require_trained_model(self) -> None:
        if self.pca_model_ is None:
            raise RuntimeError('You must train the model first using "fit()"!')


    @property
    def num_components(self):
        """Returns the number of the highest selected principal component."""
        return max(self.sel_components) + 1

    @property
    def loadings_(self):
        X = self.pca_model_.components_.T[:, self.sel_components].copy()
        genes = self.genes_.copy()
        dim_labels = get_component_labels(self.sel_components)

        loadings = ExpMatrix(genes=genes, cells=dim_labels, X=X)
        return loadings


    def fit_transform(self, matrix: ExpMatrix,
                      include_var: bool = False) -> ExpMatrix:
        """Train the PCA model and return transformed expression matrix."""

        # determine median transcript count
        transcript_count = float(matrix.sum(axis=0).median())

        # perform PCA
        tmatrix, pca_model = pp.pca(matrix, self.num_components,
                                    include_var=include_var, seed=self.seed)

        tmatrix = tmatrix.iloc[self.sel_components]

        self.genes_ = matrix.genes.copy()
        self.pca_model_ = pca_model
        self.transcript_count_ = transcript_count

        return tmatrix


    def fit(self, matrix: ExpMatrix) -> None:
        """Train the PCA model."""

        self.fit_transform(matrix)


    def transform(self, matrix: ExpMatrix,
                  include_var: bool = False) -> ExpMatrix:
        """Apply the PCA model."""

        self._require_trained_model()

        # apply PCA
        #tmatrix = apply_pca(
        #    matrix, self.pca_model_, self.transcript_count_,
        #    valid_genes=self.genes_)

        ### 1. Make sure genes are the same
        # provide some metrics
        num_valid_genes = matrix.genes.isin(self.genes_).sum()
        _LOGGER.info('Recognized %d / %d genes (%.1f %%) in the '
                     'expression matrix.', num_valid_genes, matrix.p,
                     100*(num_valid_genes/matrix.p))

        #num_found_genes = valid_genes.isin(matrix.genes).sum()
        #_LOGGER.info('Found %d / %d genes (%.1f %%) present in the '
        #                'training data.', num_valid_genes, len(valid_genes),
        #                100*(num_found_genes/len(valid_genes)))

        # re-index matrix and fill with zeros
        matrix = matrix.reindex(
            index=self.genes_, fill_value=0)
        
        ### 2. normalize to specified transcript count
        num_transcripts = matrix.sum(axis=0)
        _LOGGER.info('Matrix will be scaled %.2fx (on average) before '
                    'FT-transform and projection into into PC space.',
                    self.transcript_count_ / num_transcripts.median())
        matrix = (self.transcript_count_ / num_transcripts) * matrix

        ### 3. FT-transform
        tmatrix = pp.freeman_tukey_transform(matrix)

        ### 4. Project into PC space
        Y = self.pca_model_.transform(tmatrix.T).T
        Y = Y[self.sel_components, :]

        total_var = tmatrix.var(axis=1).sum()
        pc_var = Y.var(axis=1).sum()
        _LOGGER.info('The projection onto the selected PCs retains %.1f %% '
                     'of the total variance in the (FT-transformed) data.',
                     100 * (pc_var / total_var))

        if include_var:
            dim_labels = get_component_labels(
                self.sel_components, self.pca_model_)
        else:
            dim_labels = get_component_labels(
                self.sel_components)

        # generate ExpMatrix object
        tmatrix = ExpMatrix(genes=dim_labels, cells=matrix.cells, X=Y)

        return tmatrix
