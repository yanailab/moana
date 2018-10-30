# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Utility functions for scRNA-Seq cell type classification with Moana."""


import logging
from typing import Tuple, Union, Iterable, Dict, List
import time

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np
from math import log, ceil, floor

from ..core import ExpMatrix, CellAnnVector
from .. import preprocess as pp
from ..util import get_sel_components, get_component_labels
from .. import tools

_LOGGER = logging.getLogger(__name__)


def split_cells(
        cells: pd.Index,
        train_size: Union[float, int] = 2/3,
        test_size: Union[float, int] = None,
        seed: int = 0):
    ss = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=seed)
    X = np.empty((cells.size, 1))
    train_ind, test_ind = list(ss.split(X))[0]
    train_cells = cells[train_ind]
    test_cells = cells[test_ind]
    return train_cells, test_cells


def balance_cells(
        cell_labels: CellAnnVector,
        max_cells: int = 2000,
        seed: int = 0) -> CellAnnVector:
    """Select cells to create a balanced dataset (for training)."""

    vc = cell_labels.value_counts()
    if vc.size > 2:
        raise ValueError('Cannot handle more than two classes!')

    # determine number of cells in the minority class
    num_cells2 = min(vc[1], int(max_cells / 2.0))
    num_cells1 = min(vc[0], max_cells - num_cells2)

    # select corresponding number of cells from the majority class
    sel1 = (cell_labels == vc.index[0])
    sel2 = (cell_labels == vc.index[1])

    sel_cells1 = cell_labels.loc[sel1].cells
    if sel_cells1.size > num_cells1:
        sel_cells1 = tools.downsample_cells(
            sel_cells1, num_cells1, seed=seed)

    sel_cells2 = cell_labels.loc[sel2].cells
    if sel_cells2.size > num_cells2:
        sel_cells2 = tools.downsample_cells(
            sel_cells2, num_cells2, seed=seed)

    sel_cells = sel_cells1.append(sel_cells2)
    sel_cell_labels = cell_labels.loc[sel_cells]

    return sel_cell_labels


def calculate_mean_precision(truth: CellAnnVector, predictions: CellAnnVector):
    """Calculate mean precision across all classes."""
    mean_precision = precision_score(truth, predictions, average='macro')
    return mean_precision


def get_precision_summary(truth: CellAnnVector, predictions: CellAnnVector):
    # calculate average precision score
    avg_precision = precision_score(
        truth, predictions, average='macro')

    vc = truth.value_counts()

    # calculate cell-type specific precision scores
    #def print_precision()
    ctype_precisions = {}
    for ctype in vc.index:
        ctype_precisions[ctype] = precision_score(
            truth, predictions, labels=[ctype], average='macro')
    precision_str = '; '.join(
        '%s - %.1f %%' % (ctype, 100*ctype_precisions[ctype])
        for ctype in vc.index)
    
    summary_str = 'Mean = %.1f %%. %s.' % (100*avg_precision, precision_str)

    return summary_str


def apply_pca(matrix: ExpMatrix,
              pca_model: PCA,
              pca_transcript_count: float,
              components: Union[int, Iterable[int]] = None,
              include_var: bool = True,
              valid_genes: pd.Index = None) -> ExpMatrix:
    """Apply PCA to a smoothed matrix."""

    if components is not None:
        sel_components = get_sel_components(components)
    else:
        sel_components = list(range(pca_model.n_components))

    num_components = max(sel_components) + 1
    if num_components > pca_model.n_components:
        raise ValueError('Cannot compute highest selected PC (%d), because '
                         'the model only contains %d components.'
                         % (num_components, pca_model.n_components))

    # 1. Make sure genes are the same
    if valid_genes is not None:
        # provide some metrics
        num_valid_genes = matrix.genes.isin(valid_genes).sum()
        _LOGGER.info('Recognized %d / %d genes (%.1f %%) in the '
                        'expression matrix.', num_valid_genes, matrix.p,
                        100*(num_valid_genes/matrix.p))

        #num_found_genes = valid_genes.isin(matrix.genes).sum()
        #_LOGGER.info('Found %d / %d genes (%.1f %%) present in the '
        #                'training data.', num_valid_genes, len(valid_genes),
        #                100*(num_found_genes/len(valid_genes)))

        # re-index matrix and fill with zeros
        matrix = matrix.reindex(
            index=valid_genes, fill_value=0)
    
    # 2. normalize to specified transcript count
    num_transcripts = matrix.sum(axis=0)
    _LOGGER.info('Matrix will be scaled %.2fx (on average) before '
                 'FT-transform and projection into into PC space.'
                 % (pca_transcript_count / num_transcripts.median()))
    tmatrix = (pca_transcript_count / num_transcripts) * matrix

    # 3. FT-transform
    tmatrix = pp.freeman_tukey_transform(tmatrix)

    total_var = tmatrix.var(axis=1).sum()

    # 4. Project into PC space
    Y = pca_model.transform(tmatrix.T).T
    Y = Y[sel_components, :]

    pc_var = Y.var(axis=1)

    if include_var:
        dim_labels = ['PC %d (%.1f %%)' % (c+1, 100*(pc_var[c] / total_var))
                      for c in sel_components]
    else:
        dim_labels = ['PC %d' % (c+1) for c in sel_components]

    tmatrix = ExpMatrix(genes=dim_labels, cells=matrix.cells, X=Y)

    return tmatrix


def select_pca_model(
        transcript_count: Union[float, int],
        smooth_pca_models: Dict[float, PCA],
        min_transcript_frac: Union[float, int]) -> Tuple[PCA, float]:
    """Selects the appropriate PCA model for the given transcript count."""

    try:
        selected_count, selected_model = next(iter(smooth_pca_models.items()))
    except StopIteration:
        raise ValueError('No smoothing PCA models available!')

    for count, pca_model in smooth_pca_models.items():
        if count * min_transcript_frac <= transcript_count:
            selected_model = pca_model
            selected_count = count
        else:
            break

    return selected_model, selected_count


def apply_smoothing(
        matrix: ExpMatrix,
        transcript_count: float,
        min_transcript_frac: Union[float, int],
        smooth_pca_models: Dict[float, PCA],
        valid_genes: pd.Index = None) -> Tuple[ExpMatrix, int]:
    """Apply smoothing using pre-specified PCA transforms."""

    num_transcripts = matrix.sum(axis=0)
    cur_transcript_count = num_transcripts.median()
    _LOGGER.info('Median transcript count before smoothing: %.1f',
                 cur_transcript_count)
    _LOGGER.info('Transcript count threshold to be exceeded using '
                 'smoothing: %.1f',
                 transcript_count * min_transcript_frac)

    total_transcripts = num_transcripts.sum()
    if total_transcripts < transcript_count * min_transcript_frac:
        raise ValueError(
            'Specified transcript count threshold is unattainable!')

    X = np.array(matrix.X, dtype=np.float64, order='F', copy=False)
    smoothed_matrix = matrix.copy()
    k = 1
    while cur_transcript_count < transcript_count * min_transcript_frac:
        k = min(k*2, matrix.n)
        _LOGGER.info('Smoothing with k=%d...', k)

        pca_model, pca_transcript_count = select_pca_model(
            cur_transcript_count, smooth_pca_models, min_transcript_frac)
        _LOGGER.info('Selected PCA model with transcript_count=%.1f',
                     pca_transcript_count)

        tmatrix = apply_pca(smoothed_matrix, pca_model, pca_transcript_count,
                            valid_genes=valid_genes)

        t0 = time.time()
        D = pairwise_distances(tmatrix.T, n_jobs=1, metric='euclidean')
        t1 = time.time()
        _LOGGER.info('Calculating the pairwise distance matrix took %.1f s.',
                     t1-t0)

        t0 = time.time()
        A = np.argsort(D, axis=1, kind='mergesort')
        S = np.array(smoothed_matrix.values, dtype=np.float64, order='F',
                     copy=False)
        for j in range(matrix.shape[1]):
            ind = A[j, :k]
            S[:, j] = np.sum(X[:, ind], axis=1)

        t1 = time.time()
        _LOGGER.info('Calculating the smoothed expression matrix took %.1f s.',
                     t1-t0)

        smoothed_matrix = ExpMatrix(
            genes=matrix.genes, cells=matrix.cells, X=S)
        cur_transcript_count = smoothed_matrix.sum(axis=0).median()
        _LOGGER.info('The new transcript count is: %.1f', cur_transcript_count)

    _LOGGER.info('Applied smoothing with k=%d', k)

    return smoothed_matrix, k


def calculate_accuracy_scores(
        truth: CellAnnVector, pred: CellAnnVector) -> pd.Series:

    vc = truth.value_counts()
    cell_types = vc.index.tolist()
    
    level1 = ['overall', 'overall_class_avg', 'overall_cohen'] + cell_types
    index = pd.Index(level1)
    #level2 = ['accuracy', 'std_dev']    
    #tuples = (level1, level2)
    #columns = pd.MultiIndex.from_product(tuples, names=['cell_type', 'value'])
    
    acc = pd.Series(index=level1)
    acc.loc['overall'] = accuracy_score(truth, pred)
    acc.loc['overall_cohen'] = cohen_kappa_score(truth, pred)
    ct_acc = []
    for ct in cell_types:
        sel = (truth == ct)
        score = accuracy_score(truth.loc[sel], pred.loc[sel])
        acc.loc[ct] = score
        ct_acc.append(score)
        
    acc.loc['overall_class_avg'] = np.float64(ct_acc).mean()
    
    return acc    


def create_training_data(
        cell_labels: CellAnnVector, max_cells: int = 2000, seed: int = 0):
    # we assume that there are only two populations
    vc = cell_labels.value_count()
    pop_size = min(vc[-1], max_cells / 2.0)
    
    #all_cells = 
    #if vc[-1] >= max_cells / 2.0:
