#from typing import Dict
import logging
import random
from itertools import chain

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

from ..core import ExpMatrix
from ..core import CellAnnVector

_LOGGER = logging.getLogger(__name__)


def downsample_cells(cells: pd.Index, num_cells: int,
                     cell_labels: CellAnnVector = None,
                     seed: int = 0) -> pd.Index:
    """Downsample cells while maintaining cluster size proportions."""

    if num_cells >= cells.size:
        raise ValueError('Number of cells to sample (%d) needs to be smaller '
                         'than total number of cells (%d)'
                         % (num_cells, cells.size))

    if num_cells <= 0:
        raise ValueError('Number of cells to sample (%d) must be a positive '
                         'integer.' % num_cells)

    if cell_labels is None:
        # make a mock cluster
        cell_labels = CellAnnVector(cells=cells, data=np.zeros(cells.size))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=num_cells,
                                 test_size=cells.size-num_cells,
                                 random_state=seed)

    # Somehow, the cluster labels change the results of the split
    # so we replace them with labels that are invariant to re-labeling.
    # (Probably, scikit-learn sorts the labels alphanumerically...)

    # get all the cluster labels in the order they appear
    unique_clusters = cell_labels.loc[~cell_labels.duplicated()]

    # convert them to numbers
    cluster_labels = dict([(label, i)
                           for i, label in enumerate(unique_clusters)])

    anonymous_labels = cell_labels.map(cluster_labels)
    
    sel_indices = list(sss.split(np.arange(cells.size),
                                 anonymous_labels))[0][0]
    
    sampled_cells = cells[sel_indices]
    return sampled_cells


def downsample_matrix(matrix: ExpMatrix, num_cells: int,
                      cell_labels: CellAnnVector = None,
                      seed: int = 0) -> ExpMatrix:
    """Downsample cells in matrix while maintaining cluster size proportions.
    
    """
    sampled_cells = downsample_cells(matrix.cells, num_cells=num_cells,
                                     cell_labels=cell_labels,
                                     seed=seed)
    sample_matrix = matrix.loc[:, sampled_cells]
    return sample_matrix


def resample_cells(cells: pd.Index,
                   cell_labels: CellAnnVector = None,
                   seed: int = 0) -> pd.Index:
    """Resample cells (with replacement)."""

    if cell_labels is None:
        cell_labels = CellAnnVector(cells=cells, data=[0]*cells.size)
    else:
        cell_labels = cell_labels.loc[cells]

    vc = cell_labels.value_counts()
    random.seed(seed)
    sample_seeds = [random.getrandbits(32) for i in range(len(vc.index))]
    sampled_cells = []
    for i, (ctype, count) in enumerate(vc.iteritems()):
        sampled_cells.append(
            (cells[cell_labels == ctype]).to_series().sample(
                n=count, random_state=sample_seeds[i], replace=True).tolist())

    sampled_cells = pd.Index(chain.from_iterable(sampled_cells))
    # shuffle the index
    sampled_cells = sampled_cells.to_series().sample(
        frac=1.0, replace=False, random_state=seed).index

    return sampled_cells


def resample_matrix(matrix: ExpMatrix, cell_labels: CellAnnVector = None,
                    seed: int = 0) -> ExpMatrix:
    """Resample cells in matrix (with replacement)."""
    cells_sampled = resample_cells(matrix.cells, cell_labels=cell_labels,
                                   seed=seed)
    sample_matrix = matrix.loc[:, cells_sampled]
    # make sure cell names are unique
    sample_matrix.cells = ['%s_%d' % (c, j)
                           for j, c in enumerate(sample_matrix.cells)]
    return sample_matrix, cells_sampled
