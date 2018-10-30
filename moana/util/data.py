# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Utility functions for analyzing scRNA-Seq data."""

import sys
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

from ..core import ExpMatrix

_LOGGER = logging.getLogger(__name__)


def read_colorscale(cscale_fpath: str) -> List[Tuple[float, str]]:
    """Return a colorscale in the format expected by plotly.

    Parameters
    ----------
    cscale_fpath : str
        Path of a plain-text file containing the colorscale. 

    Returns
    -------
    list
        The colorscale.
        
    Notes
    -----
    A plotly colorscale is a list where each item is a pair
    (i.e., a tuple with two elements) consisting of
    a decimal number x between 0 and 1 and a corresponding "rgb(r,g,b)" string,
    where r, g, and b are integers between 0 and 255.

    The `cmap_file` is a tab-separated text file containing four columns
    (x,r,g,b), so that each row corresponds to an entry in the list
    described above.
    """
    cm = np.loadtxt(cscale_fpath, delimiter='\t', dtype=np.float64)
    # x = cm[:, 0]
    rgb = np.int64(cm[:, 1:])  # normalize to 0-1?
    n = cm.shape[0]
    colorscale = []
    for i in range(n):
        colorscale.append(
            (i / float(n - 1),
             'rgb(%d, %d, %d)' % (rgb[i, 0], rgb[i, 1], rgb[i, 2]))
        )
    return colorscale


def filter_variable_genes(matrix: ExpMatrix,
                          num_variable_genes: int) -> ExpMatrix:
    var = matrix.var(axis=1).sort_values(ascending=False)
    sel = var.index[:num_variable_genes]
    filt_matrix = matrix.loc[sel]
    return filt_matrix


def convert_from_ensembl_ids(matrix: ExpMatrix,
                             gene_table: pd.DataFrame) -> ExpMatrix:
    """Convert matrix from Ensembl IDs to gene names."""
    
    _LOGGER.info('Converting matrix with shape: %s', str(matrix.shape))
    
    # only keep expressed genes
    matrix = matrix.loc[matrix.sum(axis=1)>0]
    _LOGGER.info('Shape after filtering for expressed genes: %s', str(matrix.shape))

    # determine number of transcripts per cell
    total_counts = matrix.sum(axis=0)

    # check number of duplicates
    #num_duplicates = matrix.genes.duplicated().sum()
    #print('Number of duplicates:', num_duplicates)
    
    # only keep valid genes (e.g., protein-coding)
    all_gene_ids = set(gene_table.index.tolist())
    matrix = matrix.loc[matrix.genes.isin(all_gene_ids)]
    _LOGGER.info('Shape after filtering for valid genes: %s', str(matrix.shape))
    
    # map Ensembl IDs to gene names
    matrix.genes = matrix.genes.map(lambda x: gene_table.loc[x, 'name'])
    
    # check number of duplicates
    num_duplicates = matrix.genes.duplicated().sum()
    _LOGGER.info('Number of duplicate gene names: %d', num_duplicates)
    
    if num_duplicates > 0:
        _LOGGER.info('Aggregating duplicates...')
        sys.stdout.flush()
        matrix = ExpMatrix(matrix.groupby(level=0).aggregate(np.sum))
        _LOGGER.info('done!')
        _LOGGER.info('Shape after collapsing duplicated genes: %s', str(matrix.shape))

    
    # determine fraction of transcripts kept per cell
    kept_counts = matrix.sum(axis=0)
    frac_counts_kept = kept_counts / total_counts
    
    _LOGGER.info('Kept % of transcripts:')
    _LOGGER.info((100*frac_counts_kept).describe())
    _LOGGER.info('')

    return matrix
