# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions for filtering scRNA-Seq data."""

import pandas as pd

from ..core import ExpMatrix, CellAnnMatrix
from .util import get_ribosomal_genes, get_mitochondrial_genes

from typing import List, Union, Tuple

#_DEFAULT_COLORSCALE_FILE = resource_filename(
#    'moana', 'data/RdBu_r_colormap.tsv')


def filter_genes(genes: pd.Index, species: str = 'human') -> pd.Index:
    """Remove ribosomal and mitochondrial genes."""
    ribo_genes = get_ribosomal_genes(species)
    mito_genes = get_mitochondrial_genes(species)
    
    excluded_genes = ribo_genes + mito_genes
    sel_genes = genes[~genes.isin(excluded_genes)]
    
    return sel_genes


def filter_matrix(matrix: ExpMatrix, min_transcripts: int = 1000,
                  max_mito_frac: Union[float, int] = 0.10) \
                  -> Tuple[ExpMatrix, pd.DataFrame]:

    ribo_genes = get_ribosomal_genes()
    mito_genes = get_mitochondrial_genes()

    num_transcripts = matrix.sum(axis=0)    
    frac_ribo = matrix.loc[matrix.genes & ribo_genes].sum(axis=0) / \
            num_transcripts
    frac_mito = matrix.loc[matrix.genes & mito_genes].sum(axis=0) / \
            num_transcripts

    # exclude ribosomal and mitochondrial genes
    matrix = matrix.sort_genes()
    matrix = matrix.loc[~matrix.genes.isin(mito_genes + ribo_genes)]

    # select cells that pass QC filter
    filt_num_transcripts = matrix.sum(axis=0)
    sel = (filt_num_transcripts >= min_transcripts) & \
            (frac_mito <= max_mito_frac)
    matrix = matrix.loc[:, sel]
    num_transcripts = num_transcripts.loc[sel]
    frac_ribo = frac_ribo.loc[sel]
    frac_mito = frac_mito.loc[sel]

    df = pd.concat([num_transcripts, frac_ribo, frac_mito], axis=1)
    df.columns = ['num_transcripts', 'frac_ribo', 'frac_mito']

    df = CellAnnMatrix(df)

    return matrix, df
