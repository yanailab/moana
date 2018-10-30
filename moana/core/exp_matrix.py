# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `ExpMatrix` class."""

import logging
import importlib
import hashlib
from collections import OrderedDict
from typing import Iterable
import tarfile
import io
import os
import csv

import pandas as pd
import numpy as np
import scipy.io
from scipy import sparse

from . import Matrix

exp_vector = importlib.import_module('.exp_vector', package='moana.core')
# - "from . import exp_vector" fails due to cyclical imports

_LOGGER = logging.getLogger(__name__)


class ExpMatrix(Matrix):
    """A gene expression matrix.

    This class inherits from `moana.core.Matrix`.

    Parameters
    ----------
    X : 2-dimensional `numpy.ndarray`
        See :attr:`X` attribute.

    Keyword-only Parameters
    -----------------------
    genes : list or tuple of str
        See :attr:`genes` attribute.
    cells : list or tuple of str
        See :attr:`cells` attribute.

    Additional Parameters
    ---------------------
    All `pandas.DataFrame` parameters.

    Attributes
    ----------
    genes : tuple of str
        The names of the genes (rows) in the matrix.
    cells : tuple of str
        The names of the cells (columns) in the matrix.
    X : 2-dimensional `numpy.ndarray`
        The matrix of expression values.
    """
    def __init__(self, *args,
                 X: np.ndarray = None,
                 genes: Iterable[str] = None, gene_label: str = None,
                 cells: Iterable[str] = None, cell_label: str = None,
                 **kwargs) -> None:
        
        # check if user provided "X" keyword argument...
        # if so, pass it as the "data" argument to the DataFrame constructor
        if X is not None:
            if 'data' in kwargs:
                raise ValueError(
                    'Cannot specify both "X" and "data" arguments.')
            kwargs['data'] = X

        if genes is not None:
            if 'index' in kwargs:
                raise ValueError(
                    'Cannot specify both "genes" and "index" arguments.')
            kwargs['index'] = genes

        if cells is not None:
            if 'columns' in kwargs:
                raise ValueError(
                    'Cannot specify both "cells" and "columns" arguments.')
            kwargs['columns'] = cells

        # call base class constructor
        Matrix.__init__(self, *args, **kwargs)

        # set index name (default: "Genes")
        if gene_label is not None:
            self.index.name = gene_label
        elif self.index.name is None:
            self.index.name = 'Genes'

        # set column name (default: "Cells")
        if cell_label is not None:
            self.columns.name = cell_label
        elif self.columns.name is None:
            self.columns.name = 'Cells'

    # def __eq__(self, other):
    #     return Matrix.__eq__(self, other)

    # def __ne__(self, other):
    #     return Matrix.__ne__(self, other)

    # def __repr__(self):
    #     return Matrix.__repr__(self)

    #def __str__(self):
    #    return '<%s object with p=%d genes and n=%d cells>' \
    #           % (self.__class__.__name__, self.p, self.n)

    # @property
    # def hash(self):
    #     # warning: involves copying all the data
    #     gene_str = ','.join(str(s) for s in self.genes)
    #     cell_str = ','.join(str(s) for s in self.cells)
    #     data_str = ';'.join([gene_str, cell_str]) + ';'
    #     data = data_str.encode('UTF-8') + self.X.tobytes()
    #     return str(hashlib.md5(data).hexdigest())

    @property
    def _constructor(self):
        return ExpMatrix
    
    @property
    def _constructor_sliced(self):
        return exp_vector.ExpVector

    @property
    def p(self):
        """The number of genes."""
        return self.shape[0]

    @property
    def n(self):
        """The number of cells."""
        return self.shape[1]

    @property
    def genes(self):
        """Alias for `DataFrame.index`."""
        # return tuple(str(g) for g in self.index)
        return self.index

    @genes.setter
    def genes(self, gene_list):
        self.index = gene_list

    @property
    def cells(self):
        """Alias for `DataFrame.columns`."""
        # return tuple(str(s) for s in self.columns)
        return self.columns

    @cells.setter
    def cells(self, cell_list):
        self.columns = cell_list

    @property
    def X(self):
        """Alias for `DataFrame.values`."""
        return self.values


    def sum(self, axis=0, dtype=None):
        """Sum over rows or columns.
        
        Overrides the default `pandas.DataFrame.sum()` function to prevent
        temporary in-memory copies.
        """
        if axis not in [0, 1, 'index', 'columns']:
            raise ValueError('"axis" parameter must be one of 0, 1, "index", '
                             'or "columns".')

        sum_kwargs = {}
        if dtype is not None:
            sum_kwargs['dtype'] = dtype
        y = self.values.sum(axis=axis, **sum_kwargs)

        if axis == 0 or axis == 'index':
            y = exp_vector.ExpVector(y, genes=self.cells)
        else:
            y = exp_vector.ExpVector(y, genes=self.genes)

        return y


    def sort_genes(self, stable=True, inplace=False, ascending=True):
        """Sort the rows of the matrix alphabetically by gene name.

        Parameters
        ----------
        stable: bool, optional
            Whether to use a stable sorting algorithm. [True]
        inplace: bool, optional
            Whether to perform the operation in place.[False]
        ascending: bool, optional
            Whether to sort in ascending order [True]
        
        Returns
        -------
        `ExpMatrix`
            The sorted matrix.
        """
        kind = 'quicksort'
        if stable:
            kind = 'mergesort'
        return self.sort_index(kind=kind, inplace=inplace, ascending=ascending)


    def sort_cells(self, stable=True, inplace=False, ascending=True):
        """Sort the columns of the matrix alphabetically by cell name.

        Parameters
        ----------
        stable: bool, optional
            Whether to use a stable sorting algorithm. [True]
        inplace: bool, optional
            Whether to perform the operation in place.[False]
        ascending: bool, optional
            Whether to sort in ascending order [True]

        Returns
        -------
        `ExpMatrix`
            The sorted matrix.
        """
        kind = 'quicksort'
        if stable:
            kind = 'mergesort'
        return self.sort_index(axis=1, kind=kind, inplace=inplace,
                               ascending=ascending)

    def center_genes(self, use_median=False, inplace=False):
        """Center the expression of each gene (row)."""
        if use_median:
            X = self.X - \
                np.tile(np.median(self.X, axis=1), (self.n, 1)).T
        else:
            X = self.X - \
                np.tile(np.mean(self.X, axis=1), (self.n, 1)).T

        if inplace:
            self.X[:,:] = X
            matrix = self
        else:
            matrix = ExpMatrix(genes=self.genes, cells=self.cells,
                               X=X)
        return matrix


    def standardize_genes(self, inplace=False):
        """Standardize the expression of each gene (row), yielding z-scores."""
        matrix = self.center_genes(inplace=inplace)
        matrix.X[:,:] = matrix.X / \
            np.tile(np.std(matrix.X, axis=1, ddof=1), (matrix.n, 1)).T
        return matrix


    def filter_genes(self, gene_names : Iterable[str], inplace=False):
        """Filter the expression matrix against a _genome (set of genes).

        Parameters
        ----------
        gene_names: list of str
            The genome to filter the genes against.
        inplace: bool, optional
            Whether to perform the operation in-place.

        Returns
        -------
        ExpMatrix
            The filtered expression matrix.
        """

        return self.drop(set(self.genes) - set(gene_names),
                         inplace=inplace)

    @property
    def cell_correlations(self):
        """Returns an `ExpMatrix` containing all pairwise cell correlations.

        Returns
        -------
        `ExpMatrix`
            The cell correlation matrix.

        """
        C = np.corrcoef(self.X.T)
        corr_matrix = ExpMatrix(genes=self.cells, cells=self.cells, X=C)
        return corr_matrix


    @classmethod
    def read_tsv(cls, file_path: str,
                 encoding: str = 'UTF-8', sep: str = '\t',
                 **kwargs):
        """Read expression matrix from a tab-delimited text file.

        Parameters
        ----------
        file_path: str
            The path of the text file.
        encoding: str, optional
            The file encoding. ("UTF-8")
        sep: str, optional
            The separator. ("\t")

        Returns
        -------
        `ExpMatrix`
            The expression matrix.
        """

        matrix = super(ExpMatrix, cls).read_tsv(
            file_path, encoding, sep, **kwargs)

        if matrix.index.name is None:
            matrix.index.name = 'Genes'
        
        if matrix.columns.name is None:
            matrix.columns.name = 'Cells'

        return matrix


    def write_tsv(self, file_path: str,
                  encoding: str = 'UTF-8', sep: str = '\t') -> None:
        """Write expression matrix to a tab-delimited text file.

        Parameters
        ----------
        file_path: str
            The path of the output file.
        encoding: str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        None
        """
        Matrix.write_tsv(self, file_path, encoding, sep, float_format='%.5f')

        #_LOGGER.info('Wrote %d x %d expression matrix to "%s".',
        #            self.p, self.n, file_path)


    def write_sparse(self, file_path: str) -> None:
        """Write a sparse representation to a tab-delimited text file.
        
        TODO: docstring"""

        coo = sparse.coo_matrix(self.X.T)
        data = OrderedDict([(0, coo.row+1), (1, coo.col+1), (2, coo.data)])
        df = pd.DataFrame(data, columns=data.keys())
        with open(os.path.expanduser(file_path), 'w') as ofh:
            ofh.write('%%MatrixMarket matrix coordinate real general\n')
            ofh.write('%%%s\n' % '\t'.join(self.cells.astype(str)))
            ofh.write('%%%s\n' % '\t'.join(self.genes.astype(str)))
            ofh.write('%\n')
            ofh.write('%d %d %d\n' % (coo.shape[0], coo.shape[1], coo.nnz))
            df.to_csv(ofh, sep=' ', float_format='%.5f',
                      header=None, index=None)
        _LOGGER.info('Wrote matrix with %d genes and %d cells to "%s".',
                     self.p, self.n, file_path)


    @classmethod
    def read_sparse(cls, file_path: str):
        """Read a sparse representation from a tab-delimited text file.
        
        TODO: docstring"""
        
        with open(os.path.expanduser(file_path)) as fh:
            next(fh)  # skip header line
            cells = next(fh)[1:-1].split('\t')
            genes = next(fh)[1:-1].split('\t')
            next(fh)
            m, n, nnz = [int(s) for s in next(fh)[:-1].split(' ')]
        
        t = pd.read_csv(file_path, sep=' ', skiprows=5, header=None,
                        dtype={0: np.uint32, 1: np.uint32})
        
        i = t[0].values - 1
        j = t[1].values - 1
        data = t[2].values

        assert data.size == nnz

        X = sparse.coo_matrix((data, (i,j)), shape=[m, n]).T.todense()

        matrix = cls(X=X, genes=genes, cells=cells)
        _LOGGER.info('Read matrix with %d genes and %d cells from "%s".',
                     matrix.shape[0], matrix.shape[1], file_path)

        return matrix

    @classmethod
    def read_10xgenomics(cls, tarball_fpath: str, prefix: str,
                         use_ensembl_ids: bool = False):
        """Read a 10X genomics compressed tarball containing expression data.
        
        Note: common prefix patterns:
        - "filtered_gene_bc_matrices/[annotations]/"
        - "filtered_matrices_mex/[annotations]/"

        TODO: docstring"""

        _LOGGER.info('Reading file: %s', tarball_fpath)

        with tarfile.open(os.path.expanduser(tarball_fpath), mode='r:gz') as tf:
            ti = tf.getmember('%smatrix.mtx' % prefix)
            with tf.extractfile(ti) as fh:
                mtx = scipy.io.mmread(fh)

            ti = tf.getmember('%sgenes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                i = 1
                if use_ensembl_ids:
                    i = 0
                gene_names = \
                        [row[i] for row in csv.reader(wrapper, delimiter='\t')]

            ti = tf.getmember('%sbarcodes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                barcodes = \
                        [row[0] for row in csv.reader(wrapper, delimiter='\t')]


            assert mtx.shape[0] == len(gene_names)
            assert mtx.shape[1] == len(barcodes)
        
        _LOGGER.info('Matrix dimensions: %s', str(mtx.shape))
        X = mtx.todense()
        matrix = cls(X=X, genes=gene_names, cells=barcodes)
        
        return matrix
