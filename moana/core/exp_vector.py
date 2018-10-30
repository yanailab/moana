# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `ExpVector` class."""

import logging
import importlib
from typing import Iterable

import pandas as pd
import numpy as np

from . import Vector

exp_matrix = importlib.import_module('.exp_matrix', package='moana.core')
# "from . import matrix" does not work, due to cyclical imports

_logger = logging.getLogger(__name__)


class ExpVector(Vector):
    """A gene expression vector.

    This class inherits from `moana.core.Vector`.

    Parameters
    ----------
    x : 1-dimensional `numpy.ndarray`
        See :attr:`x` attribute.
        
    Keyword-only Parameters
    -----------------------
    label : str
        See :attr:`label` attribute.
    index_label : str
        See :attr:`index_label` attribute.
        
    Additional Parameters
    ---------------------
    All `pandas.Series` parameters.

    Attributes
    ----------
    x : 1-dimensional `numpy.ndarray`
        Alias for :attr:`pandas.Series.data`.
    genes : `pandas.Index`
        Alias for :attr:`pandas.Series.index`. Can be used if vector represents
        the expression profile of a cell.
    cells : `pandas.Index`
        Alias for :attr:`pandas.Series.index`. Can be used if vector represents
        the expression pattern of a gene.
    label : str
        Alias for :attr:`pandas.Series.name`.
    index_label : str
        Alias for :attr:`pandas.Series.index.name`.
    """
    def __init__(self, *args,
                 x: np.ndarray = None,
                 genes: Iterable[str] = None,
                 cells: Iterable[str] = None, 
                 **kwargs) -> None:
        
        # check if user provided "x" keyword argument
        if x is not None:
            if 'data' in kwargs:
                raise ValueError(
                    'Cannot specify both "x" and "data" arguments.')
            #if x.ndim != 1:
            #    raise ValueError('Must provide a one-dimensional array.')
            kwargs['data'] = x

        index_label = None

        if genes is not None and cells is not None:
            raise ValueError(
                'Cannot specify both "genes" and "cells" arguments.')
            
        if genes is not None:
            if 'index' in kwargs:
                raise ValueError(
                    'Cannot specify both "genes" and "index" arguments.')
            kwargs['index'] = genes
            index_label = 'Genes'

        elif cells is not None:
            if 'index' in kwargs:
                raise ValueError(
                    'Cannot specify both "cells" and "index" arguments.')            
            kwargs['index'] = cells
            index_label = 'Cells'

        # call base class constructor
        Vector.__init__(self, *args, **kwargs)

        self.index_label = index_label
        
    # def __eq__(self, other):
    #     if self is other:
    #         return True
    #     elif type(self) is type(other):
    #         return (self.label == other.label and
    #                 self.index.equals(other.index) and
    #                 self.equals(other))
    #     else:
    #         return Vector.__eq__(self, other)

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    # def __repr__(self):
    #     return '<%s instance (label="%s", size=%d, hash="%s">' \
    #            % (self.__class__.__name__, self._label_str,
    #               self.size, self.hash)

    #def __str__(self):
    #    if self.label is not None:
    #        label_str = self._label_str
    #    else:
    #        label_str = '(unlabeled)'
    #    return '<%s %s with p=%d genes>'  \
    #           % (self.__class__.__name__, label_str, self.p)

    # @property
    # def _label_str(self):
    #     return str(self.label) if self.label is not None else ''

    @property
    def _constructor(self):
        return ExpVector

    @property
    def _constructor_expanddim(self):
        return exp_matrix.ExpMatrix

    @property
    def genes(self):
        """Alias for `Series.index`."""
        return self.index

    @genes.setter
    def genes(self, gene_list):
        self.index = gene_list

    @property
    def cells(self):
        """Alias for `Series.index`."""
        return self.index

    @cells.setter
    def cells(self, gene_list):
        self.index = gene_list

    @property
    def x(self):
        """Alias for `Series.values`."""
        return self.values

    @x.setter
    def x(self, x):
        self.x[:] = x


    @classmethod
    def read_tsv(cls, filepath_or_buffer: str,
                 sep: str = '\t',
                 encoding='UTF-8',
                 **kwargs):
        """Read expression vector from a tab-delimited text file.

        Parameters
        ----------
        filepath_or_buffer: str
            The path (or buffer) of the text file.
        sep : str, optional
            The separator to use. ("\t")
        encoding : str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        `ExpVector`
            The expression vector.
        """
        e = super(ExpVector, cls).read_tsv(filepath_or_buffer, sep=sep,
                            encoding=encoding, **kwargs)

        if e.index_label is None:
            e.index_label = 'Genes'

        return e


    def write_tsv(self, path: str, sep: str = '\t',
                  encoding: str = 'UTF-8', **kwargs) -> None:
        """Write expression matrix to a tab-delimited text file.

        Parameters
        ----------
        path: str
            The path of the output file.
        sep: str, optional
            The separator to use. ("\t")
        encoding: str, optional
            The file encoding. ("UTF-8")

        Returns
        -------
        None
        """

        float_format = kwargs.pop('float_format', '%.5f')

        Vector.write_tsv(self,
            path, sep=sep, mode='w',
            encoding=encoding, header=True, float_format=float_format
        )

        #logger.info('Wrote expression vector "%s" with %d entries to "%s".',
        #            self.name, self.size, path)
