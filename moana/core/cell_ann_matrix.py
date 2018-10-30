# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `CellAnnMatrix` class."""

import logging
import importlib
from typing import Iterable

from . import Matrix

cell_ann_vector = importlib.import_module('.cell_ann_vector',
                                          package='moana.core')
# - "from . import exp_vector" fails due to cyclical imports

_LOGGER = logging.getLogger(__name__)


class CellAnnMatrix(Matrix):
    """A cell annotation matrix.

    This class inherits from `moana.core.Matrix`.

    Parameters
    ----------

    Keyword-only Parameters
    -----------------------
    cells : list or tuple of str
        See :attr:`cells` attribute.
    cell_label : str
        See :attr:`cell_label` attribute.

    Additional Parameters
    ---------------------
    All `moana.core.Matrix` parameters.

    Attributes
    ----------
    cells : tuple of str
        The names of the cells (rows) in the matrix.
    cell_label : tuple of str
        Alias for `moana.core.Matrix.index`.

    Attributes
    ----------
    All `moana.core.Matrix` attributes.
    """
    def __init__(self, *args,
                 cells: Iterable[str] = None, cell_label: str = None,
                 **kwargs) -> None:
        
        if cells is not None:
            if 'index' in kwargs:
                raise ValueError(
                    'Cannot specify both "cells" and "index" arguments.')
            kwargs['index'] = cells

        # call base class constructor
        Matrix.__init__(self, *args, **kwargs)

        # set index name (default: "Genes")
        if cell_label is not None:
            self.index.name = cell_label
        elif self.index.name is None:
            self.index.name = 'Cells'


    # def __eq__(self, other):
    #     if self is other:
    #         return True
    #     elif type(self) is type(other):
    #         return (self.index.equals(other.index) and \
    #                 self.columns.equals(other.columns) and \
    #                 self.equals(other))
    #     else:
    #         return Matrix.__eq__(self, other)

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    # def __repr__(self):
    #     return '<%s instance (p=%d, n=%d, hash="%s">' \
    #            % (self.__class__.__name__, self.p, self.n, self.hash)

    #def __str__(self):
    #    return '<%s object with p=%d genes and n=%d cells>' \
    #           % (self.__class__.__name__, self.p, self.n)

    @property
    def _constructor(self):
        return CellAnnMatrix
    
    @property
    def _constructor_sliced(self):
        return cell_ann_vector.CellAnnVector

    @property
    def cells(self):
        """Alias for `Vector.index`."""
        # return tuple(str(s) for s in self.columns)
        return self.index

    @cells.setter
    def cells(self, cells):
        self.index = cells


    @classmethod
    def read_tsv(cls, file_path: str,
                 encoding: str = 'UTF-8', sep: str = '\t', **kwargs):
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

        matrix = super(CellAnnMatrix, cls).read_tsv(
            file_path, encoding, sep, **kwargs)

        if matrix.index.name is None:
            matrix.index.name = 'Cells'
        
        return matrix


    # def write_tsv(self, file_path: str, encoding: str = 'UTF-8',
    #               sep: str = '\t'):
    #     """Write expression matrix to a tab-delimited text file.

    #     Parameters
    #     ----------
    #     file_path: str
    #         The path of the output file.
    #     encoding: str, optional
    #         The file encoding. ("UTF-8")

    #     Returns
    #     -------
    #     None
    #     """
    #     Matrix.write_tsv(self, file_path, encoding, sep, float_format='%.5f')

        #_LOGGER.info('Wrote %d x %d expression matrix to "%s".',
        #            self.p, self.n, file_path)
