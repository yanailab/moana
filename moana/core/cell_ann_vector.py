# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `CellAnnVector` class."""

import logging
import importlib
from typing import Iterable

import pandas as pd
import numpy as np

from . import Vector

cell_ann_matrix = importlib.import_module('.cell_ann_matrix', package='moana.core')
# "from . import matrix" does not work, due to cyclical imports

_logger = logging.getLogger(__name__)


class CellAnnVector(Vector):
    """A cell annotation vector.

    This class inherits from `moana.core.Vector`.

    Keyword-only Parameters
    -----------------------
    cells : str
        See :attr:`cells` attribute.
        
    Additional Parameters
    ---------------------
    All `moana.core.Vector` parameters.

    Attributes
    ----------
    cells : Iterable of str
        Alias for :attr:`Vector.index`.

    Additional Attributes
    ---------------------
    All `moana.core.Vector` attributes.

    """
    def __init__(self, *args,
                 cells: Iterable[str] = None, 
                 **kwargs) -> None:
        
        if cells is not None:
            if 'index' in kwargs:
                raise ValueError(
                    'Cannot specify both "cells" and "index" arguments.')            
            kwargs['index'] = cells

        # call base class constructor
        Vector.__init__(self, *args, **kwargs)

        if self.index_label is None:
            self.index_label = 'Cells'
        
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
        return CellAnnVector

    @property
    def _constructor_expanddim(self):
        return cell_ann_matrix.CellAnnMatrix

    @property
    def cells(self):
        """Alias for `Series.index`."""
        return self.index

    @cells.setter
    def cells(self, cells):
        self.index = cells

    @classmethod
    def read_tsv(cls, filepath_or_buffer: str,
                 sep: str = '\t',
                 encoding='UTF-8',
                 **kwargs):
        """Read cell vector from a tab-delimited text file.

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
        `CellVector`
            The cell vector.
        """
        e = super(CellAnnVector, cls).read_tsv(filepath_or_buffer, sep=sep,
                            encoding=encoding, **kwargs)

        if e.index_label is None:
            e.index_label = 'Cells'

        return e


    # def write_tsv(self, path: str, sep: str = '\t',
    #               encoding: str = 'UTF-8', **kwargs) -> None:
    #     """Write expression matrix to a tab-delimited text file.

    #     Parameters
    #     ----------
    #     path: str
    #         The path of the output file.
    #     sep: str, optional
    #         The separator to use. ("\t")
    #     encoding: str, optional
    #         The file encoding. ("UTF-8")

    #     Returns
    #     -------
    #     None
    #     """

    #     Vector.write_tsv(self,
    #         path, sep=sep, mode='w',
    #         encoding=encoding, header=True,
    #     )

    #     #logger.info('Wrote expression vector "%s" with %d entries to "%s".',
    #     #            self.name, self.size, path)
