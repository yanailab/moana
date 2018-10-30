# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `Matrix` class."""

import logging
import importlib
import hashlib

import pandas as pd
import numpy as np
import csv

vector = importlib.import_module('.vector', package='moana.core')
# - "from . import vector" fails due to cyclical imports

_LOGGER = logging.getLogger(__name__)


class Matrix(pd.DataFrame):
    """A matrix.

    This class inherits from `pandas.DataFrame`.

    Parameters
    ----------
    All `pandas.DataFrame` parameters.

    Attributes
    ----------
    All `pandas.DataFrame` attributes.
    """
    def __init__(self, *args, **kwargs) -> None:
        # call base class constructor
        pd.DataFrame.__init__(self, *args, **kwargs)

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) is type(other):
            return self.index.equals(other.index) and \
                   self.columns.equals(other.columns) and \
                   self.equals(other)
        else:
            return pd.DataFrame.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '<%s instance (p=%d, n=%d, hash="%s">' \
               % (self.__class__.__name__, self.p, self.n, self.hash)

    # rely on pd.DataFrame for "str"
    #def __str__(self):
    #    return '<%s object with p=%d genes and n=%d cells>' \
    #           % (self.__class__.__name__, self.p, self.n)

    @property
    def hash(self):
        # warning: involves copying all the data
        index_str = ','.join(str(s) for s in self.index)
        col_str = ','.join(str(s) for s in self.columns)
        data_str = ';'.join([index_str, col_str]) + ';'
        data = data_str.encode('UTF-8') + self.X.tobytes()
        return str(hashlib.md5(data).hexdigest())

    @property
    def _constructor(self):
        return Matrix
    
    @property
    def _constructor_sliced(self):
        return vector.Vector


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
        # use pd.read_csv to parse the tsv file into a DataFrame
        matrix = cls(pd.read_csv(file_path, sep=sep, index_col=0, header=0,
                                 encoding=encoding, **kwargs))

        # parse index column separately
        # (this seems to be the only way we can prevent pandas from converting
        #  "nan" or "NaN" to floats in the index)['1_cell_306.120', '1_cell_086.024', '1_cell_168.103']
        #ind = pd.read_csv(file_path, sep=sep, usecols=[0, ], header=0,
        #                  encoding=encoding, na_filter=False)
        ind = pd.read_csv(file_path, sep=sep, usecols=[0, ], header=None,
                          skiprows=1, encoding=encoding, na_filter=False)

        matrix.index = ind.iloc[:, 0]

        return matrix


    def write_tsv(self, file_path: str, encoding: str = 'UTF-8',
                  sep: str = '\t', **kwargs):
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
        #if six.PY2:
        #    sep = sep.encode('UTF-8')

        self.to_csv(
            file_path, sep=sep, mode='w',
            encoding=encoding, quoting=csv.QUOTE_NONE, **kwargs)

        _LOGGER.info('Wrote %d x %d matrix to "%s".',
                    self.shape[0], self.shape[1], file_path)
