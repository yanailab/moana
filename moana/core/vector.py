# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Module containing the `Vector` class."""

import logging
import importlib
import hashlib
from typing import Iterable

import pandas as pd
import numpy as np

matrix = importlib.import_module('.matrix', package='moana.core')
# "from . import matrix" does not work, due to cyclical imports

_logger = logging.getLogger(__name__)


class Vector(pd.Series):
    """A vector.

    This class inherits from `pandas.Series`.

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
    label : str
        Alias for :attr:`pandas.Series.name`.
    index_label : str
        Alias for :attr:`pandas.Series.index.name`.

    Additional attributes
    ---------------------
    All `pandas.Series` attributes.
    """
    def __init__(self, *args,
                 label: str = None,
                 index_label: str = None,
                 **kwargs) -> None:
        
        if label is not None:
            if 'name' in kwargs:
                raise ValueError(
                    'Cannot specify both "label" and "name" arguments.')
            kwargs['name'] = label

        # call base class constructor
        pd.Series.__init__(self, *args, **kwargs)
        
        if index_label is not None:
            # set index name
            self.index.name = index_label

    #def __eq__(self, other):
    #        return pd.Series.__eq__(self, other)

    #def __ne__(self, other):
    #    return ~self.__eq__(other)

    def __repr__(self):
        return '<%s instance (label="%s", size=%d, hash="%s">' \
               % (self.__class__.__name__, self._label_str,
                  self.size, self.hash)

    # rely on str function from pd.Series
    #def __str__(self):
    #    if self.label is not None:
    #        label_str = self._label_str
    #    else:
    #        label_str = '(unlabeled)'
    #    return '<%s %s with p=%d genes>'  \
    #           % (self.__class__.__name__, label_str, self.p)

    @property
    def _label_str(self):
        return str(self.label) if self.label is not None else ''

    @property
    def _constructor(self):
        return Vector

    @property
    def _constructor_expanddim(self):
        return matrix.Matrix

    @property
    def hash(self):
        # warning: involves copying all the data
        index_str = ','.join([str(l) for l in self.index])
        data_str = ';'.join([self._label_str, index_str]) + ';'
        data = data_str.encode('UTF-8') + self.data.tobytes()
        return str(hashlib.md5(data).hexdigest())

    @property
    def label(self):
        """Alias for `Series.name`."""
        return self.name

    @label.setter
    def label(self, label):
        self.name = label

    @property
    def index_label(self):
        """Alias for `Series.index.name`."""
        return self.index.name

    @index_label.setter
    def index_label(self, index_label):
        self.index.name = index_label

    @classmethod
    def read_tsv(cls, filepath_or_buffer: str,
                 sep: str = '\t',
                 encoding='UTF-8',
                 **kwargs):
        """Read vector from a tab-delimited text file.

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
        `Vector`
            The expression vector.
        """
        # "squeeze = True" ensures that a pd.read_tsv returns a series
        # as long as there is only one column

        squeeze = kwargs.pop('squeeze', True)
        index_col = kwargs.pop('index_col', 0)
        header = kwargs.pop('header', 0)

        e = cls(pd.read_csv(filepath_or_buffer, sep=sep,
                            index_col=index_col, header=header,
                            encoding=encoding, squeeze=squeeze, **kwargs))

        return e


    def write_tsv(self, path: str, sep: str = '\t',
                  encoding: str = 'UTF-8', **kwargs) -> None:
        """Write vector to a tab-delimited text file.

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
        mode = kwargs.pop('mode', 'w')
        header = kwargs.pop('header', True)

        self.to_csv(
            path, sep=sep, mode=mode,
            encoding=encoding, header=header,
            **kwargs
        )

        _logger.info('Wrote vector "%s" with %d entries to "%s".',
                    self.name, self.size, path)
