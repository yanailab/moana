# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the cell clustering functions in the `cluster` module."""

import pytest

import pandas as pd

#from moana.core import ExpMatrix
from moana import cluster


def test_cluster_cells_dbscan(my_matrix):

    clusters = cluster.cluster_cells_dbscan(my_matrix)

    assert isinstance(clusters, pd.Series)
