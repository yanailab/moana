# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

import pytest
import numpy as np

from moana.core import ExpMatrix


@pytest.fixture
def my_gene_names():
    return ['d', 'c', 'b', 'a']


@pytest.fixture
def my_gene_ids():
    ensembl_ids = ['ENSG0000001', 'ENSG0000002', 'ENSG0000003', 'ENSG0000004']
    return ensembl_ids


@pytest.fixture
def my_unknown_gene_name():
    return 'X'


@pytest.fixture
def my_cells():
    return ['c1', 'c2', 'c3']


@pytest.fixture
def my_x():
    a = np.arange(4, dtype=np.float64)
    return a

@pytest.fixture
def my_cell_x():
    a = np.arange(3, 6, dtype=np.float64)
    return a

@pytest.fixture
def my_X(my_x):
    X = []
    for i in range(0, -3, -1):
        X.append(np.roll(my_x,i))
    X = np.float64(X).T
    return X


@pytest.fixture
def my_matrix(my_gene_names, my_cells, my_X):
    #genes = ['a', 'b', 'c', 'd']
    #samples = ['s1', 's2', 's3']
    # X = np.arange(12, dtype=np.float64).reshape(4, 3)
    matrix = ExpMatrix(genes=my_gene_names, cells=my_cells, X=my_X)
    return matrix