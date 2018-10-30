# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

from pkg_resources import resource_filename

import pytest
import pandas as pd

from moana.core import ExpMatrix


@pytest.fixture(scope='session')
def my_expression_file():
    return resource_filename(
            'moana', 'data/test/test_expression.tsv')

@pytest.fixture(scope='session')
def my_matrix(my_expression_file):
    return ExpMatrix.read_tsv(my_expression_file)

@pytest.fixture(scope='session')
def my_cells(my_matrix):
    return my_matrix.cells    

@pytest.fixture(scope='session')
def my_clusters(my_matrix):
    cells = my_matrix.cells
    labels = my_matrix.cells.str.split('_').map(lambda x:x[0])
    clusters = pd.Series(index=cells, data=labels)
    return clusters
