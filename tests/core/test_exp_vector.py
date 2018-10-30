# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the `ExpVector` class."""

import pytest
import numpy as np

from moana.core import ExpVector, ExpMatrix


@pytest.fixture(scope='session')
def my_label():
    return 'cell'


@pytest.fixture
def my_vector(my_gene_names, my_label, my_x):
    vec = ExpVector(genes=my_gene_names, label=my_label, x=my_x)
    return vec


@pytest.fixture
def my_vector2(my_gene_names, my_x):
    vec= ExpVector(genes=my_gene_names, label=None, x=my_x)
    return vec


@pytest.fixture
def my_cell_vector(my_cells, my_cell_x):
    vec = ExpVector(cells=my_cells, x=my_cell_x)
    return vec

def test_init(my_vector, my_vector2, my_cell_vector, my_gene_names, my_x):
    for vec in [my_vector, my_vector2]:
        assert isinstance(vec, ExpVector)
        assert isinstance(repr(vec), str)
        assert isinstance(str(vec), str)
        assert isinstance(vec.hash, str)
        # assert vec.p == len(my_gene_names)
        assert np.array_equal(vec.x, my_x)
        assert np.array_equal(vec.genes, my_gene_names)
        assert vec.genes.name == 'Genes'
    assert my_cell_vector.cells.name == 'Cells'
    assert my_vector is not my_vector2
    assert my_vector.label != my_vector2.label


def test_expanddim(my_vector):
    matrix = my_vector.to_frame()
    assert isinstance(matrix, ExpMatrix)
    assert matrix.genes.name == 'Genes'

def test_tsv(tmpdir, my_vector):
    tmp_file = tmpdir.join('expression_profile.tsv').strpath
    my_vector.write_tsv(tmp_file)
    other = ExpVector.read_tsv(tmp_file)
    assert other.equals(my_vector)
    assert other.genes.name == 'Genes'

def test_copy(my_vector, my_gene_names, my_x):
    vec = my_vector.copy()
    assert vec is not my_vector
    assert vec.hash == my_vector.hash
    assert vec.equals(my_vector)
    vec.genes = my_gene_names
    vec.x = my_x
