# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the `ExpMatrix` class."""

import pytest
import numpy as np

from moana.core import ExpMatrix, ExpVector


def test_basic(my_matrix, my_gene_names, my_cells, my_X):
    assert isinstance(my_matrix, ExpMatrix)
    assert isinstance(repr(my_matrix), str)
    assert isinstance(str(my_matrix), str)
    assert isinstance(my_matrix.hash, str)

    assert np.array_equal(my_matrix.genes.values, my_gene_names)
    assert np.array_equal(my_matrix.cells.values, my_cells)
    assert np.array_equal(my_matrix.X, my_X)


def test_slice(my_matrix):
    vec = my_matrix.iloc[:, 0]
    assert isinstance(vec, ExpVector)


def test_sort(my_matrix):
    other = my_matrix.copy()
    sorted = my_matrix.sort_genes()
    assert sorted != my_matrix
    assert my_matrix == other
    sorted = my_matrix.sort_genes(ascending=False)
    assert sorted == my_matrix
    assert sorted is not my_matrix
    sorted = my_matrix.sort_cells(ascending=False)
    assert sorted is not my_matrix
    assert sorted != my_matrix


def test_transformation(my_matrix):
    other = my_matrix.copy()
    other.center_genes(inplace=True)
    assert np.allclose(other.mean(axis=1), 0.0)
    other = my_matrix.copy()
    other.standardize_genes(inplace=True)
    assert np.allclose(other.std(axis=1, ddof=1), 1.0)


def test_indices(my_matrix):
    assert my_matrix.genes.name == 'Genes'
    assert my_matrix.cells.name == 'Cells'


def test_copy(my_matrix):
    other = my_matrix.copy()
    assert other is not my_matrix
    assert other == my_matrix


def test_tsv(tmpdir, my_matrix):
    output_file = tmpdir.join('expression_matrix.tsv').strpath
    my_matrix.write_tsv(output_file)
    # data = open(str(path), mode='rb').read()
    # h = hashlib.md5(data).hexdigest()
    # assert h == 'd34bf3d376eb613e4fea894f7c9d601f'
    other = ExpMatrix.read_tsv(output_file)
    assert other is not my_matrix
    assert other == my_matrix


def test_sparse(tmpdir, my_matrix):
    """Test reading/writing of sparse text format."""
    output_file = tmpdir.join('expression_matrix.mtx').strpath
    my_matrix.write_sparse(output_file)
    other = ExpMatrix.read_sparse(output_file)
    assert other is not my_matrix
    assert other == my_matrix
