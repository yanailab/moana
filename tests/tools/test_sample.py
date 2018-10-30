# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the sampling functions in the tools module."""

import pytest

from moana import preprocess

NUM_CELLS = 20


def test_sample_cells_flat(my_cells):
    num_cells = NUM_CELLS
    sample = preprocess.sample_cells(my_cells, num_cells=num_cells)
    assert sample.size == num_cells
    #print(sample)
    #print(my_clusters.loc[sample].value_counts())

def test_sample_cells_clusters(my_cells, my_clusters):
    num_cells = NUM_CELLS
    sample = preprocess.sample_cells(my_cells, num_cells=num_cells, clusters=my_clusters)
    assert sample.size == num_cells
    #print(sample)
    #print(my_clusters.loc[sample].value_counts())

def test_bootstrap_cells_flat(my_cells):
    num_cells = NUM_CELLS
    sample = preprocess.bootstrap_cells(my_cells, num_cells=num_cells)
    assert sample.size == num_cells

def test_bootstrap_cells_clusters(my_cells, my_clusters):
    num_cells = NUM_CELLS
    sample = preprocess.bootstrap_cells(my_cells, num_cells=num_cells, clusters=my_clusters)
    assert sample.size == num_cells
