# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for Moana's cell visualization methods."""

import hashlib

#from moana import analysis
from moana import visual


def test_cell_pca_plot(my_matrix):
    fig, _ = visual.cell_pca_plot(
        my_matrix, title='Test', width=1000, jitter=0, seed=0)

    h = hashlib.md5(str(fig).encode('utf-8')).hexdigest()
    print('hash:', h)
    assert h == '2f1f10e4dce17e9d749d08787c6c68de'


def test_cell_mds_plot(my_matrix):
    fig, _ = visual.cell_mds_plot(
        my_matrix, title='Test', width=1000, jitter=0, seed=0)

    h = hashlib.md5(str(fig).encode('utf-8')).hexdigest()
    print('hash:', h)
    assert h == '199f248a3af0e8b01c1b1287fde6e99a'


def test_cell_tsne_plot(my_matrix):
    fig, _ = visual.cell_mds_plot(
        my_matrix, title='Test', width=1000, jitter=0, seed=0)

    h = hashlib.md5(str(fig).encode('utf-8')).hexdigest()
    print('hash:', h)
    assert h == '199f248a3af0e8b01c1b1287fde6e99a'


#def test_plot_heatmap(my_matrix):
#
#    fig = plotting.plot_heatmap(my_matrix.iloc[:1000])
#    h = hashlib.md5(str(fig).encode('utf-8')).hexdigest()
#    print('hash:', h)
#    assert h == 'd5638c878496f92f82566526f08f9c8a'
