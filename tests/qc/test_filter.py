# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the filtering functions of the `qc` module."""

import pytest

from moana import qc

def test_mito_genes():
    mito_genes = qc.get_mitochondrial_genes('human')
    assert isinstance(mito_genes, list)
    mito_genes = qc.get_mitochondrial_genes('mouse')
    assert isinstance(mito_genes, list)
    mito_genes = qc.get_mitochondrial_genes('zebrafish')
    assert isinstance(mito_genes, list)

def test_ribo_genes():
    mito_genes = qc.get_mitochondrial_genes('human')
    assert isinstance(mito_genes, list)
    mito_genes = qc.get_mitochondrial_genes('mouse')
    assert isinstance(mito_genes, list)
    mito_genes = qc.get_mitochondrial_genes('zebrafish')
    assert isinstance(mito_genes, list)

