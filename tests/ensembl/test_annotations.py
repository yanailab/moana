# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the `ensembl.annotations` module."""

import hashlib

from pandas.util import hash_pandas_object

from moana import ensembl


def test_get_genes(my_gene_annotation_file):
    """Test the extraction of genes from gene annotation (GTF) files."""

    # test protein-coding genes
    genes = ensembl.get_protein_coding_genes(my_gene_annotation_file)
    assert len(genes) == 4
    h = hashlib.md5(hash_pandas_object(genes).values.tobytes()).hexdigest()
    assert h == 'a72941001a18af44ffdd555c4fe30bd8'
