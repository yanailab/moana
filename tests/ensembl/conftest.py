# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Fixtures for `ensembl.annotations` tests."""

from pkg_resources import resource_filename

import pytest

@pytest.fixture(scope='session')
def my_gene_annotation_file():
    return resource_filename(
            'moana', 'data/test/human_ensembl88_1000.gtf.gz')
