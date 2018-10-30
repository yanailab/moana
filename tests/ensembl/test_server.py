# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Tests for the ensembl.server module."""

from moana import ensembl

def test_release():
    release = ensembl.get_latest_release()
    #print('Latest ensembl release:', release)
    assert isinstance(release, int)


def test_annotation_urls():
    urls = ensembl.get_annotation_urls_and_checksums('homo_sapiens')
    #print('Urls:', urls)
    assert isinstance(urls, dict)
