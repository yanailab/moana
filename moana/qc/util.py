# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Utility functions for filtering scRNA-Seq data."""

from typing import List

from pkg_resources import resource_filename


def get_mitochondrial_genes(species: str = 'human') -> List[str]:
    """Get a list of all mitochondrial genes for a given species.
    
    "Mitochondrial genes" are defined here as all genes on the mitochondrial
    chromosome.
    """
    path = resource_filename('moana',
                             'data/gene_lists/mitochondrial_%s.txt' % species)
    with open(path) as fh:
        return fh.read().split('\n')


def get_mt_ribosomal_genes(species: str = 'human') -> List[str]:
    """Get a list of all mitochondrial ribosomal genes for a given species.
    
    "Mitochondrial ribosomal genes" are defined here as all protein-coding
    genes whose protein products are a structural component of the small or
    large submit of mitochondrial ribosomes.
    """
    path = resource_filename('moana',
                             'data/gene_lists/mt_ribosomal_%s.txt' % species)
    with open(path) as fh:
        return fh.read().split('\n')
    

def get_ribosomal_genes(species: str = 'human') -> List[str]:
    """Get a list of all ribosomal genes for a given species.
    
    "Ribosomal genes" are defined here as all protein-coding genes whose
    protein products are a structural component of the small or large submit of
    cytosolic ribosomes (including fusion genes).
    """
    path = resource_filename('moana',
                             'data/gene_lists/ribosomal_%s.txt' % species)
    with open(path) as fh:
        return fh.read().split('\n')

