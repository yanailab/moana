# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Miscellaneous functions for preprocessing scRNA-Seq data."""

import numpy as np

from ..core import ExpMatrix


def median_normalize(matrix: ExpMatrix) -> ExpMatrix:
    """Performs median-normalization."""
    num_transcripts = matrix.sum(axis=0)
    norm_matrix = (np.median(num_transcripts) / num_transcripts) * matrix
    return norm_matrix


def freeman_tukey_transform(matrix: ExpMatrix) -> ExpMatrix:
    """Applies the Freeman-Tukey transformation."""
    tmatrix = np.sqrt(matrix) + np.sqrt(matrix + 1)
    return tmatrix
