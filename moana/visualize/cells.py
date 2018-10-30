# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Functions for visualizing cells using plotly."""

from collections import OrderedDict
from typing import Union, Iterable, Tuple, Dict, Any
import logging
import sys

from sklearn.manifold import MDS, TSNE
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix, ExpVector, CellAnnVector
from .. import cluster
from .. import tools
from ..util import get_sel_components
from .. import preprocess as pp
from .util import plot_cells, prepare_cell_clusters

_LOGGER = logging.getLogger(__name__)


def cell_pca_plot(
        matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        profile: ExpVector = None,
        cell_labels: CellAnnVector = None,
        seed: int = 0,
        **kwargs) -> go.Figure:

    # determine which PCs were specified 
    sel_components = get_sel_components(components)

    # perform PCA
    num_components = max(sel_components) + 1 
    tmatrix, _ = pp.pca(
        pp.median_normalize(matrix), num_components, seed=seed)
    
    # only keep specified PCs
    tmatrix = tmatrix.iloc[sel_components]

    # plot cells
    fig = plot_cells(tmatrix, profile=profile, cell_labels=cell_labels,
                     seed=seed, **kwargs)
    return fig


def cell_mds_plot(
        matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,        
        profile: ExpVector = None,
        cell_labels: CellAnnVector = None,
        seed: int = 0,
        num_cells: int = 500,
        init_tsne: bool = True, tsne_perplexity: float = 80,
        mds_kwargs: Dict[str, Any] = None,
        **kwargs) -> Tuple[go.Figure, ExpMatrix, go.Figure]:
    """Visualize cells using MDS."""

    if mds_kwargs is None:
        mds_kwargs = {}

    # determine seed for MDS
    mds_seed = mds_kwargs.pop('random_state', seed)

    # determine which PCs were specified
    sel_components = get_sel_components(components)

    # perform PCA
    num_components = max(sel_components) + 1
    tmatrix, _ = pp.pca(
        pp.median_normalize(matrix), num_components, seed=seed)

    # only keep specified PCs
    tmatrix = tmatrix.iloc[sel_components]

    # prepare clusters
    cell_labels = prepare_cell_clusters(tmatrix, cell_labels)

    # perform sampling, if necessary
    if num_cells is not None and num_cells < tmatrix.n:
        sel_cells = tools.downsample_cells(tmatrix.cells, num_cells,
                                           cell_labels=cell_labels, seed=seed)
        tmatrix = tmatrix.loc[:, sel_cells]
        cell_labels = cell_labels.loc[sel_cells]
        if profile is not None:
            profile = profile.loc[sel_cells]

    if init_tsne:
        # run t-SNE to obtain initialization
        _LOGGER.info('Performing t-SNE to obtain an initialization...')
        init_matrix = tools.tsne(
            tmatrix,
            perplexity=tsne_perplexity,
            seed=seed)
    else:
        init_matrix = None

    # perform MDS on sampled cells
    _LOGGER.info('Performing MDS...')
    mds_matrix = tools.mds(tmatrix, seed=mds_seed, init=init_matrix,
                           **mds_kwargs)

    # plot t-SNE initialization results
    if init_matrix is not None:
        init_fig = plot_cells(
            init_matrix, profile=profile, cell_labels=cell_labels,
            fixed_aspect_ratio=False,
            seed=seed, **kwargs)
    else:
        init_fig = None

    # plot MDS results
    mds_fig = plot_cells(
        mds_matrix, profile=profile, cell_labels=cell_labels,
        fixed_aspect_ratio=True,
        seed=seed, **kwargs)

    # return MDS plot, MDS matrix, and plot of MDS initialization
    return mds_fig, mds_matrix, init_fig


def cell_tsne_plot(
        matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        profile: ExpVector = None,
        cell_labels: CellAnnVector = None,
        dither: float = 0.0,
        perplexity: float = 30.0,
        num_cells: int = None, seed: int = 0,
        tsne_kwargs: Dict = None,
        **kwargs) -> Tuple[go.Figure, ExpMatrix]:
    """Visualize cells using t-SNE."""

    if tsne_kwargs is None:
        tsne_kwargs = {}

    # determine seed for t-SNE
    tsne_seed = tsne_kwargs.pop('random_state', seed)

    # determine which PCs were specified
    sel_components = get_sel_components(components)

    # perform PCA
    num_components = max(sel_components) + 1
    tmatrix, _ = pp.pca(
        pp.median_normalize(matrix), num_components, seed=seed)

    # only keep specified PCs
    tmatrix = tmatrix.iloc[sel_components]

    # if no clusters are given, create a dummy cluster comprising all cells
    cell_labels = prepare_cell_clusters(tmatrix, cell_labels)

    # perform sampling, if necessary
    if num_cells is not None and num_cells < tmatrix.n:
        sel_cells = tools.downsample_cells(
            tmatrix.cells, num_cells,
            cell_labels=cell_labels, seed=seed)
        tmatrix = tmatrix.loc[:, sel_cells]
        cell_labels = cell_labels.loc[sel_cells]
        if profile is not None:
            profile = profile.loc[sel_cells]

    # perform t-SNE on sampled cells
    tsne_matrix = tools.tsne(tmatrix, dither=dither,
                             perplexity=perplexity, seed=tsne_seed,
                             **tsne_kwargs)

    # plot t-SNE results
    fig = plot_cells(
        tsne_matrix, profile=profile, cell_labels=cell_labels,
        seed=seed, fixed_aspect_ratio=False, **kwargs)

    return fig, tsne_matrix
