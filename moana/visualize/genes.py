# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

"""Functions for visualizing genes using plotly."""

import logging
from typing import Union, Iterable

import plotly.graph_objs as go
import numpy as np

from ..core import ExpMatrix
# from .. import tools
from ..util import get_sel_components
from .. import preprocess as pp
from .util import plot_cells

_LOGGER = logging.getLogger(__name__)


def gene_loading_plot(
        matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        top: int = 10,
        gene_font_size: Union[int, float] = 20,
        seed: int = 0) -> go.Figure:
    """Plot loadings of genes with highest absolute loadings per PC."""

    # determine which PCs were specified 
    sel_components = get_sel_components(components)

    # perform PCA
    num_components = max(sel_components) + 1 
    tmatrix, pca_model = pp.pca(
        pp.median_normalize(matrix), num_components, seed=seed)
    
    # only keep specified PCs
    tmatrix = tmatrix.iloc[sel_components]

    # only keep the first two PCs
    #tmatrix = tmatrix.iloc[:2]
    dim_labels = ['PC %d loading' % (d+1) for d in sel_components]

    loadings = ExpMatrix(genes=matrix.genes, cells=dim_labels,
                         data=pca_model.components_.T[:, sel_components])
    #print(loadings.shape)

    #dim_labels = ['PC %d loading' % (d+1) for d in sel_components]
    #components = ExpMatrix(genes=clf.genes_, cells=dim_labels, data=clf.pca_model_.components_[sel_components, :].T)

    sel_genes = []
    for i in range(loadings.shape[1]):
        a = np.argsort(loadings.iloc[:, i])
        a = a[::-1]
        sel_genes.extend(loadings.genes[a[:top]].tolist())
    sel_genes = sorted(set(sel_genes))
    loadings = loadings.loc[sel_genes]    
    #print(loadings.shape)

    fig = plot_cells(
        loadings.T, width=800, marker_size=15,
        margin_left=140, margin_right=130, margin_bottom=100,
        fixed_aspect_ratio=True)
    mx = max(np.max(np.abs(fig.layout.xaxis.range)),
        np.max(np.abs(fig.layout.yaxis.range)))
    l = int(mx*10)/10.0
    fig.layout.xaxis.tickvals = [-l, -l/2, 0, l/2, l]
    fig.layout.yaxis.tickvals = [-l, -l/2, 0, l/2, l]
    
    fig.layout.xaxis.range = [-mx, mx]
    fig.layout.yaxis.range = [-mx, mx]
    fig.layout.xaxis.ticklen = 5
    fig.layout.yaxis.ticklen = 5
    fig.layout.xaxis.showticklabels=True
    fig.layout.xaxis.showline=False
    fig.layout.xaxis.zeroline=True
    fig.layout.yaxis.showticklabels=True
    fig.layout.yaxis.showline=False
    fig.layout.yaxis.zeroline=True
    fig.layout.xaxis.showgrid=True
    fig.layout.yaxis.showgrid=True
    
    annotations = []
    sort = loadings.iloc[:, 1].sort_values(ascending=False)
    loadings = loadings.loc[sort.index]
    factor = 0.9 / loadings.shape[0]
    #print(loadings)
    for i, gene in enumerate(loadings.genes):
        x, y = loadings.loc[gene].values[:2]
        ax = mx
        ay = mx - (2*mx*factor*i)
        annotations.append(
            dict(
                x=x,
                y=y,
                text='<i>%s</i>' % gene,
                font=dict(size=gene_font_size),
                arrowhead=0,
                arrowcolor='rgba(0,0,0,0.2)',
                ax=ax,
                ay=ay,
                axref='x',
                ayref='y',
                xanchor='left',
            )
        )
    fig.layout.annotations.extend(annotations)
    return fig
