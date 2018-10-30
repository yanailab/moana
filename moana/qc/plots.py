# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions for plotting QC statistics."""

import numpy as np
import plotly.graph_objs as go

from . import get_ribosomal_genes, get_mitochondrial_genes
from ..core import ExpMatrix

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

def plot_transcriptome_components(
        matrix: ExpMatrix, species: str = 'human',
        num_cells: int = 1000, seed: int = 0, opacity: int = 0.7):
    """Plot fractions of ribosomal and mitochondrial transcripts per cell."""

    np.random.seed(seed)
    if num_cells < matrix.n:
        sel = np.random.choice(matrix.n, size=num_cells, replace=False)
        matrix = matrix.iloc[:, sel]

    # layout:
    #   trace1 (ribo bar), trace2 (ribo scatter)
    #   trace3 (mito bar), trace4 (mito scatter)
    #
    # => make sure Y axes (t1/t2 and t3/t4) are shared
    def generate_scatter_trace(matrix, sel_genes, color, opacity):
        """Generate scatterplot trace."""

        num_transcripts = matrix.sum(axis=0)
        sel_sum = matrix.loc[matrix.genes & sel_genes].sum(axis=0)
        sel_frac = sel_sum / num_transcripts

        trace = go.Scatter(
            x=num_transcripts,
            y=100*sel_frac,
            text=matrix.cells,
            mode='markers',
            marker=dict(
                opacity=opacity,
                color=color,
            ),
        )

        return trace

    def generate_hist_trace(matrix, sel_genes, color, opacity):
        """Generate histogram trace."""
        
        sel_sum = matrix.loc[matrix.genes & sel_genes].sum(axis=0)
        sel_frac = sel_sum / matrix.sum(axis=0)
        
        fill_color = 'rgba(%s, %f)' % (color[4:-1], opacity)
        edge_color = color

        trace = go.Histogram(
            y=100*sel_frac,
            autobiny=False,
            ybins=dict(
                start=0,
                end=100.1,
                size=5.0001,
            ),
            marker=dict(
                color=fill_color,
                line=dict(
                    width=1.0,
                    color=edge_color,
                ),
            ),
            histnorm='percent',
        )
        return trace

    ribo_genes = get_ribosomal_genes(species)
    mito_genes = get_mitochondrial_genes(species)

    trace1 = generate_hist_trace(matrix, ribo_genes, DEFAULT_PLOTLY_COLORS[0], opacity)
    trace2 = generate_scatter_trace(matrix, ribo_genes, DEFAULT_PLOTLY_COLORS[0], opacity)

    trace3 = generate_hist_trace(matrix, mito_genes, DEFAULT_PLOTLY_COLORS[1], opacity)
    trace4 = generate_scatter_trace(matrix, mito_genes, DEFAULT_PLOTLY_COLORS[1], opacity)

    trace1.xaxis = 'x'
    trace1.yaxis = 'y'
    trace2.xaxis = 'x2'
    trace2.yaxis = 'y'

    trace3.xaxis = 'x'
    trace3.yaxis = 'y2'
    trace4.xaxis = 'x2'
    trace4.yaxis = 'y2'

    data = [trace1, trace2, trace3, trace4]

    left = 0.12
    bottom = 0.10

    axis_font_size = 28

    num_transcripts = matrix.sum(axis=0)
    max_transcripts = num_transcripts.max()
    max_x = max_transcripts * 1.05

    ygap = 0.08
    ysize = (1.0 - bottom - ygap) / 2

    xgap = 0.08
    xsize = (1.0 - left - xgap) / 2

    layout = go.Layout(
        margin=dict(b=0, l=0, t=100),
        width=1200,
        height=1000,
        font=dict(size=32, family='serif'),
        grid=dict(
            subplots=[['xy', 'x2y'], ['xy2', 'x2y2']]
        ),
        xaxis=dict(
            titlefont=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            domain=[left, left + xsize],
            range=[0, 101],
            ticklen=2,
            tickcolor='white',
            title='Fraction of cells (%)',
        ),
        yaxis=dict(
            titlefont=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            domain=[bottom + ygap + ysize, 1.0],
            ticklen=3,
            tickcolor='white',
            range=[0, 101],
            zeroline=True,
            showgrid=True,
            title='Fraction of ribosomal<br> transcripts (%)',
        ),
        xaxis2=dict(
            titlefont=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            domain=[left + xsize + xgap, 1],
            range=[0, max_x],
            ticklen=2,
            tickcolor='white',
            title='Total # of transcripts',
        ),
        yaxis2=dict(
            titlefont=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            domain=[bottom, bottom + ysize],
            range=[0, 100],
            ticklen=3,
            tickcolor='white',
            showgrid=True,
            zeroline=True,
            title='Fraction of mitochondrial<br> transcripts (%)',
        ),
        annotations=[],
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    return fig
