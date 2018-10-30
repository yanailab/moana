# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018 New York University
#
# This file is part of Moana.

from pkg_resources import resource_filename

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from .. import util
from ..core import ExpMatrix

_DEFAULT_COLORSCALE_FILE = resource_filename(
    'moana', 'data/RdBu_r_colormap.tsv')

def plot_heatmap(
        matrix: ExpMatrix, title=None,
        emin=None, emax=None,
        width=800, height=400,
        colorbar_label: str = 'Expression',
        colorscale=None,
        gene_aliases: dict = None,
        margin_left=100, margin_bottom=60, margin_top=60, margin_right=0,
        colorbar_size=0.4,
        xaxis_label=None, yaxis_label=None,
        xaxis_nticks=None, yaxis_nticks=None,
        xtick_angle=30,
        gene_annotations: dict = None,
        font='"Droid Serif", "Open Serif", serif',
        font_size=12, title_font_size=None,
        show_sample_labels=True, **kwargs):
    """Generate a plotly figure of the heatmap.
    
    Parameters
    ----------
    emin : int, float, or None, optional
        The expression value corresponding to the lower end of the
        colorscale. If None, determine, automatically. [None]
    emax : int, float, or None, optional
        The expression value corresponding to the upper end of the
        colorscale. If None, determine automatically. [None]
    margin_left : int, optional
        The size of the left margin (in px). [100]
    margin_right : int, optional
        The size of the right margin (in px). [0]
    margin_top : int, optional
        The size of the top margin (in px). [30]
    margin_bottom : int, optional
        The size of the bottom margin (in px). [60]
    colorbar_size : int or float, optional
        The sze of the colorbar, relative to the figure size. [0.4]
    xaxis_label : str or None, optional
        X-axis label. If None, use `ExpMatrix` default. [None]
    yaxis_label : str or None, optional
        y-axis label. If None, use `ExpMatrix` default. [None]
    xtick_angle : int or float, optional
        X-axis tick angle (in degrees). [30]
    font : str, optional
        Name of font to use. Can be multiple, separated by comma, to
        specify a prioritized list.
        [' "Droid Serif", "Open Serif", "serif"']
    font_size : int or float, optional
        Font size to use throughout the figure, in points. [12]
    title_font_size : int or float or None, optional
        Font size to use for labels on axes and the colorbar. If None,
        use `font_size` value. [None]
    show_sample_labels : bool, optional
        Whether to show the sample labels. [True]

    Returns
    -------
    `plotly.graph_objs.Figure`
        The plotly figure.
    """

    # emin and/or emax are unspecified, set to data min/max values
    if emax is None:
        emax = matrix.X.max()
    if emin is None:
        emin = matrix.X.min()

    if gene_aliases is None:
        gene_aliases = {}

    if gene_annotations is None:
        gene_annotations = {}

    if title_font_size is None:
        title_font_size = font_size

    if colorscale is None:
        colorscale = util.read_colorscale(_DEFAULT_COLORSCALE_FILE)

    colorbar = go.ColorBar(
        lenmode='fraction',
        len=colorbar_size,
        title=colorbar_label,
        titlefont=dict(
            size=title_font_size,
        ),
        titleside='right',
        xpad=0,
        ypad=0,
        outlinewidth=0,  # no border
        thickness=20,  # in pixels
        # outlinecolor = '#000000',
    )

    def fix_plotly_label_bug(labels):
        """
        This fixes a bug whereby plotly treats labels that look
        like numbers (integers or floats) as numeric instead of
        categorical, even when they are passed as strings. The fix consists
        of appending an underscore to any label that looks like a number.
        """
        # assert isinstance(labels, Iterable)
        fixed_labels = []
        for l in labels:
            try:
                float(l)
            except (ValueError, TypeError):
                fixed_labels.append(str(l))
            else:
                fixed_labels.append(str(l) + '_')
        return fixed_labels

    x = fix_plotly_label_bug(matrix.cells)

    gene_labels = matrix.genes.tolist()

    if gene_aliases:
        for i, gene in enumerate(gene_labels):
            try:
                alias = gene_aliases[gene]
            except KeyError:
                pass
            else:
                gene_labels[i] = '%s/%s' % (gene, alias)
        
    gene_labels = fix_plotly_label_bug(gene_labels)


    data = [
        go.Heatmap(
            z=matrix.X,
            x=x,
            y=gene_labels,
            zmin=emin,
            zmax=emax,
            colorscale=colorscale,
            colorbar=colorbar,
            hoverinfo='x+y+z',
            **kwargs
        ),
    ]

    xticks = 'outside'
    if not show_sample_labels:
        xticks = ''

    if xaxis_label is None:
        if matrix.cells.name is not None:
            xaxis_label = matrix.cells.name
        else:
            xaxis_label = 'Cells'
        xaxis_label = xaxis_label + ' (n = %d)' % matrix.n

    if yaxis_label is None:
        if matrix.genes.name is not None:
            yaxis_label = matrix.genes.name
        else:
            yaxis_label = 'Genes'
        yaxis_label = yaxis_label + ' (p = %d)' % matrix.p

    layout = go.Layout(
        width=width,
        height=height,
        title=title,
        titlefont=go.Font(
            size=title_font_size
        ),
        font=go.Font(
            size=font_size,
            family=font
        ),
        xaxis=go.XAxis(
            title=xaxis_label,
            titlefont=dict(size=title_font_size),
            showticklabels=show_sample_labels,
            ticks=xticks,
            nticks=xaxis_nticks,
            tickangle=xtick_angle,
            showline=True
        ),
        yaxis=go.YAxis(
            title=yaxis_label,
            titlefont=dict(size=title_font_size),
            nticks=yaxis_nticks,
            autorange='reversed',
            showline=True
        ),

        margin=go.Margin(
            l=margin_left,
            t=margin_top,
            b=margin_bottom,
            r=margin_right,
            pad=0
        ),
    )

    # add annotations

    # we need separate, but overlaying, axes to place the annotations
    layout['xaxis2'] = go.XAxis(
        overlaying = 'x',
        showline = False,
        tickfont = dict(size=0),
        autorange=False,
        range=[-0.5, matrix.n-0.5],
        ticks='',
        showticklabels=False
    )

    layout['yaxis2'] = go.YAxis(
        overlaying='y',
        showline=False,
        tickfont=dict(size=0),
        autorange=False,
        range=[matrix.p-0.5, -0.5],
        ticks='',
        showticklabels=False
    )

    # gene (row) annotations
    for ann in gene_annotations:
        i = matrix.genes.get_loc(ann.gene)
        xmn = -0.5
        xmx = matrix.n-0.5
        ymn = i-0.5
        ymx = i+0.5
        #logger.debug('Transparency is %.1f', ann.transparency)
        data.append(
            go.Scatter(
                x=[xmn, xmx, xmx, xmn, xmn],
                y=[ymn, ymn, ymx, ymx, ymn],
                mode='lines',
                hoverinfo='none',
                showlegend=False,
                line=dict(color=ann.color),
                xaxis='x2',
                yaxis='y2',
                #opacity=0.5,
                opacity=1-ann.transparency,
            )
        )
        if ann.label is not None:
            layout.annotations.append(
                go.Annotation(
                    text=ann.label,
                    x=0.01,
                    y=i-0.5,
                    #y=i+0.5,
                    xref='paper',
                    yref='y2',
                    xanchor='left',
                    yanchor='bottom',
                    showarrow=False,
                    bgcolor='white',
                    #opacity=1-ann.transparency,
                    opacity=0.8,
                    borderpad=0,
                    #textangle=30,
                    font=dict(color=ann.color)
                )
            )

    fig = go.Figure(
        data=data,
        layout=layout
    )

    return fig
