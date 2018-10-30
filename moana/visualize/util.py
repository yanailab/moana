import logging
from typing import Tuple, Dict
from collections import OrderedDict

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from ..core import ExpMatrix, ExpVector, CellAnnVector
from .. import tools

_LOGGER = logging.getLogger(__name__)

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


COLORSCALES = {
    2: {
        'Viridis': ['#440154', '#fde725'],
    },
    4: {
        'Viridis': ['#440154', '#2c728e', '#3fbc73', '#fde725'],
    },
}

def transform_clustering(clusters):
    vc = pd.Series(clusters).value_counts()
    d = OrderedDict.fromkeys(vc.index)
    for k in vc.index:
        d[k] = (clusters == k).nonzero()[0]
    return d


def prepare_cell_clusters(
        matrix: ExpMatrix, cell_clusters: CellAnnVector) -> CellAnnVector:
    """Prepare cell clusters for use in plotting functions.
    
    """
    # if no clusters are given, create a dummy cluster comprising all cells
    if cell_clusters is None:
        cell_clusters = CellAnnVector(cells=matrix.cells,
                                      data=['all'] * matrix.n)

    # add sample sizes to cluster names, if requested
    #if show_n:
    #    vc  = cell_clusters.value_counts()
    #    cell_clusters = cell_clusters.map(lambda x: '%s (n=%d)' % (x, vc[x]))
    
    return cell_clusters


def plot_cells(
        matrix: ExpMatrix, profile: ExpVector = None,
        cell_labels: CellAnnVector = None,
        cluster_order='frequency', cluster_colors=None,
        cluster_labels: Dict[str, str] = None,
        dims: Tuple[int, int] = None,
        title: str = None,
        marker_symbol: str = 'circle',
        jitter=0.00, seed=0,
        colorscale='RdBu', padding=0.05,
        show_cells=None, emin=None, emax=None, opacity=0.7, marker_size=10,
        show_n: bool = True,
        colorbar_label='# transcripts', colorbar_length=0.5,
        width=900, height=650, fixed_aspect_ratio=False,
        margin_left=50, margin_top=65, margin_bottom=50, margin_right=50,
        flip_x=False, flip_y=False,
        showlegend=None, legend_font_size=None,
        legend_x=1.0, legend_y=0.98,
        legend_xanchor='left', legend_yanchor='top',
        font_size=28, font_family='serif', borderwidth=1.0):

    if dims is None:
        dims = (0, 1)

    if cluster_colors is None:
        cluster_colors = {}

    if cluster_labels is None:
        cluster_labels = {}

    dim1, dim2 = dims

    cell_labels = prepare_cell_clusters(matrix, cell_labels)

    # generate jitter
    np.random.seed(seed)
    dx = np.random.rand(matrix.shape[1]) - 0.5
    dy = np.random.rand(matrix.shape[1]) - 0.5
    
    if show_cells is not None and show_cells < matrix.n:
        sel_cells = tools.downsample_cells(matrix.cells, show_cells, 
                                           cell_labels=cell_labels, seed=seed)
        matrix = matrix.loc[:, sel_cells]
        cell_labels = cell_labels.loc[sel_cells]
        if profile is not None:
            profile = profile.loc[sel_cells]

    xmn = matrix.iloc[dim1, :].min()
    xmx = matrix.iloc[dim1, :].max()
    ymn = matrix.iloc[dim2, :].min()
    ymx = matrix.iloc[dim2, :].max()

    ptp_x = xmx - xmn
    ptp_y = ymx - ymn
    
    #if plot_type.lower() == 'mds':
    if fixed_aspect_ratio:
        # TODO: make sure data is centered
        ptp_max = max(ymx-ymn, xmx-xmn)
        #xmn = ptp_max 
        xmn = xmn - (ptp_max - ptp_x)*0.5
        xmx = xmx + (ptp_max - ptp_x)*0.5
        
        ymn = ymn - (ptp_max - ptp_y)*0.5
        ymx = ymx + (ptp_max - ptp_y)*0.5

        range_x = [xmn-ptp_max*padding, xmx+ptp_max*padding]
        range_y = [ymn-ptp_max*padding, ymx+ptp_max*padding]

    else:
        range_x = [xmn-ptp_x*padding, xmx+ptp_x*padding]
        range_y = [ymn-ptp_y*padding, ymx+ptp_y*padding]

    dx = dx*ptp_x*jitter
    dy = dy*ptp_y*jitter
    
    #z = (profile - profile.mean()) / profile.std(ddof=1)
    #print(z.min(), z.max())
    #z = np.log10(profile)

    xlabel = matrix.genes[dim1]
    ylabel = matrix.genes[dim2]

    labels_old = None
    try:
        labels_old = profile.index
    except AttributeError:
        pass

    if profile is not None:
        showscale = True
    else:
        showscale = False

    data = []
    
    if profile is not None:
        cell_labels[:] = 'all'

    vc = cell_labels.value_counts()

    # determine cluster ordering
    if cluster_order == 'frequency':
        ordered_labels = vc.index.tolist()
    elif cluster_order == 'alphabetical':
        ordered_labels = sorted(vc.index.tolist())
    else:
        # assume cluster_order is a list of labels
        ordered_labels = cluster_order[:]

    for i, label in enumerate(ordered_labels):
        #count = vc.loc[label]
        if label not in vc.index:
            _LOGGER.warning('No cells with label "%s".', label)
            continue
        sel = (cell_labels == label).nonzero()[0]


        x = matrix.iloc[dim1, sel] + dx[sel]
        y = matrix.iloc[dim2, sel] + dy[sel]
    
        try:
            name = cluster_labels[label]
        except KeyError:
            name = label

        if show_n:
            name = str(name) + ' (<i>n</i>=%d)' % vc.loc[label]
            #name = name + ' ($n=%d$)' % vc.loc[label]

        text = None
        if profile is not None:
            color = profile.iloc[sel]
            if labels_old is not None:
                text = labels_old[sel]
        else:
            try:
                color = cluster_colors[label]
            except KeyError:
                color = DEFAULT_PLOTLY_COLORS[i]
            text = matrix.cells[sel]

        trace = go.Scatter(
            x=x,
            y=y,
            text=text,
            mode='markers',
            name=name,
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=color,
                colorscale=colorscale,
                cmin=emin,
                cmax=emax,
                opacity=opacity,
                showscale=showscale,
                colorbar=dict(
                    len=colorbar_length,
                    title=colorbar_label,
                    titleside='right',
                    thickness=20,
                    ticklen=5,
                )
            ),
        )

        data.append(trace)

    if flip_x:
        range_x = range_x[::-1]
    
    if flip_y:
        range_y = range_y[::-1]

    layout = go.Layout(
        margin=dict(l=margin_left, r=margin_right,
                    b=margin_bottom, t=margin_top),
        width=width,
        height=height,
        font=dict(size=font_size, family=font_family),
        title=title,
        xaxis=dict(
            title=xlabel,
            zeroline=False,
            range=range_x,
            showticklabels=False,
            showline=True,
            showgrid=False),
        yaxis=dict(
            title=ylabel,
            zeroline=False,
            range=range_y,
            showticklabels=False,
            showline=True,
            showgrid=False),
        #font=dict('')
        showlegend=showlegend,
        legend=dict(
            borderwidth=borderwidth,
            x=legend_x,
            xanchor=legend_xanchor,
            y=legend_y,
            yanchor=legend_yanchor,
            font=dict(
                size=legend_font_size,
            ),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return fig
