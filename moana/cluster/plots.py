from typing import Union

import plotly.graph_objs as go
from plotly import tools as plotly_tools
import numpy as np

from ..core import ExpMatrix, CellAnnVector
from .. import cluster
from ..visualize.util import DEFAULT_PLOTLY_COLORS


def fold_enrichment_plot(
        matrix: ExpMatrix,
        cell_labels: CellAnnVector,
        top: int = 15,
        font_size: Union[int, float] = 32,
        width=900, height=800,
        title_font_size: Union[int, float]=32,
        gap: float = 0.33,
        margin_left: int = 240, margin_right: int = 50,
        margin_bottom: int= 110, margin_top: int = 70) -> go.Figure:
    
    vc = cell_labels.value_counts()
    oe = cluster.get_overexpressed_genes(matrix, cell_labels, top=top)
    
    gene_coef = np.log2(oe.loc[vc.index[0]].iloc[:, 0])
    x = gene_coef.values
    y = ['<i>%s</i>' % gene for gene in gene_coef.index]
    
    trace2 = go.Bar(
        x=x,
        y=y,
        orientation='h',
    )

    gene_coef = np.log2(oe.loc[vc.index[1]].iloc[:, 0])
    x = gene_coef.values
    print(x)
    y = ['<i>%s</i>' % gene for gene in gene_coef.index]
    
    trace1 = go.Bar(
        x=x,
        y=y,
        orientation='h',
    )    
    
    fig = plotly_tools.make_subplots(rows=1, cols=2)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    
    fig.layout.update(dict(
        width=width,
        height=height,
        font=dict(size=font_size, family='serif'),
        margin=dict(l=margin_left, b=margin_bottom,
                    r=margin_right, t=margin_top),
        showlegend=False,
    ))

    fig.layout.xaxis1.update(dict(
         domain=[0, 0.5-gap/2], title='log<sub>2</sub>-Fold diff.',
        tickangle=45, showline=True, ticklen=5, zeroline=True))
    fig.layout.yaxis1.update(dict(
        autorange='reversed', title='Overexpr. genes', showline=False, ticklen=2,
        tickcolor='rgba(0,0,0,0)'))

    fig.layout.xaxis2.update(dict(
        domain=[0.5+gap/2, 1], title='log<sub>2</sub>-Fold diff.',tickangle=45,
        showline=True, ticklen=5, zeroline=True))
    fig.layout.yaxis2.update(dict(
        autorange='reversed', title='Overexpr. genes', showline=False, ticklen=2,
        tickcolor='rgba(0,0,0,0)'))
    
    #fig.layout.title = (
    #    '<span style="color:%s">%s</span> vs. <span style="color:%s">%s</span><br>'
    #    '<span style="font-size: smaller;">(<i>k</i>=%d @ %s transcripts / cell)</span>'
    #        % (DEFAULT_PLOTLY_COLORS[0], clf.cell_types_[0],
    #          DEFAULT_PLOTLY_COLORS[1], clf.cell_types_[1],
    #          clf.k,
    #          "{:,}".format(int("%.0f" % (clf.transcript_count_ / clf.k)))))
            
    fig.layout.title = (
        '<span style="color:%s">%s</span> <span style="font-size:smaller">(<i>n</i>=%d)</span> vs. '
        '<span style="color:%s">%s</span> <span style="font-size:smaller">(<i>n</i>=%d)</span>'
            % (DEFAULT_PLOTLY_COLORS[0], vc.index[1], vc[1],
              DEFAULT_PLOTLY_COLORS[1], vc.index[0], vc[0]))

    fig.layout.titlefont = dict(size=title_font_size)
    return fig
