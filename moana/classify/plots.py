import logging
from typing import Union, Iterable, List, Tuple, Dict

from sklearn.metrics import accuracy_score, f1_score
from plotly import tools as plotly_tools
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from ..core import ExpMatrix
from ..core import CellAnnVector
from .. import tools
from .. import visualize
from ..visualize.util import plot_cells, DEFAULT_PLOTLY_COLORS
from . import CellTypeClassifier
from .. import preprocess as pp
from ..util import get_sel_components
from .util import apply_pca

_LOGGER = logging.getLogger(__name__)


def sizeof_fmt(num, suffix=''):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1000.0:
            return "%s%s%s" % (float("%.3g" % num), unit, suffix)
        num /= 1000.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def gene_coef_plot(
        clf: CellTypeClassifier, top: int = 10,
        alt_gene_names: Dict[str, str] = None,
        cell_type_order: Union[List, Tuple] = None,
        font_size: Union[int, float] = 32,
        width=1200, height=800,
        title_font_size: Union[int, float] = 32,
        gap: float = 0.2,
        margin_left: int = 180, margin_right: int = 50,
        margin_bottom: int= 150, margin_top: int = 70) -> go.Figure:

    if alt_gene_names is None:
        alt_gene_names = {}

    gene_coef = clf.normalized_gene_coef_.iloc[:, 0]
    gene_coef = gene_coef.sort_values(ascending=False)

    fig = plotly_tools.make_subplots(rows=1, cols=2)

    if cell_type_order is None:
        #cell_type_order = clf.cell_types_.tolist()
        cell_type_order = clf.value_counts_.index.tolist()

    first_class = clf.cell_types_.tolist().index(cell_type_order[0])
    second_class = clf.cell_types_.tolist().index(cell_type_order[1])

    if second_class == 0:
        sel_coef = gene_coef[-top:][::-1]
        reverse_axis = 'reversed'
    else:
        sel_coef = gene_coef[:top]
        reverse_axis = None

    x = sel_coef.values

    #x = gene_coef[:top].values
    #y = ['<i>%s</i>' % gene for gene in gene_coef[:top].index]
    
    y = ['<i>%s</i>' % gene if gene not in alt_gene_names
            #else '<span style="font-size: 50%%;"><i>%s</i><br>(<i>%s</i>)</span>' % (gene, alt_gene_names[gene])
            #else '<span style="font-size: 50%%; line-height:0.1;"><i>%s</i><br style="line-height:0.1;" />(<i>%s</i>)</span>' % (gene, alt_gene_names[gene])
            #else '<i>%s</i><br>(<i>%s</i>)' % (gene, alt_gene_names[gene])
            else '<span style="font-size: 80%%"><i>%s</i>/<i>%s</i></span>' % (gene, alt_gene_names[gene])
            for gene in sel_coef.index]


    trace2 = go.Bar(
        x=x,
        y=y,
        orientation='h',
        marker=dict(color=DEFAULT_PLOTLY_COLORS[1])
    )

    fig.append_trace(trace2, 1, 2)
    fig.layout.xaxis2.update(dict(
        autorange=reverse_axis, domain=[0.5+gap/2, 1], title=None, tickangle=45, nticks=3,
        showline=True, ticklen=5, zeroline=True))

    if first_class == 0:
        sel_coef = gene_coef[-top:][::-1]
        reverse_axis = 'reversed'
    else:
        sel_coef = gene_coef[:top]
        reverse_axis = None

    x = sel_coef.values
    #y = ['<i>%s</i>' % gene for gene in gene_coef[-top:][::-1].index]
    
    y = ['<i>%s</i>' % gene if gene not in alt_gene_names
            else '<span style="font-size: 80%%"><i>%s</i>/<i>%s</i></span>' % (gene, alt_gene_names[gene])
            for gene in sel_coef.index]

    trace1 = go.Bar(
        x=x,
        y=y,
        orientation='h',
        marker=dict(color=DEFAULT_PLOTLY_COLORS[0])
    )    
    
    fig.append_trace(trace1, 1, 1)
    fig.layout.xaxis1.update(dict(
        autorange=reverse_axis, domain=[0, 0.5-gap/2], title=None, nticks=3,
        tickangle=45, showline=True, ticklen=5, zeroline=True))

    fig.layout.update(dict(
        width=width,
        height=height,
        font=dict(size=font_size, family='serif'),
        margin=dict(l=margin_left, b=margin_bottom,
                    r=margin_right, t=margin_top),
        showlegend=False,
    ))

    fig.layout.yaxis1.update(dict(
        autorange='reversed', title='Most informative genes', showline=False,
        ticklen=2, tickcolor='rgba(0,0,0,0)'))
    fig.layout.yaxis2.update(dict(
        autorange='reversed', showline=False,
        ticklen=2, tickcolor='rgba(0,0,0,0)'))

    fig.layout.annotations.append(
        dict(text='Normalized SVM coefficient', showarrow=False,
        xanchor='center', xref='paper', yref='paper', x=0.5, y=-0.25)
    )
    #fig.layout.xaxis1.update(dict(
    #    autorange='reversed', domain=[0, 0.5-gap/2], title='Normalized coef.',
    #    tickangle=45, showline=True, ticklen=5, zeroline=True))

    #fig.layout.xaxis2.update(dict(
    #    domain=[0.5+gap/2, 1], title='Normalized coef.', tickangle=45,
    #    showline=True, ticklen=5, zeroline=True))
    #fig.layout.yaxis2.update(dict(
    #    autorange='reversed', title='Informative genes', showline=False, ticklen=2,
    #    tickcolor='rgba(0,0,0,0)'))
    
    #fig.layout.title = (
    #    '<span style="color:%s">%s</span> vs. <span style="color:%s">%s</span><br>'
    #    '<span style="font-size: smaller;">(<i>k</i>=%d @ %s transcripts / cell)</span>'
    #        % (DEFAULT_PLOTLY_COLORS[0], clf.cell_types_[0],
    #          DEFAULT_PLOTLY_COLORS[1], clf.cell_types_[1],
    #          clf.k,
    #          "{:,}".format(int("%.0f" % (clf.transcript_count_ / clf.k)))))
            
    fig.layout.title = (
        '<span style="color:%s">%s</span> vs. '
        '<span style="color:%s">%s</span>'
            % (DEFAULT_PLOTLY_COLORS[0], cell_type_order[0],
              DEFAULT_PLOTLY_COLORS[1], cell_type_order[1]))

    fig.layout.titlefont = dict(size=title_font_size)
    return fig


def classification_plot(
        clf: CellTypeClassifier,
        matrix: ExpMatrix,
        cell_labels: CellAnnVector = None,
        perform_smoothing: bool = True,
        show_pc_var: bool = False,
        opacity: float = 0.2, num_steps: int = 200,
        showlegend: bool = False, show_acc: bool = True,
        acc_x: float = 0.65, acc_xanchor: str = 'left',
        acc_y: float = 0.95, acc_yanchor: str = 'top',
        xrange: Tuple[float, float] = None, yrange: Tuple[float, float] = None,
        title: str = None, **kwargs):

    cluster_order = kwargs.pop('cluster_order', None)

    # sel_components = get_sel_components(components)
    if cell_labels is None:
        show_acc = False

    # 2. Apply PCA (will select same genes)
    #tmatrix = apply_pca(
    #    matrix, clf.pca_model_, clf.transcript_count_,
    #    components=clf.sel_components, valid_genes=clf.genes_)
    tmatrix = clf.transform(matrix, include_var=show_pc_var,
                            perform_smoothing=perform_smoothing)

    predictions = clf.predict(matrix, perform_smoothing=perform_smoothing)
    if cell_labels is None:
        cell_labels = predictions

    if cluster_order is None:
        cluster_order = clf.value_counts_.index.tolist()

    if title is None:
        title = '<i>k</i>=%d, <i>ν</i>=%s' % (clf.k, str(clf.nu))

    fig = plot_cells(
        tmatrix, cell_labels=cell_labels, showlegend=showlegend,
        cluster_order=cluster_order, title=title, **kwargs)

    if xrange is None:
        xrange = fig.layout.xaxis.range

    if yrange is None:
        yrange = fig.layout.yaxis.range    

    fig.layout.xaxis.range = xrange
    fig.layout.yaxis.range = yrange    

    x, xstep = np.linspace(xrange[0], xrange[1], num_steps, endpoint=False, retstep=True)
    y, ystep = np.linspace(yrange[0], yrange[1], num_steps, endpoint=False, retstep=True)    

    x += (xstep*0.5)
    y += (ystep*0.5)

    XX, YY = np.meshgrid(x, y)
    grid = np.c_[XX.ravel(), YY.ravel()]
    predictions = clf.svm_model_.predict(grid)

    cluster_indices = dict([[label, i] for i, label in enumerate(cluster_order)])

    predictions = pd.Series(predictions).map(cluster_indices)

    ZZ = predictions.values.reshape(XX.shape)
    num_cell_types = len(clf.classes_)
    colorscale = [
        [i / (num_cell_types-1), 'rgba(%s,%s)'
         % (DEFAULT_PLOTLY_COLORS[i][4:-1], str(opacity))]
         for i in range(num_cell_types)]
    
    trace = go.Heatmap(
        z=ZZ,
        x=x,
        y=y,
        colorscale=colorscale,
        showscale=False,
    )

    fig.data.insert(0, trace)

    if show_acc:
        acc = accuracy_score(predictions, cell_labels)
        text = 'Acc=%.3f' % acc

        fig.layout.annotations.append(
            dict(
                x=acc_x,
                y=acc_y,
                xref='paper',
                yref='paper',
                text=text,
                showarrow=False,
                xanchor=acc_xanchor,
                yanchor=acc_yanchor,
                #arrowhead=7,
                #ax=0,
                #ay=-40
            )
        )

    fig.layout.xaxis.ticklen = 0
    fig.layout.yaxis.ticklen = 0
    fig.layout.showlegend = False

    return fig


def coherence_plot(
        df: pd.DataFrame, sel_cell_types: Iterable[str],
        display_labels: dict = None, title: str = None) -> go.Figure:
    """Plots a bar plot with mirror validation coherences."""

    if display_labels is None:
        display_labels = {}
    
    labels = [display_labels[ctype]
              if ctype in display_labels else ctype
              for ctype in sel_cell_types]
    
    pr = df.loc[sel_cell_types, 'Precision']
    rc = df.loc[sel_cell_types, 'Recall']

    f1 = 100 * 2 * (pr * rc) / (pr + rc)
    #y = df.loc[sel_cell_types, '']
    y = sel_cell_types

    frac = 0

    col = DEFAULT_PLOTLY_COLORS[0]
    trace = go.Bar(
        x=f1,
        y=labels,
        orientation='h',
        marker=dict(color='rgba(%s,0.7)' % col[4:-1]),
        #opacity=0.7,
        #marker=dict(color='rgba(100,149,237,0.7)'),
            #line=dict(color='rgba(100,149,237,1.0)', width=3.0)),
    )

    data = [trace]
    
    layout = go.Layout(
        width=800,
        height=700,
        font=dict(
            size=28,
            family='serif'),
        margin=dict(
            l=400,
            b=80,
            t=60,
        ),
        xaxis=dict(
            title='Coherence (%)',
            range=[60, 101],
            ticklen=5,
            showline=True,
            zeroline=False,
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            ticklen=5,
            tickcolor='rgba(255,255,255,0)',
            showline=True,
            autorange='reversed',
            tickfont=dict(size=24),
            domain=[0, 1-frac],
        ),
        yaxis2=dict(
            ticklen=5,
            tickcolor='rgba(255,255,255,0)',
            showline=True,
            autorange='reversed',
            tickfont=dict(size=24),
            domain=[1-frac, 1.0],
        ),
        #bargroupgap=0.10,
        bargap=0.4,
        title=title,
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig



def precision_recall_plot(
        df: pd.DataFrame, sel_cell_types: Iterable[str],
        display_labels: dict = None, title: str = None,
        show_overall=True) -> go.Figure:
    """Plots a bar plot with validation accuracies (precision/recall)."""

    if display_labels is None:
        display_labels = {}
    
    labels = [display_labels[ctype]
              if ctype in display_labels else ctype
              for ctype in sel_cell_types]
    
    precision_x = 100*df.loc[sel_cell_types, 'Precision']
    recall_x = 100*df.loc[sel_cell_types, 'Recall']
    #y = df.loc[sel_cell_types, '']
    y = sel_cell_types

    if show_overall:
        col = DEFAULT_PLOTLY_COLORS[2]
        overall_trace = go.Bar(
            x=[recall_x[0]],
            y=[labels[0]],
            orientation='h',
            yaxis='y2',
            xaxis='x',
            marker=dict(color='rgba(%s,0.7)' % col[4:-1]),
            #marker=dict(color='rgba(0,0,0,0.7)'),
            showlegend=False,
        )
        frac = 1 / len(labels)

        precision_x = precision_x[1:]
        recall_x = recall_x[1:]
        labels = labels[1:]
    else:
        frac = 0

    col = DEFAULT_PLOTLY_COLORS[0]
    precision_trace = go.Bar(
        x=precision_x,
        y=labels,
        orientation='h',
        marker=dict(color='rgba(%s,0.7)' % col[4:-1]),
        #opacity=0.7,
        #marker=dict(color='rgba(100,149,237,0.7)'),
            #line=dict(color='rgba(100,149,237,1.0)', width=3.0)),
        name='Precision',
    )
    col = DEFAULT_PLOTLY_COLORS[1]
    recall_trace = go.Bar(
        x=recall_x,
        y=labels,
        orientation='h',
        marker=dict(color='rgba(%s,0.7)' % col[4:-1]),
        #marker=dict(color='rgba(255,140,0,0.7)'),
            #line=dict(color='rgba(255,20,147,1.0)', width=3.0)),
        name='Recall',
    )

    if show_overall:
        data = [precision_trace, recall_trace, overall_trace]
    else:
        data = [precision_trace, recall_trace]
    
    layout = go.Layout(
        width=1000,
        height=700,
        font=dict(
            size=28,
            family='serif'),
        margin=dict(
            l=400,
            b=80,
            t=60,
        ),
        xaxis=dict(
            title='%',
            range=[0, 101],
            ticklen=5,
            showline=True,
            zeroline=False,
            tickfont=dict(size=24),
        ),
        yaxis=dict(
            ticklen=5,
            tickcolor='rgba(255,255,255,0)',
            showline=True,
            autorange='reversed',
            tickfont=dict(size=24),
            domain=[0, 1-frac],
        ),
        yaxis2=dict(
            ticklen=5,
            tickcolor='rgba(255,255,255,0)',
            showline=True,
            autorange='reversed',
            tickfont=dict(size=24),
            domain=[1-frac, 1.0],
        ),
        #bargroupgap=0.10,
        bargap=0.4,
        title=title,
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig


# we prefer PCA and t-SNE here
def joint_cell_plot(
        train_tmatrix: ExpMatrix,
        train_cell_clusters: CellAnnVector,
        test_tmatrix: ExpMatrix,
        test_cell_clusters: CellAnnVector,
        components: Union[int, Iterable[int]],
        num_train_cells: int = 500, num_test_cells: int = 500,
        show_n: bool = True, seed: int = 0, jitter=0.02,
        **kwargs) -> go.Figure:

    vc = test_cell_clusters.value_counts()
    cluster_indices = dict([(vc.index[i], i) for i in range(vc.size)])    
    i = vc.size
    for c in train_cell_clusters.value_counts().index:
        if  c not in cluster_indices:
            cluster_indices[c] = i
            i += 1
        
    # sample training cells
    if num_train_cells < train_tmatrix.n:
        train_cells = tools.downsample_cells(
            train_tmatrix.cells,
            cell_labels=train_cell_clusters,
            num_cells=num_train_cells, seed=seed)

        train_tmatrix = train_tmatrix.loc[:, train_cells]
        train_cell_clusters = train_cell_clusters.loc[train_cells]
        
    # sample test cells
    if num_test_cells < test_tmatrix.n:
        test_cells = tools.downsample_cells(
            test_tmatrix.cells,
            cell_labels=test_cell_clusters,
            num_cells=num_test_cells, seed=seed)

        test_tmatrix = test_tmatrix.loc[:, test_cells]
        test_cell_clusters = test_cell_clusters.loc[test_cells]
    
    fig1 = visualize.cell_pca_plot(
        train_tmatrix, components, cell_labels=train_cell_clusters,
        show_n=False, seed=seed, jitter=jitter, **kwargs)

    fig2 = visualize.cell_pca_plot(
        test_tmatrix, components, cell_labels=test_cell_clusters,
        show_n=False, seed=seed, jitter=jitter, **kwargs)

    for i, d in enumerate(fig1.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        d.marker.symbol = 'circle-open'
        d.showlegend = False

    for i, d in enumerate(fig2.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        if show_n:
            d.name = '%s (n=%d)' % (label, vc[label])
        d.showlegend = True
        fig1.data.append(d)

    return fig1


def validation_tsne_plot(
        clf: CellTypeClassifier,
        train_matrix: ExpMatrix,
        train_cell_labels: CellAnnVector,
        test_matrix: ExpMatrix,
        #test_cell_labels: CellAnnVector,
        test_marker: str = 'x',
        components: Union[int, Iterable[int]] = 20,
        perplexity: float = 30.0,
        num_train_cells: int = 500, num_test_cells: int = 500,
        show_n: bool = True, seed: int = 0, jitter=0.02,
        **kwargs) -> go.Figure:

    train_tmatrix = clf.transform(train_matrix)
    
    test_smoothed_matrix = clf.smooth(test_matrix)
    test_tmatrix = clf.transform(test_smoothed_matrix, perform_smoothing=False)
    
    test_cell_labels = clf.predict(test_smoothed_matrix, perform_smoothing=False)
    
    if len(train_tmatrix.index) != len(test_tmatrix.index):
        raise ValueError('PCA-transformed training and test matrices '
                         'must contain the same number of PCs.')
    
    vc = test_cell_labels.value_counts()
    cluster_indices = dict([(vc.index[i], i) for i in range(vc.size)])        
    i = vc.size
    for c in train_cell_labels.value_counts().index:
        if  c not in cluster_indices:
            cluster_indices[c] = i
            i += 1
        
    # sample training cells
    if num_train_cells < train_tmatrix.n:
        train_cells = tools.downsample_cells(
            train_tmatrix.cells,
            cell_labels=train_cell_labels,
            num_cells=num_train_cells, seed=seed)

        train_tmatrix = train_tmatrix.loc[:, train_cells]
        train_cell_labels = train_cell_labels.loc[train_cells]
        
    # sample test cells
    if num_test_cells < test_tmatrix.n:
        test_cells = tools.downsample_cells(
            test_tmatrix.cells,
            cell_labels=test_cell_labels,
            num_cells=num_test_cells, seed=seed)

        test_tmatrix = test_tmatrix.loc[:, test_cells]
        test_cell_labels = test_cell_labels.loc[test_cells]
        
    # perform t-SNE on joint matrix
    
    # force indices to be the same
    test_tmatrix.index = train_tmatrix.index
    
    joint_tmatrix = pd.concat([train_tmatrix, test_tmatrix], axis=1)
    tsne_matrix = tools.tsne(joint_tmatrix, perplexity=perplexity, seed=seed)
    mds_matrix = tools.mds(joint_tmatrix, init=tsne_matrix, seed=seed)

    train_tsne_matrix = tsne_matrix.iloc[:, :train_tmatrix.n]
    test_tsne_matrix = tsne_matrix.iloc[:, train_tmatrix.n:]

    tsne_fig1 = plot_cells(train_tsne_matrix, cell_labels=train_cell_labels,
                      show_n=False, seed=seed, jitter=jitter,
                      **kwargs)
    tsne_fig2 = plot_cells(test_tsne_matrix, cell_labels=test_cell_labels,
                      show_n=False, seed=seed, jitter=jitter,
                      **kwargs)

    vc2 = test_cell_labels.value_counts()
    
    for i, d in enumerate(tsne_fig1.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        d.marker.symbol = 'circle-open'
        d.showlegend = False

    for i, d in enumerate(tsne_fig2.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        d.marker.symbol = test_marker
        if show_n:
            d.name = '%s (n=%d)' % (label, vc2.loc[label])
        d.showlegend = True
        tsne_fig1.data.append(d)

    xrange1 = tsne_fig1.layout.xaxis.range
    xrange2 = tsne_fig2.layout.xaxis.range
    yrange1 = tsne_fig1.layout.yaxis.range
    yrange2 = tsne_fig2.layout.yaxis.range

    xrange = [min(xrange1[0], xrange2[0]), max(xrange1[1], xrange2[1])]
    yrange = [min(yrange1[0], yrange2[0]), max(yrange1[1], yrange2[1])]

    tsne_fig1.layout.xaxis.range = xrange
    tsne_fig1.layout.yaxis.range = yrange

    train_mds_matrix = mds_matrix.iloc[:, :train_tmatrix.n]
    test_mds_matrix = mds_matrix.iloc[:, train_tmatrix.n:]

    mds_fig1 = plot_cells(train_mds_matrix, cell_labels=train_cell_labels,
                      show_n=False, seed=seed, jitter=jitter,
                      **kwargs)
    mds_fig2 = plot_cells(test_mds_matrix, cell_labels=test_cell_labels,
                      show_n=False, seed=seed, jitter=jitter,
                      **kwargs)

    vc2 = test_cell_labels.value_counts()
    
    for i, d in enumerate(mds_fig1.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        d.marker.symbol = 'circle-open'
        d.showlegend = False

    for i, d in enumerate(mds_fig2.data):
        label = d.name
        idx = cluster_indices[label]
        d.marker.color = DEFAULT_PLOTLY_COLORS[idx]
        d.marker.symbol = test_marker
        if show_n:
            d.name = '%s (n=%d)' % (label, vc2.loc[label])
        d.showlegend = True
        mds_fig1.data.append(d)

    xrange1 = mds_fig1.layout.xaxis.range
    xrange2 = mds_fig2.layout.xaxis.range
    yrange1 = mds_fig1.layout.yaxis.range
    yrange2 = mds_fig2.layout.yaxis.range

    xrange = [min(xrange1[0], xrange2[0]), max(xrange1[1], xrange2[1])]
    yrange = [min(yrange1[0], yrange2[0]), max(yrange1[1], yrange2[1])]

    mds_fig1.layout.xaxis.range = xrange
    mds_fig1.layout.yaxis.range = yrange

    return tsne_fig1, mds_fig1


def circular_validation_plot(
        tmatrix: ExpMatrix,
        cell_clusters: CellAnnVector,
        pred_cell_clusters: CellAnnVector,
        perplexity: float = 160.0,
        num_cells: int = 1000,
        show_n: bool = True, seed: int = 0, jitter=0.02,
        **kwargs) -> go.Figure:
    """Plots incorrectly assigned cells after circular validation."""

    vc = cell_clusters.value_counts()
    cluster_indices = dict([(vc.index[i], i) for i in range(vc.size)])        
    i = vc.size
    for c in cell_clusters.value_counts().index:
        if  c not in cluster_indices:
            cluster_indices[c] = i
            i += 1
        
    # sample cells
    if num_cells < tmatrix.n:
        sel_cells = tools.downsample_cells(
            tmatrix.cells,
            cell_labels=cell_clusters,
            num_cells=num_cells, seed=seed)

        tmatrix = tmatrix.loc[:, sel_cells]
        cell_clusters = cell_clusters.loc[sel_cells]
        pred_cell_clusters = pred_cell_clusters.loc[sel_cells]
        
    #print(tmatrix.shape)
    #print(cell_clusters.shape)
    #print(pred_cell_clusters.shape)
    incorrect = (pred_cell_clusters != cell_clusters)        
        
    # perform t-SNE
    tsne_matrix = tools.tsne(tmatrix, perplexity=perplexity, seed=seed)
    
    vc = cell_clusters.value_counts()

    acc = dict()
    for ct in vc.index:
        sel = (cell_clusters == ct)

        acc[ct] = accuracy_score(cell_clusters.loc[sel], pred_cell_clusters[sel])
    
    cluster_mapping = dict([
        [ct, '%s (%.1f %% acc.)' % (ct, 100*acc[ct])] for ct in vc.index
    ])
    
    #cluster_mapping = dict([])
    cell_clusters2 = cell_clusters.map(cluster_mapping)
    
    fig = plot_cells(tsne_matrix, cell_labels=cell_clusters2,
                     show_n=False, seed=seed, jitter=jitter, legend_font_size=20,
                     **kwargs)
    
    marker_size = None

    np.random.seed(seed)
    dx = np.random.rand(tmatrix.shape[1]) - 0.5
    dy = np.random.rand(tmatrix.shape[1]) - 0.5
    ptp_x = tsne_matrix.iloc[0].ptp()
    ptp_y = tsne_matrix.iloc[1].ptp()
    dx = dx*ptp_x*jitter
    dy = dy*ptp_y*jitter    
    tsne_matrix.iloc[0, :] += dx
    tsne_matrix.iloc[1, :] += dy
    
    marker_size = fig.data[0].marker_size
    if marker_size is None:
        marker_size = 10
    
    for i, d in enumerate(fig.data):
        d.marker.color = DEFAULT_PLOTLY_COLORS[i]
        d.marker.size = marker_size
    
    for pop in vc.index:
        sel = (pred_cell_clusters == pop) & incorrect
        if sel.sum() == 0:
            continue
        pop_index = vc.index.tolist().index(pop)
        color = DEFAULT_PLOTLY_COLORS[pop_index]
        t = go.Scatter(
            x=tsne_matrix.iloc[0].loc[sel],
            y=tsne_matrix.iloc[1].loc[sel],
            mode='markers',
            marker=dict(
                color=color,
                symbol='circle-dot',
                size=marker_size/3.0,
            ),
            showlegend=False,
        )
        fig.data.append(t)
        t = go.Scatter(
            x=tsne_matrix.iloc[0].loc[sel],
            y=tsne_matrix.iloc[1].loc[sel],
            mode='markers',
            marker=dict(
                opacity=0.5,
                color='black',
                symbol='circle-open',
                size=marker_size,
            ),
            showlegend=False,
        )
        fig.data.append(t)

    return fig
