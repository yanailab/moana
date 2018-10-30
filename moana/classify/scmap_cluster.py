import logging
from typing import Tuple
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_distances
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from ..core import ExpMatrix, CellAnnVector
from ..core import CellAnnMatrix

_LOGGER = logging.getLogger(__name__)


class ScmapClusterClassifier:
    """Class implementing the scmap-cluster method."""

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    def __init__(self, num_genes: int = 500,
                 similarity_threshold: float = 0.7) -> None:
        self.num_genes = num_genes
        self.similarity_threshold = similarity_threshold
        
        self.logcount_mean_ = None
        self.frac_zero_ = None
        self.regression_model_ = None
        self.sel_genes_ = None
        self.cluster_medioids_ = None

    @property
    def cell_types_(self):
        return self.cell_clusters_.value_counts().index.tolist()

    def _select_genes(self, matrix: ExpMatrix) -> pd.Index:
        # we assume matrix is already normalized and log2(x+1)-transformed

        mean = matrix.mean(axis=1)
        frac_zero = (matrix == 0).sum(axis=1) / matrix.shape[1]
        
        # only use genes that aren't all zero, and that have some dropout (all)
        sel = (frac_zero > 0) & (frac_zero < 1)
        
        mean = mean.loc[sel]
        frac_zero = frac_zero.loc[sel]

        # regress fraction of dropout on mean of log expression using linear regression
        model = LinearRegression()
        model.fit(mean.values.reshape(-1, 1), np.log2(100*frac_zero))

        # calculate residuals
        res = np.log2(100*frac_zero) - model.predict(mean.values.reshape(-1,1))
        a = np.argsort(res)[::-1]
        
        sel_genes = matrix.genes.to_series().loc[sel] \
                .iloc[a[:self.num_genes]].index
        return mean, frac_zero, model, sel_genes


    def fit(self, matrix, cell_clusters) -> None:
        # we assume matrix is already normalized and log2(x+1)-transformed
        
        #if not count_matrix.cells.equals(logcount_matrix.cells):
        #    raise ValueError('"count_matrix" and "logcount_matrix" must contain identical cells!')
        
        if not cell_clusters.index.to_series().isin(matrix.cells).all():
            raise ValueError('Not all clustered cells are contained in the '
                             'expression matrix!')

        # make sure data are aligned
        #count_matrix = count_matrix.loc[:, cell_clusters.index]
        matrix = matrix.loc[:, cell_clusters.index]
        
        # select genes
        mean, frac_zero, model, sel_genes = self._select_genes(matrix)
        
        self.logcount_mean_ = mean
        self.frac_zero_ = frac_zero
        self.regression_model_ = model
        self.sel_genes_ = sel_genes
        self.cell_clusters_ = cell_clusters
        
        # calculate medioids
        vc = cell_clusters.value_counts()
        cluster_labels = vc.index.tolist()
        
        data = np.empty((sel_genes.size, len(cluster_labels)),
                        dtype=np.float64)
        med = ExpMatrix(genes=sel_genes, cells=cluster_labels, data=data)
        
        for label in cluster_labels:
            med.loc[:, label] = matrix.loc[sel_genes, cell_clusters == label] \
                    .median(axis=1)
            
        self.cluster_medioids_ = med


    def plot_gene_selection(self,
                            xrange: Tuple[float, float] = None,
                            yrange: Tuple[float, float] = None,
                            num_genes: int = None, seed: int = 0,
                            marker_size: float = 5.0) -> go.Figure:
        #mean, frac_zero, model, res, top=500, xrange=None, yrange=None, num_genes=1000, seed=0)    
        
        if self.cluster_medioids_ is None:
            raise RuntimeError('You must train the model first!')
        
        mean = self.logcount_mean_
        frac_zero = self.frac_zero_
        model = self.regression_model_
        sel_genes = self.sel_genes_
        
        if num_genes is None:
            num_genes = mean.size

        if xrange is None:
            ptp = np.ptp(mean)
            xmin = (np.amin(mean) - 0.03*ptp)
            xmax = (np.amax(mean) + 0.03*ptp)
        else:
            xmin, xmax = xrange

        np.random.seed(seed)
        sel = np.random.choice(mean.size, size=num_genes, replace=False)

        trace = go.Scatter(
            x=mean.iloc[sel],
            #y=np.log2((100*frac_zero.loc[sel])),
            y=np.log2(100*frac_zero).iloc[sel],
            mode='markers',
            marker=dict(size=marker_size),
        )

        trace2 = go.Scatter(
            x=mean.loc[sel_genes],
            #y=np.log2((100*frac_zero.loc[sel])),
            #y=np.log2(100*frac_zero.loc[sel]).iloc[a[:top]],
            y=np.log2(100*frac_zero.loc[sel_genes]),
            mode='markers',
            marker=dict(color='red', size=marker_size),
        )

        y1 = model.intercept_ + xmin*model.coef_[0]
        y2 = model.intercept_ + xmax*model.coef_[0]
        trace3 = go.Scatter(
            x=[xmin, xmax],
            y=[y1, y2],
            mode='lines',
            line=dict(color='black'),
        )

        data = [trace, trace2, trace3]

        layout = go.Layout(
            xaxis=dict(range=[xmin, xmax], zeroline=False, showline=True,
                       ticklen=5, title='Mean log2(Expression)'),
            yaxis=dict(range=yrange, zeroline=False, showline=True,
                       ticklen=5, title='log2(% of zero values)'),
            width=800,
            height=800,
            showlegend=False,
            font=dict(size=24),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

        
    def predict(self, logcount_matrix):
        
        if self.cluster_medioids_ is None:
            raise RuntimeError('You must train the model first!')
        
        cluster_medioids = self.cluster_medioids_
        
        comb = logcount_matrix.genes & cluster_medioids.genes
        
        if len(comb) == 0:
            raise ValueError('No genes in common!')
        
        sim_cosine = 1 - \
                pairwise_distances(
                    logcount_matrix.loc[comb].T,
                    cluster_medioids.loc[comb].T,
                    metric='cosine')
            
        sim_pearson = 1 - \
                pairwise_distances(
                    logcount_matrix.loc[comb].T,
                    cluster_medioids.loc[comb].T,
                    metric='correlation')
        
        # for spearman, transform the genes to ranks
        # and then calculate Pearson correlation
        medioids_ranked = cluster_medioids.loc[comb].rank(axis=0)
        matrix_ranked = logcount_matrix.loc[comb].rank(axis=0)
        
        sim_spearman = 1 - \
                pairwise_distances(
                    matrix_ranked.T,
                    medioids_ranked.T,
                    metric='correlation')
        
        # return shape: #cells-by-#clusters
        
        n = len(logcount_matrix.cells)  # the number of cells
        m = len(cluster_medioids.columns)  # the number of classes
        
        data = np.zeros((n, m), dtype=np.int64)
        votes = CellAnnMatrix(
            cells=logcount_matrix.cells,
            columns=cluster_medioids.columns,
            data=data)
        
        for sim in [sim_cosine, sim_pearson, sim_spearman]:
            #for row, col in zip(np.arange(n), sim.idxmax(axis=1)):
            for row, col in zip(np.arange(n), np.argmax(sim, axis=1)):
                votes.iat[row, col] += 1
        
        assert votes.sum().sum() == 3*n
        #num_unassigned = (votes.max(axis=1) < 2).sum()
        
        assigned = pd.Series(index=votes.index, data=np.ones(votes.shape[0],
                             dtype=np.bool_))
        assigned.loc[votes.max(axis=1) < 2] = False

        num_unassigned = (~assigned).sum()
        _LOGGER.info('Number of cells unassigned after voting: '
                     '%d / %d (%.1f %%)',
                     num_unassigned, n, 100*(num_unassigned/n))
        
        winner_labels = votes.idxmax(axis=1)
        
        winner = votes.values.argmax(axis=1)
        
        sim_max = np.stack([sim_cosine, sim_pearson, sim_spearman], axis=-1) \
                .max(axis=-1)
        winner_max_sim = np.float64([sim_max[i, winner[i]] for i in range(n)])
        
        assigned.iloc[winner_max_sim < self.similarity_threshold] = False
        
        num_unassigned = (~assigned).sum()
        _LOGGER.info('Number of cells unassigned after applying similarity '
                     'threshold: %d / %d (%.1f %%)',
                     num_unassigned, n, 100*(num_unassigned/n))

        winner_labels.loc[~assigned] = 'unassigned'
        winner_labels = CellAnnVector(winner_labels)
        return winner_labels


    def write_pickle(self, file_path: str) -> None:
        """Write classifier to file in pickle format."""
        #pred.write_pickle('')
        with open(file_path, 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Wrote classifier to "%s".', file_path)
 

    @classmethod
    def read_pickle(cls, file_path: str):
        """Read classifier from pickle file."""
        with open(file_path, 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded classifier from "%s".', file_path)
        return clf
