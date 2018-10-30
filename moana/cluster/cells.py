# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Functions for clustering cells."""

import logging
import sys
from typing import Union, Iterable

from sklearn.cluster import DBSCAN, KMeans, MeanShift
#from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from ..core import ExpMatrix, CellAnnVector
from ..util import get_sel_components
from .. import preprocess as pp

_LOGGER = logging.getLogger(__name__)


def cluster_cells_dbscan(
        smoothed_matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        min_cells_frac: float = 0.03,
        eps_frac: float = 0.10,
        seed: int = 0) -> CellAnnVector:
    """Cluster cells using DBSCAN."""

    sel_components = get_sel_components(components)

    num_components = max(sel_components) + 1 
    tmatrix, _ = pp.pca(
        pp.median_normalize(smoothed_matrix), num_components, seed=seed)

    ptp = np.ptp(tmatrix.iloc[sel_components].values, axis=1)
    eps = pow(np.sum(np.power(eps_frac*ptp, 2.0)), 0.5)
    min_cells = min_cells_frac * tmatrix.shape[1]

    model = DBSCAN(algorithm='brute', min_samples=min_cells, eps=eps)

    y = model.fit_predict(tmatrix.iloc[sel_components].T)
    cell_labels = CellAnnVector(cells=tmatrix.cells, data=y)

    return cell_labels


def cluster_cells_meanshift(
        smoothed_matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        bw_frac: float = 0.10,
        bin_seeding: bool = False,
        seed: int = 0) -> CellAnnVector:
    """Cluster cells using mean shift."""

    sel_components = get_sel_components(components)

    num_components = max(sel_components) + 1 
    tmatrix, _ = pp.pca(
        pp.median_normalize(smoothed_matrix), num_components, seed=seed)

    ptp = np.ptp(tmatrix.iloc[sel_components].values, axis=1)
    bw = pow(np.sum(np.power(bw_frac*ptp, 2.0)), 0.5)
    
    model = MeanShift(bandwidth=bw, bin_seeding=bin_seeding)

    y = model.fit_predict(tmatrix.iloc[sel_components].T)
    cell_labels = CellAnnVector(cells=tmatrix.cells, data=y)

    return cell_labels    


def cluster_cells_kmeans(
        smoothed_matrix: ExpMatrix,
        components: Union[int, Iterable[int]] = 2,
        num_clusters: int = 8,
        seed: int = 0, **kwargs) -> CellAnnVector:
    """Cluster cells using K-means."""

    sel_components = get_sel_components(components)

    num_components = max(sel_components) + 1 
    tmatrix, _ = pp.pca(
        pp.median_normalize(smoothed_matrix), num_components, seed=seed)

    model = KMeans(n_clusters=num_clusters, random_state=seed, **kwargs)

    y = model.fit_predict(tmatrix.iloc[sel_components].T)
    cell_labels = CellAnnVector(cells=tmatrix.cells, data=y)

    return cell_labels


def get_overexpressed_genes(
        matrix: ExpMatrix, cell_labels: CellAnnVector,
        low_thresh: float = None, min_transcripts: Union[int, float] = 25,
        top: int = 10):
    """Find most over-expressed genes for each cluster."""
    vc = cell_labels.value_counts()
    clusters = vc.index.tolist()

    data = []
    mean = []
    
    if low_thresh is None:
        low_thresh = min_transcripts / float(vc.iloc[-1])
    _LOGGER.info('Setting all mean expression values below %.3f to %.3f, '
                 'before calculating fold changes.', low_thresh, low_thresh)

    # first, apply per-cluster normalization
    #norm_matrix = matrix.copy()

    num_transcripts = matrix.sum(axis=0)
    norm_matrix = (num_transcripts.median() / num_transcripts) * matrix
    
    # set all small counts to low_thresh
    #norm_matrix[norm_matrix < low_thresh] = low_thresh
    
    # next, determine fold-change for all genes, for each cluster
    X = np.ones((matrix.p, len(clusters)), dtype=np.float64)
    fold_change_up = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)
    #X = np.ones((matrix.p, len(labels)), dtype=np.float64)
    #fold_change_down = ExpMatrix(genes=matrix.genes, cells=labels, X=X)
    
    X = np.zeros((matrix.p, len(clusters)), dtype=np.float64)
    cluster_exp_mean = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)

    #X = np.zeros((matrix.p, len(clusters)), dtype=np.float64)
    #cluster_exp_high = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)    
    
    #print('Cluster labels:', labels)
    
    # calculate all necessary values
    for l in clusters:
        _LOGGER.info('Now processing cluster "%s"...', l)
        sys.stdout.flush()
        sel = (cell_labels == l)
        cluster_exp_mean.loc[:, l] = norm_matrix.loc[:, sel].mean(axis=1)

    # ignore all expression lower than "low_thresh"
    cluster_exp_mean[cluster_exp_mean < low_thresh] = low_thresh
    
    # calculate fold change relative to other cluster with max. expression
    for l in clusters:
        sel = (cluster_exp_mean.cells != l)
        fold_change_up.loc[:, l] = cluster_exp_mean.loc[:, l] / \
                (cluster_exp_mean.loc[:, sel].mean(axis=1))
        
    markers_up = []
    #markers_down = []
    for l in clusters:
        fc = fold_change_up.loc[:, l].sort_values(ascending=False)
        fc = fc[:top]
        
        exp = cluster_exp_mean.loc[fc.index, l]

        label_index = [l]*top
        gene_index = fc.index
        index = pd.MultiIndex.from_arrays(
            [label_index, gene_index],
            names=['cluster', 'gene'])
        
        data = np.c_[fc.values, exp.values]
        
        markers_up.append(
            pd.DataFrame(
                index=index,
                columns=['fold_change', 'mean_exp_level'],
                data=data))
        
    markers_up = pd.concat(markers_up, axis=0)
    #markers_down = pd.concat(markers_down, axis=0)
    
    markers_up = markers_up.swaplevel(0, 1).sort_index(
        level=1, sort_remaining=False).swaplevel(0, 1)
    
    return markers_up


def get_marker_genes(
        matrix: ExpMatrix, cell_labels: CellAnnVector,
        low_thresh: float = None, min_transcripts: Union[int, float] = 25,
        top: int = 10):
    """Find most over-expressed genes for each cluster."""
    vc = cell_labels.value_counts()
    clusters = vc.index.tolist()

    data = []
    mean = []
    
    if low_thresh is None:
        low_thresh = min_transcripts / float(vc.iloc[-1])
    _LOGGER.info('Setting all mean expression values below %.3f to %.3f, '
                 'before calculating fold changes.', low_thresh, low_thresh)

    # first, apply per-cluster normalization
    #norm_matrix = matrix.copy()

    num_transcripts = matrix.sum(axis=0)
    norm_matrix = (num_transcripts.median() / num_transcripts) * matrix
    
    # set all small counts to low_thresh
    #norm_matrix[norm_matrix < low_thresh] = low_thresh
    
    # next, determine fold-change for all genes, for each cluster
    X = np.ones((matrix.p, len(clusters)), dtype=np.float64)
    fold_change_up = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)
    #X = np.ones((matrix.p, len(labels)), dtype=np.float64)
    #fold_change_down = ExpMatrix(genes=matrix.genes, cells=labels, X=X)
    
    X = np.zeros((matrix.p, len(clusters)), dtype=np.float64)
    cluster_exp_mean = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)

    #X = np.zeros((matrix.p, len(clusters)), dtype=np.float64)
    #cluster_exp_high = ExpMatrix(genes=matrix.genes, cells=clusters, X=X)    
    
    #print('Cluster labels:', labels)
    
    # calculate all necessary values
    for l in clusters:
        _LOGGER.info('Now processing cluster "%s"...', l)
        sys.stdout.flush()
        sel = (cell_labels == l)
        cluster_exp_mean.loc[:, l] = norm_matrix.loc[:, sel].mean(axis=1)

    # ignore all expression lower than "low_thresh"
    cluster_exp_mean[cluster_exp_mean < low_thresh] = low_thresh
    
    # calculate fold change relative to other cluster with max. expression
    for l in clusters:
        sel = (cluster_exp_mean.cells != l)
        fold_change_up.loc[:, l] = cluster_exp_mean.loc[:, l] / \
                (cluster_exp_mean.loc[:, sel].max(axis=1))
        
    markers_up = []
    #markers_down = []
    for l in clusters:
        fc = fold_change_up.loc[:, l].sort_values(ascending=False)
        fc = fc[:top]
        
        exp = cluster_exp_mean.loc[fc.index, l]

        label_index = [l]*top
        gene_index = fc.index
        index = pd.MultiIndex.from_arrays([label_index, gene_index], names=['cluster', 'gene'])
        
        data = np.c_[fc.values, exp.values]
        
        markers_up.append(pd.DataFrame(index=index, columns=['fold_change', 'mean_exp_level'], data=data))
        
    markers_up = pd.concat(markers_up, axis=0)
    #markers_down = pd.concat(markers_down, axis=0)
    
    markers_up = markers_up.swaplevel(0, 1).sort_index(level=1, sort_remaining=False).swaplevel(0, 1)
    
    return markers_up
