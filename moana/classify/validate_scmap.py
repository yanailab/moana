import logging
from typing import Iterable

import pandas as pd

from ..core import ExpMatrix, CellAnnVector
from . import ScmapClusterClassifier
from .util import calculate_accuracy_scores

_LOGGER = logging.getLogger(__name__)


def retrain_classifier(
        clf: ScmapClusterClassifier,
        matrix: ExpMatrix):

    # use the original classifier to predict the cell types
    # in the new dataset
    pred_cell_types = clf.predict(matrix)

    # exclude cells with cell type unassigned
    pred_cell_types = pred_cell_types.loc[pred_cell_types != 'unassigned']
    matrix = matrix.loc[:, pred_cell_types.cells]

    vc = pred_cell_types.value_counts()
    if len(vc.index) != len(clf.cell_types_):
        _LOGGER.warning(
            'Predicted cell types belong to only  %d / %d cell types present '
            'in the original classifier.', len(vc.index), len(clf.cell_types_))

    #if len(vc.index) == 1:
    #    _LOGGER.warning(
    #        'Cannot train a classifier with only a single cell type present.')
    #    return None

    # train classifier on training dataset
    new_clf = ScmapClusterClassifier(
        num_genes=clf.num_genes,
        similarity_threshold=clf.similarity_threshold)

    new_clf.fit(matrix, pred_cell_types)

    return new_clf


def validate_classifier(
        clf: ScmapClusterClassifier,
        train_matrix: ExpMatrix,
        train_cell_types: CellAnnVector,
        val_matrix: ExpMatrix):

    _LOGGER.info('Training scmap-cluster classifier on validation data...')
    val_clf = retrain_classifier(clf, val_matrix)
    
    _LOGGER.info('Using new classifier to predict cell types in '
                 'training data again...')
    pred_cell_types = val_clf.predict(train_matrix)
    
    acc = calculate_accuracy_scores(train_cell_types, pred_cell_types)
    
    return acc, pred_cell_types


def validate_classifier_numgenes(
        train_matrix: ExpMatrix,
        train_cell_types: CellAnnVector,
        val_matrix: ExpMatrix, 
        num_genes_values: Iterable[int],
        similarity_threshold = 0.7):
    """Performs circular validation for ScmapClusterClassifier.
    
    """
    accuracies = []
    predictions = []

    for i, num_genes in enumerate(num_genes_values):
        _LOGGER.info('Now processing num_genes=%d', num_genes)

        clf = ScmapClusterClassifier(
            num_genes=num_genes, similarity_threshold=similarity_threshold)
        
        clf.fit(train_matrix, train_cell_types)

        acc, pred_cell_types = validate_classifier(
            clf, train_matrix, train_cell_types, val_matrix)

        acc.name = '%d' % num_genes

        predictions.append(pred_cell_types)
        accuracies.append(acc)

    df_acc = pd.concat(accuracies, axis=1)
    df_acc.columns.name = 'num_genes'

    df_pred = pd.concat(predictions, axis=1)
    df_pred.columns.name = 'num_genes'

    return df_acc, df_pred
