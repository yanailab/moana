# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

import logging
import random
from math import log
from typing import Tuple, Union, Iterable, List

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

from ..core import ExpMatrix, CellAnnVector
from .. import preprocess as pp
from .. import tools
from . import CellTypeClassifier
from ..util import get_sel_components
from .util import calculate_accuracy_scores, balance_cells, apply_smoothing
from .util import calculate_mean_precision, balance_cells
from ..classify.util import get_precision_summary

_LOGGER = logging.getLogger(__name__)


def mirror_validation(
        clf: CellTypeClassifier,
        train_matrix,
        val_matrix,
        nu_values: Iterable[float] = None,
        max_cells: int = 2000,
        min_transcript_frac: Union[float, int] = pow(2, -0.5),
        seed: int = 0):

    if nu_values is None:
        nu_values = [0.02, 0.05, 0.10, 0.2, 0.5]

    #_LOGGER.info('Now smoothing training matrix...')
    #smoothed_train_matrix, _ = apply_smoothing(
    #    train_matrix,
    #    clf.transcript_count_, min_transcript_frac,
    #    clf.smooth_pca_models_, clf.genes_)

    # predict cell types in training data
    _LOGGER.info('Predicting cell types in training data...')
    train_predictions = clf.predict(train_matrix, predict_subtypes=False)

    # apply smoothing to validation matrix
    # (override "min_transcript_frac" parameter)
    _LOGGER.info('Smoothing validation matrix...')
    smoothed_val_matrix, val_k = clf.smoothing_model_.transform(
        val_matrix, min_transcript_frac=min_transcript_frac)
    #smoothed_val_matrix, val_k = apply_smoothing(
    #    val_matrix,
    #    clf.transcript_count_, min_transcript_frac,
    #    clf.smooth_pca_models_, clf.genes_)    
    _LOGGER.info('Applied smoothing with k=%d '
                 '(classifier was trained using k=%d)', val_k, clf.k)

    val_predictions = clf.predict(smoothed_val_matrix, perform_smoothing=False)

    # balance cells
    if max_cells is not None:
        _LOGGER.info('Balancing cells in the validation matrix...')
        val_predictions = balance_cells(
            val_predictions, max_cells=max_cells, seed=seed)
        smoothed_val_matrix = \
                smoothed_val_matrix.loc[:, val_predictions.cells]
    _LOGGER.info('Cell type abundances in the balanced validation matrix: '
                 '\n%s',
                 str(val_predictions.value_counts()))
    _LOGGER.info('Shape of the balanced validation matrix: %s',
                 str(smoothed_val_matrix.shape))

    # since the smoothing model does not depend on nu, we can
    # train a new classifier on the validation data and then apply 
    # its smoothing model to the training data
    # (we only need to do this once, not for each setting of nu)
    _LOGGER.info('Training a new classifier on the smoothed and balanced '
                 'validation data...')
    new_clf = clf.retrain(
        smoothed_val_matrix, min_transcript_frac=min_transcript_frac,
        nu=clf.nu, seed=seed,
        is_smoothed=True, smooth_k=val_k, retrain_subclassifiers=False)

    new_smoothing_model = new_clf.smoothing_model_

    _LOGGER.info('Smoothing the training matrix using the '
                 'smoothing model from the validation classifier...')
    new_smoothed_train_matrix, new_train_k = \
            new_clf.smoothing_model_.transform(train_matrix)
    _LOGGER.info('Applied smoothing with k=%d (original classifier was '
                 'trained using k=%d).', new_train_k, clf.k)
    
    precision = pd.Series(index=nu_values, dtype=np.float64,
                          name='Mean precision')
    precision.index.name = 'nu'
    
    for nu in nu_values:
        _LOGGER.info('Processing ν=%s ...', str(nu))
        new_clf = clf.retrain(
            smoothed_val_matrix, min_transcript_frac=min_transcript_frac,
            nu=nu, seed=seed, is_smoothed=True, smooth_k=val_k,
            smoothing_model=new_smoothing_model)
        
        new_train_predictions = new_clf.predict(
            new_smoothed_train_matrix, perform_smoothing=False)

        precision.loc[nu] = calculate_mean_precision(
            train_predictions, new_train_predictions)

        precision_summary = get_precision_summary(
            train_predictions, new_train_predictions)
        
        _LOGGER.info('Mirror validation accuracy (precision): %s',
                     precision_summary)
        #_LOGGER.info('Mean precision: %.3f' % precision.loc[nu])
        
        #break
        
    return precision



def retrain_classifier(
        clf: CellTypeClassifier,
        matrix: ExpMatrix,
        max_cells: int = 2000,
        retrain_subclassifiers: bool = True,
        min_transcript_frac: float = None,
        nu: float = None,
        seed: int = 0):
    """Recursively re-train a cell type classifier on a new dataset.

    """
    sublogger = logging.getLogger('moana.preprocess.smooth')
    prev_level = sublogger.level
    sublogger.setLevel(logging.WARNING)

    sublogger2 = logging.getLogger('moana.classify.util')
    prev_level2 = sublogger2.level
    sublogger2.setLevel(logging.WARNING)

    if min_transcript_frac is None:
        # if not provided, use the parameter value from the original
        # classifier
        min_transcript_frac = clf.min_transcript_frac

    # perform smoothing...we need to apply smoothing to the entire matrix
    # anyway, because we need to predict the cell types for all cells
    _LOGGER.info('Predicting cell types in new dataset...')
    smoothed_matrix, k = apply_smoothing(
        matrix,
        clf.transcript_count_, clf.min_transcript_frac,
        clf.smooth_pca_models_, clf.genes_)

    predictions = clf.predict(smoothed_matrix, is_smoothed=True)
    _LOGGER.info('The new classifier will be trained on data smoothed '
                 'with k=%d (original classifier: k=%d).', k, clf.k)

    vc = predictions.value_counts()
    if len(vc.index) != len(clf.cell_types_):
        _LOGGER.warning(
            'Predicted cell types in the validation dataset '
            'belong to only  %d / %d cell types present '
            'in the original classifier.',
            len(vc.index), len(clf.cell_types_))

    if len(vc.index) == 1:
        _LOGGER.warning(
            'Cannot train a classifier with only a single cell type present.')
        return None
    
    _LOGGER.info('Balancing the cell types and downsampling (if necessary)...')
    train_cell_labels = balance_cells(predictions, max_cells, seed=seed)
    train_matrix = matrix.loc[:, train_cell_labels.cells]
    train_matrix = train_matrix.loc[train_matrix.var(axis=1) > 0]

    if nu is not None:
        new_nu = nu
    else:
        new_nu = clf.nu

    # then, train a new classifier on the new data
    new_clf = CellTypeClassifier(
        k=k, d=clf.d, nu=new_nu,
        min_transcript_frac=min_transcript_frac, seed=seed)

    _LOGGER.info('Training a new classifier on the new dataset...')
    new_clf.fit(train_matrix, train_cell_labels)

    if not retrain_subclassifiers:
        # we're done
        return new_clf

    # retrain any existing subclassifiers
    for ctype in clf.cell_types_:

        subclf = clf.get_subclassifier(ctype)
        if subclf is None:
            # no subclassifier present for this cell type
            continue

        subtype_matrix = matrix.loc[:, predictions == ctype]
        if subtype_matrix.n == 0:
            _LOGGER.error(
                'No cells for cell type "%s", therefore cannot train '
                'sub-classifier!')
            continue
        _LOGGER.info('Train sub-classifier for cell type "%s"...', ctype)

        try:
            new_subclf = retrain_classifier(
                subclf, subtype_matrix, nu=nu, seed=seed)
        except ValueError as err:
            _LOGGER.error('Retraining of subclassifier produced an error '
                          'message ("%s"). Skipping...',  str(err))
        else:
            new_clf.add_subclassifier(ctype, new_subclf)

    sublogger.setLevel(prev_level)
    sublogger2.setLevel(prev_level2)

    return new_clf


def calculate_accuracies(true_labels: CellAnnVector,
                         pred_labels: CellAnnVector) -> pd.DataFrame:
    """Calculates overall coherence as well as precision/recall per cell type.

    """
    vc = true_labels.value_counts()
    
    columns = ['Precision', 'Recall']
    index = ['Overall accuracy'] + vc.index.tolist()
    
    df = pd.DataFrame(index=index, columns=columns, dtype=np.float64)
    df.loc['Overall accuracy'] = accuracy_score(true_labels, pred_labels)
    
    for ctype, num_cells in vc.iteritems():
        pred_sel = (pred_labels == ctype)
        # precision: what fraction of the predicted cells of this type are actually that type?
        precision = (true_labels.loc[pred_sel] == ctype).sum() / float(pred_sel.sum())
        # recall: what fraction of the actual cells of this type were correctly predicted?
        recall = (true_labels.loc[pred_sel] == ctype).sum() / float(num_cells)
        df.loc[ctype] = (precision, recall)
    
    return df


def get_mirror_predictions(
        clf: CellTypeClassifier,
        train_matrix: ExpMatrix,
        mirror_matrix: ExpMatrix,
        apply_subclassifiers: bool = True,
        nu: float = None,
        seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:

    _LOGGER.info('Training new classifier on mirror data...')
    new_clf = retrain_classifier(
        clf, mirror_matrix,
        retrain_subclassifiers=apply_subclassifiers,
        nu=nu,
        seed=seed)
    
    _LOGGER.info('Using new classifier to predict cell types in '
                 'training data again...')
    predicted_labels = new_clf.predict(train_matrix)
    
    #acc = calculate_accuracies(train_cell_labels, predicted_labels)
    return predicted_labels
