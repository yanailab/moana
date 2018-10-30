# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Cell type classifier for single-cell RNA-Seq data."""

import logging
import pickle
from typing import Dict, Optional, Tuple, Iterable, Union, List
import time
import sys
import copy
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.svm import NuSVC
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

from ..core import ExpMatrix, CellAnnVector, CellAnnMatrix
from .. import preprocess as pp
from .. import tools
from .. import visualize
from ..util import get_sel_components, get_component_labels
from ..util import get_component_labels
from .util import apply_smoothing, apply_pca
from .util import get_precision_summary
from .util import balance_cells
from . import PCAModel, SmoothingModel

_LOGGER = logging.getLogger(__name__)


class CellTypeClassifier:

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    """A cell type classifier for smoothed scRNA-Seq data."""
    def __init__(self, k: int = 1, d: int = 20,
                 components: Union[Iterable[int], int] = None,
                 #components: Union[Int, Iterable[Int]] = None,
                 nu: float = 0.02,
                 min_transcript_frac: Union[float, int] = 0.9,
                 smooth_min_transcripts: Union[int, float] = 500,
                 sub_classifiers=None,
                 name: str = None,
                 seed: int = 0) -> None:

        if components is None:
            components = d

        sel_components = get_sel_components(components)
        #if num_components > d:
        #    raise ValueError('The highest selected principal component (%d) '
        #                     'cannot be higher than "d" (%d).'
        #                     % (num_components, d))

        if sub_classifiers is None:
            sub_classifiers = {}

        self.name = name
        self.k = k
        self.d = d
        #self.num_components
        self.sel_components = sel_components
        self.nu = nu
        self.seed = seed
        self.min_transcript_frac = min_transcript_frac
        self.smooth_min_transcripts = smooth_min_transcripts
        self.sub_classifiers = sub_classifiers

        self.cell_labels_ = None
        self.smoothing_model_ = None
        self.pca_model_ = None
        self.svm_model_ = None

    #@property
    #def sel_components(self) -> List[int]:
    #    return get_sel_components(self.d)

    @property
    def genes_(self):
        return self.pca_model_.genes_

    @property
    def transcript_count_(self):
        return self.pca_model_.transcript_count_

    @property
    def num_components(self) -> int:
        """Returns the number of the highest selected principal component."""
        return max(self.sel_components) + 1


    def __str__(self) -> str:
        try:
            name = self.name
        except AttributeError:
            name = None
        
        if name is None:
            name_str = '(no name)'
        else:
            name_str = '"%s"' % name

        param_str = '\n'.join([
            '- d = %d' % self.d,
            '- transcript_count = %s\n'
            '  (k = %d; %s / cell)'
            % (str(self.transcript_count_), self.k,
               str(self.transcript_count_ / self.k)),
            '- sel_components = %s' % str(self.sel_components),
            '- nu = %s' % str(self.nu),
            '- min_transcript_frac = %s' % str(self.min_transcript_frac),
            '- smooth_min_transcripts = %s' % str(self.smooth_min_transcripts),
            '- seed = %d' % self.seed])

        if self.svm_model_ is None:
            header_str = ('Moana cell type classifier (**untrained**)\n'
                          '------------------------------------------')
        else:
            header_str = ('Moana cell type classifier\n'
                          '--------------------------')

        clf_str = ('%s\n' % header_str + 
                   'Name: %s\n\n' % name_str + 
                   'Top-level model parameters:\n'
                   '%s' % param_str)

        if self.svm_model_ is None:
            return clf_str

        ctype_str = self._get_ctype_str()

        msg = ('%s\n\n' % clf_str +
               'Classifification tree (* = final cell type):\n'
               '%s' % ctype_str)

        return msg


    def _get_ctype_str(self, prefix='-', include_subtypes: bool = True) -> str:
        """Get a bullet list of all cell types / subtypes of this classifier.
        
        """
        ctype_list = []
        #for ctype, n in self.value_counts_.iteritems():
        for ctype in sorted(self.cell_types_):
            #if include_n:
            #    subtype_str = '%s %s (n=%d)' % (prefix, ctype, n)
            subclf = None
            try:
                subclf = self.sub_classifiers[ctype]
            except KeyError:
                pass
            if subclf is None:
                suffix = '*'
            else:
                suffix = ''
            subtype_str = '%s %s%s' % (prefix, ctype, suffix)
            if include_subtypes:
                if subclf is not None:
                    subtype_str += ('\n%s' % subclf._get_ctype_str(prefix + '-'))
            ctype_list.append('%s' % subtype_str)

        return '\n'.join(ctype_list)


    def _require_trained_classifier(self) -> None:
        if self.svm_model_ is None:
            raise RuntimeError('You must train the classifier first!')


    @property
    def num_cells_(self) -> int:
        """Returns the number of cells in the training data."""
        return self.cell_labels_.size


    @property
    def value_counts_(self) -> pd.DataFrame:
        """Returns the value counts for the training labels."""
        return self.cell_labels_.value_counts()


    @property
    def classes_(self) -> List[str]:
        return self.cell_types_


    @property
    def cell_types_(self) -> List[str]:
        self._require_trained_classifier()
        return self.svm_model_.classes_


    @property
    def gene_loadings_(self) -> ExpMatrix:
        """Returns a matrix with the gene loadings from the selected PCs."""
        return self.pca_model_.loadings_


    @property
    def gene_coef_(self) -> ExpMatrix:
        """Returns a matrix with gene coefficients for SVM classifier."""
        gene_loadings = self.gene_loadings_
        svm_coef = self.svm_model_.coef_

        n = len(self.cell_types_)
        clf_labels = [
            '"%s" vs "%s"' % (self.cell_types_[i], self.cell_types_[j])
            for i in range(n-1) for j in range(i+1, n)]

        gene_coef = ExpMatrix(
            genes=self.genes_, cells=clf_labels,
            dtype=np.float64)
        for j in range(len(svm_coef)):
            gene_coef.iloc[:, j] = gene_loadings.values.dot(svm_coef[j])

        return gene_coef.copy()


    @property
    def normalized_gene_coef_(self) -> ExpMatrix:
        gene_coef = self.gene_coef_
        normalized_gene_coef = gene_coef.copy()
        svm_coef = self.svm_model_.coef_
        for j in range(len(svm_coef)):
            sel = (svm_coef[j] >= 0)
            size = (svm_coef[j][sel].sum() - svm_coef[j][~sel].sum())
            factor = 1/size
            normalized_gene_coef.iloc[:, j] *= factor
        return normalized_gene_coef


    def __getattribute__(self, name):
        if name == 'transcript_count_':
            try:
                val = super().__getattribute__(name)
            except AttributeError:
                val = super().__getattribute__('med_num_transcripts_')
            return val
        else:
            return super().__getattribute__(name)


    def fit(self, matrix: ExpMatrix,
            cell_labels: CellAnnVector,
            is_smoothed: bool = False,
            smoothing_model: SmoothingModel = None) -> None:
        """Train a cell classifier for scRNA-Seq data.
        
        The user can supply either a smoothed or an unsmoothed
        expression matrix (see `is_smoothed` parameter).
        
        Notes
        -----
        The reason we request the entire CellTypeClassifier object instead of
        only the PCA object is that we also need the gene list from the other
        PCA model, which is not contained in the scikit-learn PCA object.
        """
        _LOGGER.info('Training a Moana classifier on a dataset containing ' 
                     '%d cells...', matrix.n)

        if not cell_labels.index.to_series().isin(matrix.cells).all():
            raise ValueError('Not all cells in cell type vector are '
                             'contained in the expression matrix!')

        if set(matrix.cells) != set(cell_labels.cells):
            _LOGGER.warning('Cell type vector and expression matrix do not '
                            'contain the same set of cells!')

        # make sure the two datasets are aligned
        matrix = matrix.loc[:, cell_labels.index]

        if is_smoothed:
            smoothed_matrix = matrix
        else:
            _LOGGER.info('Performing smoothing...')
            smoothed_matrix = pp.knn_smoothing(matrix, self.k, self.d)

        if smoothing_model is None:
            _LOGGER.info('Training the smoothing model...')
            smoothing_model = SmoothingModel(
                self.k, self.d, components=self.sel_components, seed=self.seed,
                min_transcript_frac=self.min_transcript_frac,
                smooth_min_transcripts=self.smooth_min_transcripts)

            smoothing_model.fit(smoothed_matrix, is_smoothed=True)
        #_LOGGER.info('Smoothed matrix hash: %s', smoothed_matrix.hash)

        # train the PCA model and select specific PCs
        # (currently we always select all the PCs)
        _LOGGER.info('Training the PCA model...')
        pca_model = PCAModel(self.sel_components, seed=self.seed)
        tmatrix = pca_model.fit_transform(smoothed_matrix)

        # report fraction of variance explained
        #frac_variance_explained = \
        #        pca_model.explained_variance_ratio_[self.sel_components].sum()
        #_LOGGER.info('Moana training -- The selected PCs represent %.1f %% of '
        #             'total variance.', 100*frac_variance_explained)

        # set training variables

        _LOGGER.info('Training the SVM model...'); sys.stdout.flush()
        # set seed before sampling
        np.random.seed(self.seed)

        # perform semi-random oversampling to balance cluster sizes
        vc = cell_labels.value_counts()
        max_cells = vc.values[0]

        train_tmatrix = []
        train_labels = []
        for cluster_label in vc.index:
            sel = (cell_labels == cluster_label)
            sub_tmatrix = tmatrix.loc[:, sel]
            num_reps = max_cells // sub_tmatrix.n
            num_remaining_cells = max_cells - (num_reps * sub_tmatrix.n)
            sub_tmatrix_rep = pd.concat(
                [sub_tmatrix]*num_reps +
                [sub_tmatrix.sample(n=num_remaining_cells, axis=1)],
                axis=1)
            sub_labels_rep = CellAnnVector(cells=sub_tmatrix_rep.cells,
                                           data=[cluster_label]*max_cells)

            train_tmatrix.append(sub_tmatrix_rep)
            train_labels.append(sub_labels_rep)

        train_tmatrix = pd.concat(train_tmatrix, axis=1)
        train_labels = pd.concat(train_labels)

        ### Train NuSVC model
        svm_model = NuSVC(nu=self.nu, kernel='linear',
                          decision_function_shape='ovo',
                          random_state=self.seed)  # intialize the model

        svm_model.fit(train_tmatrix.T, train_labels)  # train the model

        self.cell_labels_ = cell_labels.copy()
        self.smoothing_model_ = smoothing_model
        self.pca_model_ = pca_model
        self.svm_model_ = svm_model

        # report performance on training data
        predictions = self.predict(smoothed_matrix)

        #predictions2 = self.predict(matrix, is_smoothed=True)
        precision_summary = get_precision_summary(cell_labels, predictions)

        _LOGGER.info('SVM classifier performance (precision) '
                     'on smoothed training data: %s', precision_summary)


    def smooth(self, matrix: ExpMatrix) -> ExpMatrix:
        """Apply the classifier's smoothing model."""

        self._require_trained_classifier()

        #sublogger = logging.getLogger('moana.classify.smoothing_model')
        #prev_level = sublogger.level
        #sublogger.setLevel(logging.WARNING)

        smoothed_matrix, k = self.smoothing_model_.transform(matrix)
        _LOGGER.info('Applied smoothing with k=%d '
                     '(classifier was trained using k=%d)', k, self.k)

        #sublogger.setLevel(prev_level)

        return smoothed_matrix


    def transform(self, matrix: ExpMatrix,
                  perform_smoothing: bool = True,
                  include_var: bool = False) -> ExpMatrix:
        """Apply the classifier's projection model."""

        self._require_trained_classifier()

        #sublogger = logging.getLogger('moana.classify.util')
        #prev_level = sublogger.level
        #sublogger.setLevel(logging.WARNING)

        if perform_smoothing:
            smoothed_matrix = self.smooth(matrix)
        else:
            smoothed_matrix = matrix

        #num_transcripts = smoothed_matrix.sum(axis=0)
        #_LOGGER.info('Data will be scaled %.2f-fold before '
        #             'FT-transformation and projection into PC space.',
        #             self.transcript_count_ / num_transcripts.median())

        tmatrix = self.pca_model_.transform(smoothed_matrix,
                                            include_var=include_var)

        #sublogger.setLevel(prev_level)

        return tmatrix


    def predict(self, matrix: ExpMatrix,
                perform_smoothing: bool = True,
                predict_subtypes: bool = True) -> CellAnnVector:
        """Predict cell types."""

        self._require_trained_classifier()

        _LOGGER.info('Using Moana classifier to predict cell types in a '
                     'dataset containing %d cells...', matrix.n)

        _LOGGER.info(
            'This classifier uses %d principal components, '
            'num_transcripts=%d (k=%d), '
            'and min_transcript_frac=%.2f.'
            % (len(self.sel_components), self.transcript_count_,
               self.k, self.min_transcript_frac))

        t0 = time.time()

        #if perform_smoothing and predict_subtypes and self.has_subclassifiers:
        #    _LOGGER.warning('Cannot predict subtypes when given an '
        #                    'expression matrix that is already smoothed. '
        #                    'Therefore disabling sub-classification.')
        #    predict_subtypes = False

        ### 1. Perform smoothing
        if perform_smoothing:
            _LOGGER.info('Performing smoothing...'); sys.stdout.flush()
            smoothed_matrix = self.smooth(matrix)
        else:
            # skip smoothing
            _LOGGER.info('Not applying smoothing (as requested by user).')
            smoothed_matrix = matrix

        ### 2. Project cells into PC space
        _LOGGER.info('Projecting cells into PC space...')
        tmatrix = self.pca_model_.transform(smoothed_matrix)

        ### 3. Apply SVM classifier
        _LOGGER.info('Applying SVM classifier...')
        y = self.svm_model_.predict(tmatrix.T)
        predictions = CellAnnVector(cells=matrix.cells, data=y)
        predictions.name = 'Predicted cell type'

        result_str = '; '.join(
            '%s - %d' % (ctype, n)
            for ctype, n in predictions.value_counts().iteritems())
        _LOGGER.info('Prediction results: %s', result_str)

        ### 4. Apply subclassifiers (if there are any)
        if predict_subtypes:
            for ctype in self.classes_:
                subclf = None
                try:
                    subclf = self.sub_classifiers[ctype]
                except KeyError:
                    pass
                if subclf is None:
                    continue

                matrix_sub = matrix.loc[:, predictions == ctype]
                if matrix_sub.n == 0:
                    _LOGGER.info('No cells with predicted cell type "%s"!',
                    ctype)
                    continue

                _LOGGER.info('Running subclassifier for cell type "%s"...',
                             ctype)

                try:
                    pred_sub = subclf.predict(
                        matrix_sub)
                except ValueError as err:
                    # not enough cells, not enough transcripts available etc.
                    _LOGGER.error(
                        'Subclassifier produced an error ("%s"), therefore '
                        'skipping the prediction of subtypes.', str(err))
                else:
                    predictions.loc[pred_sub.cells] = pred_sub

        t1 = time.time()
        _LOGGER.info('Cell type prediction took %.1f s.', t1-t0)

        return predictions


    def retrain(
            self, matrix: ExpMatrix,
            min_transcript_frac: Union[float, int] = None,
            nu: float = None,
            seed: int = None,
            max_cells: int = None,
            is_smoothed: bool = False,
            smooth_k: int = None,
            smoothing_model: SmoothingModel = None,
            retrain_subclassifiers: bool = True):
        """Use classifier to train a new classifier on another dataset."""

        if is_smoothed:
            if retrain_subclassifiers and self.has_subclassifiers:
                _LOGGER.warning(
                    'Cannot retrain subclassifiers when provided with a '
                    'smoothed matrix. Will disable retraining of '
                    'subclassifiers.')
                retrain_subclassifiers = False
            smoothed_matrix = matrix
        else:
            smoothed_matrix, smooth_k = self.smoothing_model_.transform(
                matrix, min_transcript_frac=min_transcript_frac)

        if nu is None:
            nu = self.nu

        if min_transcript_frac is None:
            min_transcript_frac = self.min_transcript_frac

        if seed is None:
            seed = self.seed

        predictions = self.predict(
            smoothed_matrix, perform_smoothing=False, predict_subtypes=False)

        if max_cells is not None:
            # balance the classes
            train_cell_labels = balance_cells(predictions, max_cells=max_cells)
            smoothed_train_matrix = \
                    smoothed_matrix.loc[:, train_cell_labels.cells]
        else:
            train_cell_labels = predictions
            smoothed_train_matrix = smoothed_matrix

        new_clf = CellTypeClassifier(
            k=smooth_k, d=self.d, nu=nu,
            min_transcript_frac=min_transcript_frac, seed=seed,
            smooth_min_transcripts=self.smooth_min_transcripts)

        new_clf.fit(smoothed_train_matrix, train_cell_labels, is_smoothed=True,
                    smoothing_model=smoothing_model)

        if not retrain_subclassifiers:
            # we're done
            return new_clf

        # retrain any existing subclassifiers
        for ctype in self.cell_types_:

            subclf = self.get_subclassifier(ctype)
            if subclf is None:
                # no subclassifier present for this cell type
                continue

            subtype_matrix = matrix.loc[:, predictions == ctype]
            if subtype_matrix.n == 0:
                _LOGGER.error(
                    'No cells for cell type "%s", therefore cannot train '
                    'subclassifier!')
                continue
            _LOGGER.info('Trainnig subclassifier for cell type "%s"...',
                         ctype)

            try:
                new_subclf = subclf.retrain(
                    subtype_matrix,
                    min_transcript_frac=min_transcript_frac,
                    max_cells=max_cells,
                    nu=nu, seed=seed)

            except ValueError as err:
                _LOGGER.error('Retraining of subclassifier produced an error '
                            'message ("%s"). Skipping...',  str(err))

            else:
                new_clf.add_subclassifier(ctype, new_subclf)

        return new_clf


    def get_decision_function(
            self, matrix: ExpMatrix,
            cell_clusters: CellAnnVector,
            perform_smoothing: bool = True):
        """Calculate cell (average) decision func. values for their cluster."""
        
        if self.svm_model_ is None:
            raise RuntimeError('You must train the classifier first!')

        # 1. perform smoothing
        if perform_smoothing:
            smoothed_matrix, k = apply_smoothing(
                matrix,
                self.transcript_count_, self.min_transcript_frac,
                self.smooth_pca_models_, self.genes_)
            _LOGGER.info(
                'Applied smoothing with k=%d '
                '(classifier was trained using k=%d)',
                k, self.k)
        else:
            smoothed_matrix = matrix
        
        # 2. Apply PCA (will select same genes)
        #tmatrix = self._apply_pca(smoothed_matrix)
        tmatrix = apply_pca(
            self.pca_model_, smoothed_matrix,
            self.transcript_count_, components=self.sel_components,
            valid_genes=self.genes_)

        num_samples = cell_clusters.size
        num_classes = len(self.classes_)

        # get decision function values from SVM classifier
        dfun_array = self.svm_model_.decision_function(tmatrix.T)

        # calculate all averages
        dfun_summed = np.zeros((num_samples, num_classes), dtype=np.float64)
        k = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                dfun_summed[:, i] += dfun_array[:, k]
                dfun_summed[:, j] -= dfun_array[:, k]
                k += 1

        dfun_avg = dfun_summed / (num_classes - 1)
        dfun_avg = CellAnnMatrix(
            cells=tmatrix.cells, columns=self.classes_,
            data=dfun_avg)

        # select averages corresponding to cell cluster
        dfun = CellAnnVector(cells=tmatrix.cells, dtype=np.float64)
        for cell, ctype in cell_clusters.items():
            dfun.loc[cell] = dfun_avg.loc[cell, ctype]

        return dfun


    @property
    def has_subclassifiers(self):
        """Determines if the classifier has any subclassifiers."""
        has_subclf = False
        for ctype in self.cell_types_:
            try:
                subclf = self.sub_classifiers[ctype]
            except KeyError:
                pass
            else:
                if subclf is not None:
                    has_subclf = True
                    break
        return has_subclf


    def get_classifier_by_cell_type(
            self, cell_type: str, search_subclassifiers: bool = True):
        """Retrieves the (sub-)classifier for a given cell type.
        
        Raises a ValueError if the specified cell type is unknown.
        """
        ctype_str = self._get_ctype_str(include_subtypes=search_subclassifiers)

        clf = None
        if cell_type in self.cell_types_:
            return self

        if not search_subclassifiers:
            raise ValueError(
                'The top-level classifier does not have a cell type "%s". '
                'Valid cell types:\n%s' % (cell_type, ctype_str))

        for subclf in self.sub_classifiers.values():
            if subclf is None:
                continue
            try:
                clf = subclf.get_classifier_by_cell_type(cell_type)
            except ValueError:
                pass
            else:
                return clf

        raise ValueError(
            'Neither the top-level classifier nor any of its '
            'subclassifiers has a cell type "%s". '
            'Valid cell types:\n%s' % (cell_type, ctype_str))


    def add_subclassifier(self, cell_type: str, subclf,
                          search_subclassifiers: bool = True) -> None:
        """Adds a subclassifier for a given cell type.
        
        If a subclassifier for the cell type has already been defined, it will
        be replaced and a warning message will be issued.
        """

        if subclf is None:
            _LOGGER.warning('No classifier specified, will do nothing.'
                            'Use "remove_subclassifier" to remove an existing '
                            'subclassifier')
            return

        clf = self.get_classifier_by_cell_type(
            cell_type, search_subclassifiers=search_subclassifiers)        

        try:
            other_subclf = clf.sub_classifiers[cell_type]
        except KeyError:
            other_subclf = None

        if other_subclf is not None:
            # a subclassifier for the specified cell type already exists
            _LOGGER.warning(
                'Replacing existing subclassifier for cell type "%s".',
                cell_type)

        clf.sub_classifiers[cell_type] = subclf
        _LOGGER.info('Added subclassifier for cell type "%s"', cell_type)


    def get_subclassifier(self, cell_type: str, must_exist: bool = False,
                          search_subclassifiers: bool = True):
        """Retrieves the subclassifier for a given cell type.
        
        Optionally raises an exception if no classifier is found.
        Always raises an exception if the specified cell type is unknown.
        """

        clf = self.get_classifier_by_cell_type(
            cell_type, search_subclassifiers=search_subclassifiers)

        subclf = None
        try:
            subclf = clf.sub_classifiers[cell_type]
        except KeyError:
            pass

        if subclf is None and must_exist:
            raise ValueError(
                'Cell type "%s" has no subclassifier!' % cell_type)

        return subclf        
            

    def remove_subclassifier(self, cell_type: str,
                             search_subclassifiers: bool = True) -> None:
        """Removes the subclassifier for a given cell type.

        An exception is raised if no subclassifier exists for this cell type.
        """
        clf = self.get_classifier_by_cell_type(
            cell_type, search_subclassifiers=search_subclassifiers)

        subclf = None
        try:
            subclf = clf.sub_classifiers[cell_type]
        except KeyError:
            pass

        if subclf is None:
            raise ValueError(
                'Cell type "%s" has no subclassifier!' % cell_type)

        del clf.sub_classifiers[cell_type]
        _LOGGER.info('Removed subclassifier for cell type "%s".' % cell_type)


    #def predict_proba(self, matrix: ExpMatrix,
    #                  is_smoothed: bool = False) -> CellAnnMatrix:
    #    """Predict cell type probabilities."""

    #    if self.svm_model_ is None:
    #        raise RuntimeError('You must fit the model first!')

    #    # 1. Perform smoothing
    #    if not is_smoothed:
    #        smoothed_matrix = pp.knn_smoothing(
    #            matrix, self.k, d=self.d, dither=self.dither,
    #            seed=self.seed)
    #    else:
    #        smoothed_matrix = matrix
        
    #    # 2. Apply PCA
    #    tmatrix = self.transform(smoothed_matrix)
        
    #    # 3. Predict cell type probabilities
    #    Y = self.svm_model_.predict_proba(tmatrix.T)
    #    clusters_sorted = self.value_counts_.index.sort_values()
    #    pred = CellAnnMatrix(cells=matrix.cells, columns=clusters_sorted,
    #                         data=Y)
    #    return pred


    def write_pickle(self, file_path: str) -> None:
        """Write classifier to file in pickle format."""
        #pred.write_pickle('')
        with open(file_path, 'wb') as ofh:
            pickle.dump(self, ofh, self.PICKLE_PROTOCOL_VERSION)
        _LOGGER.info('Wrote Moana classifier to "%s".', file_path)


    @classmethod
    def read_pickle(cls, file_path: str):
        """Read classifier from pickle file."""
        with open(file_path, 'rb') as fh:
            clf = pickle.load(fh)
        _LOGGER.info('Loaded Moana classifier from "%s".', file_path)
        return clf
