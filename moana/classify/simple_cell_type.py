# Author: Florian Wagner <florian.wagner@nyu.edu>
# Copyright (c) 2018, New York University
#
# This file is part of Moana.

"""Single-cell classifier for smoothed scRNA-Seq data."""

import logging
import pickle
from typing import Dict, Optional, Tuple, Iterable, Union, List
import time
import copy
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.svm import NuSVC
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

from .. import preprocess as pp
from .. import tools
from .. import visualize
from ..core import ExpMatrix, CellAnnVector, CellAnnMatrix
from ..util import get_sel_components, get_component_labels
from .util import apply_pca, get_precision_summary

_LOGGER = logging.getLogger(__name__)


class SimpleCellTypeClassifier:

    PICKLE_PROTOCOL_VERSION = 4  # requires Python 3.4 or higher

    """A cell type classifier for smoothed scRNA-Seq data."""
    def __init__(self, d: int = 10,
                 nu: float = 0.20, seed: int = 0,
                 sub_classifiers = None,
                 name: str = None) -> None:

        #sel_components = get_sel_components(d)
        #n_components = max(sel_components) + 1
        #if n_components > d:
        #    raise ValueError('The highest selected principal component (%d) '
        #                     'cannot be higher than "d" (%d).'
        #                     % (n_components, d))

        if sub_classifiers is None:
            sub_classifiers = {}

        self.name = name
        self.d = d
        #self.sel_components = sel_components
        self.nu = nu
        self.seed = seed
        self.sub_classifiers = sub_classifiers
        
        self.transcript_count_ = None
        self.pca_model_ = None
        self.genes_ = None
        self.cell_labels_ = None
        self.svm_model_ = None


    @property
    def sel_components(self) -> List[int]:
        return get_sel_components(self.d)

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
                % (str(self.transcript_count_)),
            '- nu = %s' % str(self.nu),
            #'- sel_components = %s' % str(self.sel_components),
            '- seed = %d' % self.seed])

        if self.svm_model_ is None:
            header_str = ('Moana cell type classifier (**untrained**)\n'
                          '------------------------------------------')
        else:
            header_str = ('Moana cell type classifier\n'
                          '--------------------------')

        clf_str = ('%s\n' % header_str + 
                   'Name: %s\n\n' % name_str + 
                   'Parameters:\n'
                   '%s' % param_str)

        if self.svm_model_ is None:
            return clf_str

        ctype_str = self._get_ctype_str()

        msg = ('%s\n\n' % clf_str +
               'Cell types / subtypes:\n'
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
            subtype_str = '%s %s' % (prefix, ctype)
            if include_subtypes:
                subclf = self.get_subclassifier(ctype, search_subclassifiers=False)
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
        data = self.pca_model_.components_.T[:, self.sel_components]
        dim_labels = get_component_labels(self.sel_components)
        loadings = ExpMatrix(genes=self.genes_, cells=dim_labels, data=data)
        return loadings.copy()


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
            normalized_gene_coef.iloc[:, 0] *= factor
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
            cell_labels: CellAnnVector) -> None:
        """Train a cell classifier for scRNA-Seq data.
        
        """

        sublogger = logging.getLogger('moana.preprocess')
        prev_level = sublogger.level
        sublogger.setLevel(logging.WARNING)

        if not cell_labels.index.to_series().isin(matrix.cells).all():
            raise ValueError('Not all cells in cell type vector are '
                             'contained in the expression matrix!')

        if set(matrix.cells) != set(cell_labels.cells):
            _LOGGER.warning('Cell type vector and expression matrix do not '
                            'contain the same set of cells!')

        # make sure the two datasets are aligned
        matrix = matrix.loc[:, cell_labels.index]

        # determine median transcript count of smoothed matrix
        transcript_count = float(matrix.sum(axis=0).median())

        ### perform PCA
        _LOGGER.info('Moana training -- Performing PCA...')

        # normalize matrix
        matrix = pp.median_normalize(matrix)

        # perform PCA
        tmatrix, pca_model = pp.pca(
            matrix, self.num_components, seed=self.seed)

        # select specific principal components
        # (currently we always select all the PCs)
        tmatrix = tmatrix.iloc[self.sel_components]

        # report fraction of variance explained
        frac_variance_explained = \
                pca_model.explained_variance_ratio_[self.sel_components].sum()
        _LOGGER.info('Moana training -- The selected PCs represent %.1f %% of '
                     'total variance.', 100*frac_variance_explained)

        # set training variables
        self.genes_ = matrix.genes.copy()
        self.transcript_count_ = transcript_count
        self.pca_model_ = pca_model
        self.cell_labels_ = cell_labels.copy()

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
        self.svm_model_ = NuSVC(nu=self.nu, kernel='linear',
                                decision_function_shape='ovo',
                                random_state=self.seed)  # intialize the model

        self.svm_model_.fit(train_tmatrix.T, train_labels)  # train the model

        # report performance on training data
        predictions = self.predict(matrix)
        precision_summary = get_precision_summary(cell_labels, predictions)
        _LOGGER.info('Moana training -- SVM classifier performance (precision) '
                     'on training data: %s', precision_summary)

        sublogger.setLevel(prev_level)


    def transform(self, matrix: ExpMatrix) -> ExpMatrix:
        """Project a matrix into the PC space defined by the training data."""

        if self.svm_model_ is None:
            raise RuntimeError('You must train the classifier first!')

        tmatrix = apply_pca(
            matrix, self.pca_model_, self.transcript_count_,
            components=self.sel_components, valid_genes=self.genes_)

        return tmatrix


    def predict(self, matrix: ExpMatrix,
                predict_subtypes: bool = True) -> CellAnnVector:
        """Predict cell types."""

        if self.svm_model_ is None:
            raise RuntimeError('You must train the classifier first!')

        t0 = time.time()

        sublogger = logging.getLogger('moana.preprocess.smooth')
        prev_level = sublogger.level
        sublogger.setLevel(logging.WARNING)

        sublogger2 = logging.getLogger('moana.classify.util')
        prev_level2 = sublogger2.level
        sublogger2.setLevel(logging.WARNING)

        _LOGGER.info(
            'Moana prediction -- This classifier was trained using d=%d, '
            'nu=%s, at %s transcripts / cell.',
            self.d, str(self.nu), str(self.transcript_count_))


        ### 1. Apply PCA
        _LOGGER.info(
            'Moana prediction -- Data will be scaled %.2f-fold before '
            'FT-transformation and projection into PC space.',
            self.transcript_count_ / matrix.sum(axis=0).median())
        tmatrix = apply_pca(
            matrix, self.pca_model_, self.transcript_count_,
            components=self.sel_components, valid_genes=self.genes_)

        ### 3. Predict cell type
        y = self.svm_model_.predict(tmatrix.T)
        predictions = CellAnnVector(cells=matrix.cells, data=y)
        predictions.name = 'Predicted cell type'

        result_str = '; '.join(
            '%s - %d' % (ctype, n)
            for ctype, n in predictions.value_counts().iteritems())
        _LOGGER.info('Moana prediction -- Prediction results: %s', result_str)

        # 4. Apply subclassifiers (if there are any)
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

        sublogger.setLevel(prev_level)
        sublogger2.setLevel(prev_level2)

        t1 = time.time()
        _LOGGER.info('Cell type prediction took %.1f s.', t1-t0)

        return predictions


    def get_decision_function(
            self, matrix: ExpMatrix,
            cell_labels: CellAnnVector):
        """Calculate cell (average) decision func. values for their cluster."""
        
        if self.svm_model_ is None:
            raise RuntimeError('You must train the classifier first!')

        # 2. Apply PCA (will select same genes)
        tmatrix = apply_pca(
            matrix, self.pca_model_, self.transcript_count_,
            components=self.sel_components, valid_genes=self.genes_)

        num_samples = cell_labels.size
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
        for cell, ctype in cell_labels.items():
            dfun.loc[cell] = dfun_avg.loc[cell, ctype]
        
        return dfun


    @property
    def has_subclassifiers(self):
        """Determines if the classifier has any subclassifiers."""
        return any([self.get_subclassifier(ctype) is not None
                   for ctype in self.cell_types_])


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


    # def predict_proba(self, matrix: ExpMatrix) -> CellAnnMatrix:
    #    """Predict cell type probabilities."""

    #    if self.svm_model_ is None:
    #        raise RuntimeError('You must fit the model first!')

    #    # 1. Apply PCA
    #    tmatrix = self.transform(matrix)
        
    #    # 2. Predict cell type probabilities
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
