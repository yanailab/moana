from typing import Union, Iterable

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from .data import *
from .log import *

Int = Union[int, np.integer]

def get_sel_components(components: Union[Int, Iterable[Int]]) -> List[int]:
    """Get a list of selected principal components (0-based indexing)."""

    if isinstance(components, (int, np.integer)):
        return list(range(components))
    else:
        return list(components)


def get_component_labels(
        sel_components: Iterable[int], pca_model: PCA = None) -> List[str]:
    """Generate labels for principal components."""

    if pca_model is not None:
        labels = ['PC %d (%.1f %%)'
                  % (c+1, 100*pca_model.explained_variance_ratio_[c])
                  for c in sel_components]
    else:
        labels = ['PC %d' % (c+1) for c in sel_components]

    return labels
