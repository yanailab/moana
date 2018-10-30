import os
import pkg_resources

from . import core
from . import tools
from . import qc
from . import preprocess
from . import cluster
from . import visualize
from . import classify
from . import ensembl
from . import util

__version__ = pkg_resources.require('moana')[0].version

_root = os.path.abspath(os.path.dirname(__file__))
