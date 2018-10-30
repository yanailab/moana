## Moana: A robust and scalable cell type classification framework for single-cell RNA-Seq data

This repository contains a Python implementation of the Moana framework for predicting cell types in single-cell RNA-Seq data ([Wagner and Yanai, 2018](https://www.biorxiv.org/content/early/2018/04/09/217737)).

### Installation

#### Step 1: Install Python and the Python packages that Moana depends on

Moana requires Python version 3.5, 3.6, or 3.7, as well as the Python packages *pandas*, *scikit-learn*, and *plotly*.

The easiest way to install Python as well as these packages is to [download and install Anaconda](https://www.anaconda.com/download). Anaconda is a distribution of Python that already includes a lot of packages, including pandas, scikit-learn, and plotly. Alternatively, you can [download and install Miniconda](https://conda.io/miniconda.html), and use the *conda* command to create a new Python 3 environment and install the required packages. The latter option takes up less disk space but also requires some knowledge of how to use conda, the package/environment manager that underlies both Anaconda and Miniconda.

#### Step 2: Install Moana

To install Moana, make sure you have activated/selected the correct conda environment, and then type:

```console
$ pip install moana
```

### Changelog

#### 10/30/2018 - Version 0.1.1 released

This is the initial release of **Moana**. We used this version to construct the cell type classifiers described in our preprint. Additional documentation is forthcoming.
