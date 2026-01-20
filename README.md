# NApyPI: Efficient statistics in Python for large-scale heterogeneous data with enhanced support for missing data
![Tests](https://github.com/fabiwoller/NApyPI/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20–%203.11-orange)
![PyPI](https://img.shields.io/pypi/v/napypi)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fgigascience%2Fgiaf140-red)](https://doi.org/10.1093/gigascience/giaf140)


A python packaged version of our software NApy. NApy offers a fast python tool providing statistical tests and effect sizes for a more comprehensive and informative analysis of mixed type data in the presence of missingness. Written both in C++ and numba and parallelized with OpenMP.

## Installation

NApy is available as a Python package on the most common Windows, MacOS, and Linux architectures (64-bit only). It is easily installable via:

```bash
pip install napypi
```

## Documentation

For a detailed overview of NApy's functionality and parameter descriptions, we refer to NApy's [main repository](https://github.com/DyHealthNet/NApy).

## Citation

In case you find our tool useful, please cite our corresponding [manuscript](https://doi.org/10.1093/gigascience/giaf140):

Fabian Woller, Lis Arend, Christian Fuchsberger, Markus List, David B Blumenthal, NApy: Efficient Statistics in Python for Large-Scale Heterogeneous Data with Enhanced Support for Missing Data, GigaScience, 2025; giaf140, https://doi.org/10.1093/gigascience/giaf140

```
@article{10.1093/gigascience/giaf140,
    author = {Woller, Fabian and Arend, Lis and Fuchsberger, Christian and List, Markus and Blumenthal, David B},
    title = {NApy: Efficient Statistics in Python for Large-Scale Heterogeneous Data with Enhanced Support for Missing Data},
    journal = {GigaScience},
    pages = {giaf140},
    year = {2025},
    month = {11},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf140},
    url = {https://doi.org/10.1093/gigascience/giaf140},
}
```

