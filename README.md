# `ahe`
[![PyPI](https://img.shields.io/pypi/v/ahe.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/ahe/)
<!--
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ahe.svg?logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/ahe)
-->
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


A minimalist Python library for Adaptive Histogram Equalization,
combining the expressiveness of a user-friendly Python interface with the raw power of
a low-level implementation.

## Development status

> [!WARNING]
> pre-alpha software

## Installation

> [!IMPORTANT]
> TODO

## Usage

> [!IMPORTANT]
> TODO

## Migrating from `scikit-image`
### Why

Put simply, if all your project needs from `scikit-image` is `skimage.exposure.equalize_(adapt)hist`, `ahe` provides a much more lightweight and portable replacement.

`ahe` has no runtime dependencies beyond `numpy`. Additionally, its binaries are orders of magnitude lighter than `scikit-image`'s, as well as future-compatible with yet-unreleased versions of Python.

<!-- Generated with `uv run scripts/doc_graphs.py` -->
<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/test-b/assets/wheel-size-dark.svg" width="900">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/test-b/assets/wheel-size-light.svg" width="900">
    <img alt="Shows a bar chart comparing wheel sizes" src="https://raw.githubusercontent.com/neutrinoceros/ahe/test-b/assets/wheel-size-dark.svg" width="900">
  </picture>
</p>

> [!IMPORTANT]
> TODO

- better performance
- improved guarantees on transformation invariants

### How
> [!IMPORTANT]
> TODO
