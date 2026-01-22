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
> `ahe` is pre-alpha software

`ahe` is developed in the open, but currently unstable.
- essential features are missing:
  - boundary conditions
  - contrast limitations
- documentation, examples and references are currently lacking
- binaries are not published


## Installation

> [!WARNING]
> pre-built binaries (wheels) are not published at this stage.
# A rust compiler toolchain is needed in order to install this package.

```
$ python -m pip install ahe
```

## Usage

> [!IMPORTANT]
> TODO

## Migrating from `scikit-image`
### Why

Put simply, if all your project needs from `scikit-image` is
`skimage.exposure.equalize_(adapt)hist`, `ahe` provides a much more lightweight
and portable replacement.

`ahe` has no runtime dependencies beyond `numpy`. Additionally, its binaries are
orders of magnitude lighter than `scikit-image`'s, as well as future-compatible
with yet-unreleased versions of Python.

<!-- Generated with `uv run scripts/doc_graphs.py` -->
<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/wheel-size-dark.svg" width="900">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/wheel-size-light.svg" width="900">
    <img alt="Shows a bar chart comparing wheel sizes" src="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/wheel-size-light.svg" width="900">
  </picture>
</p>

(*: `numpy` itself, as the common dependency to `ahe` and `scikit-image` is
excluded from this graph)

`ahe.equalize_histogram` supports more tiling scheme than
`skimage.exposure.equalize_hist` and `skimage.exposure.equalize_adapthist` combined.
In particular, it offers an exact implementation of Adaptive Histogram Equalization
implemented as a sliding-tile, while `skimage.exposure.equalize_adapthist` only
supports tile-interpolation (*also* available in `ahe`), which is generally faster,
but also a less accurate approximation of a true AHE.

`ahe.equalize_histogram` also provides stricter guarantees regarding the
transformation's geometric invariants.
Outputs are guaranteed to be invariant (to machine precision) to left/right and top/
bottom symmetries. In contrast, `skimage.exposure.equalize_adapthist`'s outputs are
subject to biases on, because it does not enforce symmetry in its internal tiling
scheme (as of `scikit-image` `v0.26.0`). This improved tiling scheme comes at the
cost of stricter requirements in `ahe.equalize_histogram`: the tile-interpolation
strategy only supports tiles and images with even sizes in both directions.


> [!IMPORTANT]
> TODO
>- better performance
>- boundary conditions

### How
> [!IMPORTANT]
> TODO
