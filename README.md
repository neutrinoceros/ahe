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
- contrast limitation, deemed an essential feature, is currently missing
- documentation is incomplete
- binaries are not published


## Installation

> [!WARNING]
> pre-built binaries (wheels) are not published at this stage.
> A rust compiler toolchain is needed in order to install this package.

```
$ python -m pip install ahe
```

## Usage

### Simple Histogram Equalization

We'll start by defining an image composed of noise.

```python
import ahe
import numpy as np

image_shape = (128, 128)
prng = np.random.default_rng(0)
image = np.clip(
    prng.normal(
        loc=0.5,
        scale=0.25,
        size=np.prod(image_shape),
    ).reshape(image_shape),
    a_min=0.0,
    a_max=1.0,
)
```

Non-adaptive histogram equalization is performed as follow
```python
image_eq = ahe.equalize_histogram(image)
```
This method is least expensive in terms of strain put on hardware resources.
However, as a single histogram is computed and adjusted over the entire image,
this technique is known to amplify noise in near-uniform regions.

### Adaptive Histogram equalization (AHE)
Adaptive Histogram Equalization (AHE) was designed to overcome this limitation by
instead computing more localized (and numerous) histograms, improving the *local*
contrast in all regions, at the cost of a reduced efficiency.
As illustrated in the following, there are two main variants of AHE, sliding-tile and
tile-interpolation. As the names suggest, both methods rely on the use of of tiles,
also known as *contextual regions*, that define sub-domains in which different histograms
are computed and applied.

#### Prioritizing accuracy: sliding-tile

True AHE is intrinsically an expensive operation to perform, as it requires computing
a different histogram *per pixel*. The least inefficient way to accomplish this,
originally proposed by [Pizer et al. (1987)](#references), reduces the redundancy in
intermediate computations and is known as the sliding-tile variant of AHE.
Here's how to use it in `ahe`

```python
image_eq = ahe.equalize_histogram(
    image,
    adaptive_strategy={
        'kind': 'sliding-tile',
        'tile-size': 15,
    },
)
```

> [!NOTE]
> This strategy requires odd-sized tile shapes, but supports
> image shapes with any parity.

While an exact implementation of AHE, this option remains resource-demanding and is
not recommended for production.

#### Prioritizing performance: tile-interpolation

Alternatively, very similar results can be obtained at a fraction of the cost using
an approximative method known as the tile-interpolation variant of AHE, also
introduced by [Pizer et al. (1987)](#references).
In this method, an image is split into equal-sized sub domains (tiles), which may be
specified either from a tile size

```python

image_eq = ahe.equalize_histogram(
    image,
    adaptive_strategy={
        'kind': 'tile-interpolation',
        'tile-size': 16,
    },
)
```

or as a number of tiles to split the domain into (in each direction)
```python
image_eq = equalize_histogram(
    image,
    adaptive_strategy={
        'kind': 'tile-interpolation',
        'tile-into': 8,
    },
)
```

> [!NOTE]
> This strategy requires even-sized tile and image shapes.

### General rules for tiling schemes

In all AHE strategies, all tiles created will be the exact same size, regardless
of the pixel's relative position in the image. The whole domain is generally padded
internally in order to respect this rule. The exact method used for padding is
controlled by the `boundaries` keyword argument.

Both `'tile-size'` and `'tile-into'` will accept either a shape as a pair of
integers `(n, m)`, or a single integer `n`, which is a shorthand for `(n, n)`, as
illustrated above.


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

(*: `numpy` itself, as the common dependency to `ahe` and `scikit-image`, is
excluded from this graph)

`ahe.equalize_histogram` supports more tiling scheme than
`skimage.exposure.equalize_hist` and `skimage.exposure.equalize_adapthist` combined,
 within a consistent interface and a unified feature set.
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

`ahe.equalize_histogram` also supports periodic boundary conditions, which can be
specified as `boundaries='periodic'`.

> [!IMPORTANT]
> TODO
>- better performance

### How
> [!IMPORTANT]
> TODO


## References

1. Pizer, Stephen M. et al. (1987). Adaptive Histogram Equalization and Its Variations. *Compute Vizion, Graphics, and Image Processing*, 39, 355-368
