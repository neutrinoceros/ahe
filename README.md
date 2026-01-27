# `ahe`
[![PyPI](https://img.shields.io/pypi/v/ahe.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/ahe/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


A minimalist Python library for Adaptive Histogram Equalization,
combining the expressiveness of a user-friendly Python interface with the raw power of
a low-level implementation.

<!-- TOC -->

- [Development status](#development-status)
- [Installation](#installation)
- [Usage](#usage)
    - [Simple Histogram Equalization](#simple-histogram-equalization)
    - [Adaptive Histogram equalization AHE](#adaptive-histogram-equalization-ahe)
        - [Prioritizing accuracy: sliding-tile](#prioritizing-accuracy-sliding-tile)
        - [Prioritizing performance: tile-interpolation](#prioritizing-performance-tile-interpolation)
    - [General rules for tiling schemes](#general-rules-for-tiling-schemes)
- [Migrating from scikit-image](#migrating-from-scikit-image)
    - [TL;DR](#tldr)
    - [Disclaimer: missing features](#disclaimer-missing-features)
    - [Dependency Minimalism](#dependency-minimalism)
    - [Better performance](#better-performance)
    - [Interface Consistency](#interface-consistency)
    - [Conservation of transformation invariants](#conservation-of-transformation-invariants)
    - [Additional features](#additional-features)
    - [Migration Guide](#migration-guide)
        - [HE](#he)
        - [AHE with implicit kernel size](#ahe-with-implicit-kernel-size)
        - [CLAHE with explicit kernel size](#clahe-with-explicit-kernel-size)
- [References](#references)

<!-- /TOC -->

## Development status

> [!WARNING]
> `ahe` is pre-alpha software

`ahe` is developed in the open, but not stable yet.

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
a different histogram *per pixel*. One efficient (although still costly) way to
accomplish this, originally proposed by [Pizer et al. (1987)](#references), reduces the
redundancy in intermediate computations and is known as the sliding-tile variant of AHE.
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

> [!IMPORTANT]
> There's a known defect with some edge cases
> https://github.com/neutrinoceros/ahe/issues/50

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

In all AHE strategies, all tiles created will be the exact same size, regardless of the
pixel's relative position in the image. The whole domain is generally padded internally
in order to respect this rule. The exact method used for padding is controlled by the
`boundaries` keyword argument.

Both `'tile-size'` and `'tile-into'` will accept either a shape as a pair of integers
`(n, m)`, or a single integer `n`, which is a shorthand for `(n, n)`, as illustrated
above.


## Migrating from `scikit-image`

### TL;DR
Put simply, if all your project needs from `scikit-image` is
`skimage.exposure.equalize_(adapt)hist`, `ahe` provides a faster, more lightweight and
portable replacement. The following contains a much more detailed explanation of the
differences. You can also jump to the final [Migration Guide](#migration-guide).

### Disclaimer: missing features
The following features from `skimage.exposure.equalize_(adapt)hist` are currently
missing from `ahe`:
- [masking](https://github.com/neutrinoceros/ahe/issues/51)
- multi-channel images objects (from `PIL`) or arrays. The expectation is that this
  should be easy to re-implement on the user side.
- higher dimensionality: `ahe` currently only supports 2D input and outputs. The
  algorithms can be generalized to work in any dimensionality. Please open an issue

Some, though not all of the above are already planned. Don't hesitate to request
the others (or anything else that could be in scope) by opening an issue.

### Dependency Minimalism
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

### Better performance
<!-- Generated with `uv run scripts/doc_graphs.py` -->
<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/benchmark-dark.svg" width="900">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/benchmark-light.svg" width="900">
    <img alt="Shows a bar chart comparing wheel sizes" src="https://raw.githubusercontent.com/neutrinoceros/ahe/main/assets/benchmark-light.svg" width="900">
  </picture>
</p>


### Interface Consistency
`scikit-image`'s implementation of histogram equalization methods are exposed as two
different functions: `skimage.exposure.equalize_hist` and
`skimage.exposure.equalize_adapthist`. Only the former supports masking, and only the
latter supports clipping. `ahe.equalize_histogram` provides a consistent feature set,
independent of the adaptive strategy (or lack thereof) selected.

Furthermore, implicit, default behavior can be hard to reproduce explicitly. For
instance, `equalize_adapthist` will, by default, create tiles by dividing the image in 8
along each direction, but only exposes a `tile_size` argument to override this; as a
result, one needs to re-implement division logic if they need something very similar to
the default, but with any other value than 8. In stark contrast,
`ahe.equalize_histogram`'s `adaptive_strategy` argument supports all these applications:
- `adaptive_strategy=None` corresponds to `skimage.exposure.equalize_hist`
- `adaptive_strategy={'kind': 'tile-interpolation', 'tile-into': 8}` corresponds to
  `skimage.exposure.equalize_adapthist`'s default, but the exact divisor(s) used can
   easily be adjusted
- `adaptive_strategy={'kind': 'tile-interpolation', 'tile-size': 64}` is akin to using
  `skimage.exposure.equalize_adapthist`'s `tile_size` argument.

Last but not least, `equalize_hist` does not support contrast limitation, while
`equalize_adapthist` enables it by default (`clip_limit` defaults to `0.01`), and
things get messy when you actually want to *disable* it:
- `clip_limit`'s name is not descriptive and only makes sense if you already know what
  it does under the hood. `ahe`'s equivalent parameter is named
  `max_normalized_bincount`, which is more verbose, but also more explicit about what
  the number represents.

- `clip_limit` (a.k.a `max_normalized_bincount`) is effectively a fraction; only values
  within the open interval `]0.0, 1.0]` are meaningful, *but* `clip_limit=0.0` is
  *allowed*, and effectivaly disable all clipping, which means it's equivalent to `1.0`,
  further mystifying the underlying behavior and meaning of the parameter. Another
  way to phrase this is that the results change discontinuously at `0.0`, which is very
  close to the default value *and* should be easy to reason about.
- results for `clip_limit=1.0` (*or* `0.0`, as we just saw), are actually *incorrect* at
  a level that is visible to the naked eye (aliasing may be prominent). In comparison,
  `ahe` does not enable contrast limitation by default: such flagrant defects would be
  immediately visible in tests.

### Conservation of transformation invariants
`ahe.equalize_histogram` also provides stricter guarantees regarding the
transformation's geometric invariants.
Outputs are guaranteed to be invariant (to machine precision) to left/right and top/
bottom symmetries. In contrast, `skimage.exposure.equalize_adapthist`'s outputs are
subject to biases on, because it does not enforce symmetry in its internal tiling scheme
(as of `scikit-image` `v0.26.0`). This improved tiling scheme comes at the cost of
stricter requirements in `ahe.equalize_histogram`: the tile-interpolation strategy only
supports tiles and images with even sizes in both directions.


### Additional features
`ahe.equalize_histogram` supports more tiling scheme than
`skimage.exposure.equalize_hist` and `skimage.exposure.equalize_adapthist` combined,
 within a consistent interface and a unified feature set.
In particular, it offers an exact implementation of Adaptive Histogram Equalization
implemented as a sliding-tile, while `skimage.exposure.equalize_adapthist` only
supports tile-interpolation (*also* available in `ahe`), which is generally faster,
but also a less accurate approximation of a true AHE.

`ahe.equalize_histogram` also supports periodic boundary conditions, which can be
specified as `boundaries='periodic'`.


### Migration Guide
This section provides some practical examples of `scikit-image` applications and their
equivalent in `ahe`.

Notes
- in `ahe`, the default `nbins` is *generally* aligned with `scikit-image`'s (256),
  except for kernels (or tiles) spanning less than 256 pixels. For maximu compatibility,
  specifying an explicit value is recommended.
- `ahe.equalize_histogram`'s `max_normalized_bincount` represents the same parameter as
  `skimage.exposure.equalize_adapthist`'s `clip_limit`, but their default values differ:
  `max_normalized_bincount` defaults to `1.0` (no contrast limitation), while
  `scikit-image`'s `clip_limit` default to `0.01`
- "kernel", "tile" and "contextual region" are different names for the same concept.

#### HE
```python
from skimage.exposure import equalize_hist

result = equalize_hist(array)

# becomes
import ahe

result = ahe.equalize_histogram(array, nbins=256)
```

#### AHE with implicit kernel size
```python
from skimage.exposure import equalize_adapthist

result = equalize_adapthist(array, clip_limit=1.0)

# becomes
import ahe

result = ahe.equalize_histogram(
    array,
    nbins=256,
    adaptive_strategy={
      "kind": "tile-interpolation",
      "tile-into": 8, # or (8, 8)
    },
)
```

#### CLAHE with explicit kernel size
```python
from skimage.exposure import equalize_adapthist

result = equalize_adapthist(array, kernel_size=64)

# becomes
import ahe

result = ahe.equalize_histogram(
    array,
    nbins=256,
    adaptive_strategy={
      "kind": "tile-interpolation",
      "tile-size": 64, # or (64, 64)
    },
    max_normalized_bincount=0.01,
)
```

## References

1. Pizer, Stephen M. et al. (1987). Adaptive Histogram Equalization and Its Variations.
  *Compute Vizion, Graphics, and Image Processing*, 39, 355-368
