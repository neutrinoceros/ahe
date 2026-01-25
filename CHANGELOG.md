# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- FEAT: add support for contrast limitation through the `max_normalized_bincount`
  keyword argument
- ENH: reduce padding used in tile-interpolation AHE to half a tile instead of a full one
- DOC: add a graph showcasing a comparison of performance with `scikit-image`
- DOC: complete migration guide

## 0.0.3 - 2026-01-23

This is an early access, pre-alpha release. Some features are missing,
only a source distribution is provided, and documentation is a work in progress.

- FEAT: add support for all-directions periodic boundary conditions
- BUG: prevent a rust panic when input image contains non-finite (inf, NaN) values,
       raise a clear Python exception instead.
- DOC: prefer light-themes graphs when undetermined (improve PyPI rendering)
- DOC: document technical differences with `scikit-image`, other than binary size and performance
- DOC: add usage examples to narrative documentation

## 0.0.2 - 2026-01-22

This is an early access, pre-alpha release. Some features are missing,
only a source distribution is provided, and documentation is a work in progress.

- BUG: fix directional bias and overall incorrect scaling factors in tile-interpolation AHE
- BUG: fix a bug where an exception would be raised when trying to specify a tile-interpolation AHE with 'tile-into'
- DOC: start documenting migration guide
- DOC: document dev status and installation


## 0.0.1 - 2026-01-18

This is an early access, pre-alpha release. Some features are missing,
tile-interpolation is known to be broken, only a source distribution is
provided, and there is no documentation yet.
