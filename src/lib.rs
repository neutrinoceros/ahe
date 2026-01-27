use interpn::one_dim::linear::LinearHoldLast1D;
use interpn::one_dim::Interp1D;
use interpn::RegularGrid1D;
use num_traits::{Float, NumCast, Signed};

use numpy::borrow::PyReadonlyArray2;
use numpy::ndarray::{s, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Dimension};
use numpy::{PyArray2, ToPyArray};
use pyo3::types::PyModuleMethods;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};
use std::collections::VecDeque;
use std::ops::{AddAssign, Mul, Sub};

#[derive(Clone, Copy, PartialEq)]
struct ArrayDimensions {
    // can be used to represent the size of an image or view
    x: usize,
    y: usize,
}

#[derive(Clone, Copy, PartialEq)]
struct PixelIndex {
    i: usize, // y
    j: usize, // x
}

trait AtLeastF32: Float + From<f32> + Signed + AddAssign<<Self as Mul>::Output> {}
impl AtLeastF32 for f32 {}
impl AtLeastF32 for f64 {}

#[derive(Clone, Copy, PartialEq)]
struct Range<T> {
    lo: T,
    hi: T,
}

impl<T: Sub<Output = T> + Copy> Range<T> {
    fn span(&self) -> T {
        self.hi - self.lo
    }
}

struct Histogram<T> {
    bins: Array1<usize>,
    range: Range<T>,
}

impl<T: AtLeastF32 + NumCast> Histogram<T> {
    fn bin_width(&self) -> T {
        self.range.span() / <T as NumCast>::from(self.bins.len()).unwrap()
    }

    fn first_bin_center(&self) -> T {
        self.range.lo + self.bin_width() / 2.0.into()
    }

    fn cdf(&self) -> Array1<usize> {
        // convert histogram to normalized cumulative distribution function
        let mut cdf = self.bins.clone();
        for i in 1..cdf.len() {
            cdf[i] += cdf[i - 1];
        }
        cdf
    }

    fn cdf_as_normalized(&self) -> Array1<T> {
        let cdf_us = self.cdf();
        let nbins = self.bins.len();
        let mut cdf: Array1<T> = cdf_us.mapv(|elem| <T as NumCast>::from(elem).unwrap());
        let tot = cdf[nbins - 1];
        for i in 0..nbins {
            cdf[i] = cdf[i] / tot;
        }
        cdf
    }
}

fn compute_subhistogram<T: AtLeastF32>(
    arr: ArrayView1<T>,
    range: Range<T>,
    nbins: usize,
) -> Histogram<T> {
    let bin_width = range.span() / <T as NumCast>::from(nbins).unwrap();

    // padding one extra bin on the right allows for a branchless optimization:
    // pixels that contain exactly vmax are counted in the extra bin
    let mut bins = Array1::<usize>::zeros(nbins + 1);
    for v in arr.iter() {
        let f = ((*v - range.lo) / bin_width).floor();
        let idx = <usize as NumCast>::from(f).unwrap();
        bins[idx] += 1;
    }

    // move data from the last bin to the previous one
    bins[nbins - 1] += bins[nbins];
    let bins = bins.slice(s![..-1]).to_owned();
    assert_eq!(bins.len(), nbins);

    Histogram { bins, range }
}

fn reduce_histogram<T: Copy>(
    subhists: &VecDeque<Histogram<T>>,
    max_bincount: usize,
) -> Histogram<T> {
    // it is assumed that all input histograms have the exact same range and nbins
    let h0 = subhists.front().unwrap();
    let nbins = h0.bins.len();
    let mut bins = Array1::<usize>::zeros(nbins);
    for h in subhists {
        for i in 0..nbins {
            bins[i] += h.bins[i];
        }
    }
    // contrast limitation -- histogram clipping

    // 1. compute effective max_bincount (generally lower than input)
    // see Pizer et al. (1987), Section 4.2, Program 3 (bisection)
    let max_bincount = {
        let mut top = max_bincount;
        let mut bot = 0usize;
        let mut excess: usize;
        while (top - bot) > 1 {
            let mid = (top + bot) / 2;
            excess = 0;
            for b in &bins {
                excess += {
                    if *b > mid {
                        b - mid
                    } else {
                        0
                    }
                };
            }
            if excess > (max_bincount - mid) * nbins {
                top = mid;
            } else {
                bot = mid;
            }
        }
        // I'm intentionally returning top instead of bot(tom), as prescribed by
        // Pizer et al. (1987), to avoid spurious clipping in perfectly uniform regions.
        top
    };
    // compute excess one last time (don't reuse previous result as it may not be correct)
    let mut excess = 0usize;
    for b in &bins {
        if *b > max_bincount {
            excess += b - max_bincount;
        };
    }

    if excess > 0 {
        // 2. clip excess
        for i in 0..nbins {
            bins[i] = bins[i].max(max_bincount);
        }

        // 3. uniformly re-distribute excess
        let r = excess % nbins;
        let d = excess / nbins + {
            // hack: prioritize exact uniformity (and simplicity of impl)
            // over exact conservation of total bincount
            if r > nbins / 2 {
                1
            } else {
                0
            }
        };
        for i in 0..nbins {
            bins[i] += d;
        }
    }

    Histogram {
        bins,
        range: h0.range,
    }
}

fn compute_histogram<T: AtLeastF32 + numpy::Element + NumCast>(
    image: ArrayView2<T>,
    nbins: usize,
    max_bincount: usize,
) -> Histogram<T> {
    let range = get_value_range(image);
    let mut subhistograms = VecDeque::with_capacity(image.shape()[0] + 1);
    for row in image.axis_iter(Axis(0)) {
        subhistograms.push_back(compute_subhistogram(row, range, nbins));
    }
    reduce_histogram(&subhistograms, max_bincount)
}

#[cfg(test)]
mod test_histogram {
    use crate::{Histogram, Range};
    use numpy::ndarray::Array1;

    #[test]
    fn test_ones() {
        let nbins = 8usize;
        let hist = Histogram {
            bins: Array1::<usize>::ones(nbins),
            range: Range { lo: 0.0, hi: 8.0 },
        };
        assert_eq!(hist.bin_width(), 1.0);
        assert_eq!(hist.first_bin_center(), 0.5);

        let cdf = hist.cdf();
        let cdf = cdf.as_slice().unwrap();
        assert_eq!(cdf, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let cdf = hist.cdf_as_normalized();
        let cdf = cdf.as_slice().unwrap();
        assert_eq!(cdf, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]);
    }
}

fn adjust_intensity<T: AtLeastF32 + numpy::Element + NumCast>(
    image: ArrayView2<T>,
    hist: Histogram<T>,
    out: &mut Array2<T>,
) {
    let cdf = hist.cdf_as_normalized();
    let grid = RegularGrid1D::new(
        hist.first_bin_center(),
        hist.bin_width(),
        cdf.as_slice().unwrap(),
    )
    .unwrap();
    let interpolator = LinearHoldLast1D::new(grid);

    let locs = image.as_slice().unwrap();
    match interpolator.eval(locs, out.as_slice_mut().unwrap()) {
        Ok(_) => (),
        Err(_) => panic!("interpolation failed"),
    }
}
fn adjust_intensity_single_pixel<T: AtLeastF32 + numpy::Element + NumCast>(
    interpolator: &LinearHoldLast1D<RegularGrid1D<'_, T>>,
    pixel: T,
) -> T {
    match interpolator.eval_one(pixel) {
        Ok(res) => res,
        Err(_) => panic!("interpolation failed"),
    }
}

fn equalize_histogram<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    nbins: usize,
    max_bincount: usize,
) -> Bound<'py, PyArray2<T>> {
    let image = image.as_array();
    let hist = compute_histogram(image, nbins, max_bincount);
    let mut out = Array2::<T>::zeros(image.raw_dim());
    adjust_intensity(image, hist, &mut out);
    out.to_pyarray(py)
}

#[derive(Copy, Clone, PartialEq)]
struct ViewRange {
    x: Range<usize>,
    y: Range<usize>,
}

fn get_tile_range(
    dims: ArrayDimensions,
    center_pixel: PixelIndex,
    tile_shape: ArrayDimensions,
) -> ViewRange {
    // since tile_shape only contains odd numbers,
    // the result of an integer division by 2 always corresponds
    // to the maximum number of pixels on one side of the central one,
    // *excluding* the latter.
    let half_tile_shape = ArrayDimensions {
        x: tile_shape.x / 2,
        y: tile_shape.y / 2,
    };
    ViewRange {
        x: Range {
            lo: center_pixel.j.saturating_sub(half_tile_shape.x),
            hi: if center_pixel.j + half_tile_shape.x > dims.x - 1 {
                dims.x - 1
            } else {
                center_pixel.j + half_tile_shape.x
            },
        },
        y: Range {
            lo: center_pixel.i.saturating_sub(half_tile_shape.y),
            hi: if center_pixel.i + half_tile_shape.y > dims.y - 1 {
                dims.y - 1
            } else {
                center_pixel.i + half_tile_shape.y
            },
        },
    }
}

fn get_tile_view<'a, T: numpy::Element>(
    image: &'a ArrayView2<'a, T>,
    range: ViewRange,
) -> ArrayView2<'a, T> {
    image.slice(s![range.y.lo..range.y.hi + 1, range.x.lo..range.x.hi + 1,])
}

fn get_value_range<A: AtLeastF32 + numpy::Element, D>(arr: ArrayView<A, D>) -> Range<A>
where
    D: Dimension,
{
    let mut lo = A::infinity();
    let mut hi = -A::infinity();
    for v in arr.iter() {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if hi == lo {
        if hi == 0.0.into() {
            hi = 1.0.into();
        } else {
            lo = 0.0.into();
        }
    }
    Range { lo, hi }
}

fn equalize_histogram_sliding_tile<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    nbins: usize,
    tile_shape: (usize, usize),
    max_bincount: usize,
) -> Bound<'py, PyArray2<T>> {
    let image = image.as_array();
    let dims = ArrayDimensions {
        x: image.shape()[1],
        y: image.shape()[0],
    };

    let tile_shape = ArrayDimensions {
        x: tile_shape.1,
        y: tile_shape.0,
    };

    let mut center_pixel = PixelIndex { i: 0, j: 0 };
    let mut out = Array2::<T>::zeros(image.raw_dim());
    let mut subhists = VecDeque::with_capacity(tile_shape.y + 1);
    let mut hist = Histogram {
        bins: (Array1::<usize>::zeros(nbins)),
        range: Range {
            lo: 0.0.into(),
            hi: 1.0.into(),
        },
    };
    let mut cdf = hist.cdf_as_normalized();
    let mut cdf_grid = RegularGrid1D::new(
        hist.first_bin_center(),
        hist.bin_width(),
        cdf.as_slice().unwrap(),
    )
    .unwrap();
    let mut cdf_interpolator = LinearHoldLast1D::new(cdf_grid);
    let mut subhists_need_reinit = true;
    let mut hist_reduction_needed = true;
    let mut previous_tile_range = ViewRange {
        x: Range { lo: 0, hi: 0 },
        y: Range { lo: 0, hi: 0 },
    };

    for j in 0..dims.x {
        center_pixel.j = j;
        let mut tile: ArrayView2<T> = image.slice(s![.., ..]);
        let mut tile_dims: ArrayDimensions = dims;
        let mut vrange: Range<T> = Range {
            lo: 0.0.into(),
            hi: 0.0.into(),
        };
        let mut row_vrange: Range<T> = vrange;

        for i in 0..dims.y {
            center_pixel.i = i;
            let tile_range = get_tile_range(dims, center_pixel, tile_shape);
            if tile_range != previous_tile_range {
                tile = get_tile_view(&image, tile_range);
                tile_dims = ArrayDimensions {
                    x: tile.shape()[1],
                    y: tile.shape()[0],
                };
                if tile_range.y.lo != previous_tile_range.y.lo {
                    subhists.pop_front();
                }
                vrange = get_value_range(tile);
                row_vrange = vrange;

                previous_tile_range = tile_range;
                subhists_need_reinit = true;
            }

            if row_vrange.lo < vrange.lo {
                vrange.lo = row_vrange.lo;
                subhists_need_reinit = true;
            }
            if row_vrange.hi > vrange.hi {
                vrange.hi = row_vrange.hi;
                subhists_need_reinit = true;
            }

            if subhists_need_reinit {
                subhists.truncate(0);
                for row in tile.axis_iter(Axis(0)) {
                    subhists.push_back(compute_subhistogram(row, vrange, nbins));
                }
                hist_reduction_needed = true;
                subhists_need_reinit = false;
            }
            if subhists.len() < tile_dims.y {
                let row = image.row(i);
                row_vrange = get_value_range(row);
                subhists.push_back(compute_subhistogram(row, vrange, nbins));
                hist_reduction_needed = true;
            }
            assert_eq!(subhists.len(), tile_dims.y);
            if hist_reduction_needed {
                hist = reduce_histogram(&subhists, max_bincount);
                cdf = hist.cdf_as_normalized();
                cdf_grid = RegularGrid1D::new(
                    hist.first_bin_center(),
                    hist.bin_width(),
                    cdf.as_slice().unwrap(),
                )
                .unwrap();
                cdf_interpolator = LinearHoldLast1D::new(cdf_grid);
                hist_reduction_needed = false;
            }

            let in_pix = image[[center_pixel.i, center_pixel.j]];
            let out_pix = &mut out[[center_pixel.i, center_pixel.j]];
            *out_pix = adjust_intensity_single_pixel(&cdf_interpolator, in_pix);
        }
    }

    out.to_pyarray(py)
}

struct ECRQuadruple<T> {
    top_left: T,
    top_right: T,
    bottom_left: T,
    bottom_right: T,
}

struct InterpolatorInputs<T: Float + numpy::Element> {
    start: T,
    step: T,
    vals: Array1<T>,
}

impl<T: Float + numpy::Element> InterpolatorInputs<T> {
    fn as_interpolator(&self) -> LinearHoldLast1D<RegularGrid1D<'_, T>> {
        let grid =
            RegularGrid1D::new(self.start, self.step, self.vals.as_slice().unwrap()).unwrap();
        LinearHoldLast1D::new(grid)
    }
}
struct EffectiveContextualRegion<'a, T: Float + numpy::Element> {
    top_left: &'a InterpolatorInputs<T>,
    top_right: &'a InterpolatorInputs<T>,
    bottom_left: &'a InterpolatorInputs<T>,
    bottom_right: &'a InterpolatorInputs<T>,
}

impl<T: AtLeastF32 + NumCast + numpy::Element> EffectiveContextualRegion<'_, T> {
    fn get_normalized_intensities(&self, intensity: T) -> ECRQuadruple<T> {
        ECRQuadruple {
            top_left: adjust_intensity_single_pixel(&self.top_left.as_interpolator(), intensity),
            top_right: adjust_intensity_single_pixel(&self.top_right.as_interpolator(), intensity),
            bottom_left: adjust_intensity_single_pixel(
                &self.bottom_left.as_interpolator(),
                intensity,
            ),
            bottom_right: adjust_intensity_single_pixel(
                &self.bottom_right.as_interpolator(),
                intensity,
            ),
        }
    }
}

struct Stencil<T> {
    x: Array1<T>,
    y: Array1<T>,
}
struct TargetTile<'a, T: Float + numpy::Element> {
    ecr: EffectiveContextualRegion<'a, T>,
    stencil: &'a Stencil<T>,
}
fn equalize_histogram_tile_interpolation<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    pimage: PyReadonlyArray2<'py, T>,
    nbins: usize,
    tile_shape: (usize, usize),
    max_bincount: usize,
) -> Bound<'py, PyArray2<T>> {
    let pimage = pimage.as_array();
    let pdims = ArrayDimensions {
        x: pimage.shape()[1],
        y: pimage.shape()[0],
    };
    let tile_shape = ArrayDimensions {
        x: tile_shape.1,
        y: tile_shape.0,
    };
    // assumptions:
    // - the full p(added)image is an integer multiple of tile sizes in every direction
    // - there's always *exactly* one entirely ghost tile in each direction
    // - it follows that every internal tile
    //   has a top, left, bottom and right neighbor
    // - external tiles do not need connectivity data (they are not to be iterated on)

    // ... define internal tiles ...
    // any "internal" may be only *partially* internal to the non-padded image,
    // but it'll always have a non-zero intersection with the unpadded domain

    let sample_mosaic_shape = ArrayDimensions {
        x: pdims.x / tile_shape.x,
        y: pdims.y / tile_shape.y,
    };

    let mut sample_interpolators = vec![];

    for i in (0..pdims.y).step_by(tile_shape.y) {
        let mut row_interpolators = vec![];
        for j in (0..pdims.x).step_by(tile_shape.x) {
            let vrange = ViewRange {
                x: Range {
                    lo: j,
                    hi: j + tile_shape.x - 1,
                },
                y: Range {
                    lo: i,
                    hi: i + tile_shape.y - 1,
                },
            };
            let tile_view = get_tile_view(&pimage, vrange);
            let hist = compute_histogram(tile_view, nbins, max_bincount);
            let cdf = hist.cdf_as_normalized();
            let ii = InterpolatorInputs {
                start: hist.first_bin_center(),
                step: hist.bin_width(),
                vals: cdf,
            };
            row_interpolators.push(ii);
        }
        sample_interpolators.push(row_interpolators);
    }
    assert_eq!(sample_interpolators.len(), sample_mosaic_shape.y);
    for si in sample_interpolators.iter().take(sample_mosaic_shape.y) {
        assert_eq!(si.len(), sample_mosaic_shape.x);
    }
    let half_tile_offset_x = tile_shape.x / 2;
    let half_tile_offset_y = tile_shape.y / 2;
    let mut tiles: Vec<Vec<TargetTile<T>>> = vec![];

    // x offset, in pixel width, between the top left tile center and the center
    // of the first pixel in the target tile. Since all tile sizes are even,
    // it follows that this is always 0.5
    let xoff: T = 0.5.into();
    let mut alpha = Array1::<T>::zeros(tile_shape.x);
    let tsx = <T as NumCast>::from(tile_shape.x).unwrap();
    for j in 0..tile_shape.x {
        let fj = <T as NumCast>::from(j).unwrap();
        alpha[j] = (xoff + fj) / tsx;
    }

    // y offset, in pixel height, between the top left tile center and the center
    // of the first pixel in the target tile.
    let yoff: T = 0.5.into();
    let mut beta = Array1::<T>::zeros(tile_shape.y);
    let tsy = <T as NumCast>::from(tile_shape.y).unwrap();
    for i in 0..tile_shape.y {
        let fi = <T as NumCast>::from(i).unwrap();
        beta[i] = (yoff + fi) / tsy;
    }

    let stencil = Stencil { x: alpha, y: beta };

    for imos in 0..sample_mosaic_shape.y - 1 {
        let mut row = vec![];
        for jmos in 0..sample_mosaic_shape.x - 1 {
            row.push(TargetTile {
                ecr: EffectiveContextualRegion {
                    top_left: &sample_interpolators[imos][jmos],
                    top_right: &sample_interpolators[imos][jmos + 1],
                    bottom_left: &sample_interpolators[imos + 1][jmos],
                    bottom_right: &sample_interpolators[imos + 1][jmos + 1],
                },
                stencil: &stencil,
            })
        }
        tiles.push(row);
    }
    let one: T = 1.0.into();

    let mut out = Array2::<T>::zeros(pimage.raw_dim());
    for (itiles, row) in tiles.iter().enumerate().take(sample_mosaic_shape.y - 1) {
        let ioff = half_tile_offset_y + itiles * tile_shape.y;
        for (jtiles, tile) in row.iter().enumerate().take(sample_mosaic_shape.x - 1) {
            let joff = half_tile_offset_x + jtiles * tile_shape.x;
            for i in 0..tile_shape.y {
                for j in 0..tile_shape.x {
                    let v = pimage[[ioff + i, joff + j]];
                    let q = tile.ecr.get_normalized_intensities(v);
                    let a = tile.stencil.x[j];
                    let b = tile.stencil.y[i];
                    // my convention for a VS b is completely different from Fizer 1987
                    // maybe I should align them.
                    out[[ioff + i, joff + j]] = b
                        * (a * q.bottom_right + (one - a) * q.bottom_left)
                        + (one - b) * (a * q.top_right + (one - a) * q.top_left);
                }
            }
        }
    }
    out.to_pyarray(py)
}

#[pymodule(gil_used = false)]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfunction]
    fn equalize_histogram_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram(py, image, nbins, max_bincount)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_f32, m)?)?;

    #[pyfunction]
    fn equalize_histogram_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram(py, image, nbins, max_bincount)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_f64, m)?)?;

    #[pyfunction]
    fn equalize_histogram_sliding_tile_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
        tile_shape: (usize, usize),
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape, max_bincount)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f32, m)?)?;

    #[pyfunction]
    fn equalize_histogram_sliding_tile_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        tile_shape: (usize, usize),
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram_sliding_tile(py, image, nbins, tile_shape, max_bincount)
    }
    m.add_function(wrap_pyfunction!(equalize_histogram_sliding_tile_f64, m)?)?;

    #[pyfunction]
    fn equalize_histogram_tile_interpolation_f32<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        nbins: usize,
        tile_shape: (usize, usize),
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        equalize_histogram_tile_interpolation(py, image, nbins, tile_shape, max_bincount)
    }
    m.add_function(wrap_pyfunction!(
        equalize_histogram_tile_interpolation_f32,
        m
    )?)?;

    #[pyfunction]
    fn equalize_histogram_tile_interpolation_f64<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        nbins: usize,
        tile_shape: (usize, usize),
        max_bincount: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        equalize_histogram_tile_interpolation(py, image, nbins, tile_shape, max_bincount)
    }
    m.add_function(wrap_pyfunction!(
        equalize_histogram_tile_interpolation_f64,
        m
    )?)?;

    Ok(())
}
