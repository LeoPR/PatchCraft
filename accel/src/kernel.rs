//! Gather-formulation fold with overlap: `out[c, y, x]` is the sum of
//! `patches[p, c, y - i*sh, x - j*sw]` (optionally kernel-weighted) over every
//! patch `p = (i, j)` covering output pixel `(y, x)`.
//!
//! Pure Rust — no pyo3 types — so `cargo test` exercises it directly.
//! Parallelized over output rows with rayon; each output pixel is written by
//! exactly one worker (no atomics, no data races) and the summation order per
//! pixel is fixed (descending patch index — i.e. ascending kernel offset,
//! matching ATen col2im's per-pixel order), so results are deterministic
//! across runs and thread counts.

use rayon::prelude::*;

/// Accumulation element. Only f32/f64 are ever exposed; the trait exists so
/// the kernel is written once.
pub trait Scalar:
    Copy + Send + Sync + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    const ZERO: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
}

/// Inclusive `[lo, hi]` range of patch indices along one axis whose patches
/// cover output coordinate `pos`: patch `i` starts at `i * step` and spans
/// `patch` samples, so it covers `pos` iff `pos + 1 - patch <= i*step <= pos`.
#[inline]
fn covering(pos: usize, patch: usize, num: usize, step: usize) -> (usize, usize) {
    let lo = (pos + 1).saturating_sub(patch).div_ceil(step);
    let hi = (pos / step).min(num - 1);
    (lo, hi)
}

/// Sum `patches` `(l, c, ph, pw)` into `out` `(c, h, w)` on the regular grid
/// with stride `(sh, sw)`, multiplying each contribution by `kernel[dy, dx]`
/// when a kernel is given. All slices are row-major contiguous. Geometry is
/// pre-validated by the Python caller; debug asserts only.
#[allow(clippy::too_many_arguments)]
pub fn fold_add<T: Scalar>(
    patches: &[T],
    out: &mut [T],
    l: usize,
    c: usize,
    ph: usize,
    pw: usize,
    h: usize,
    w: usize,
    sh: usize,
    sw: usize,
    kernel: Option<&[T]>,
) {
    debug_assert!(ph > 0 && pw > 0 && sh > 0 && sw > 0);
    debug_assert_eq!(patches.len(), l * c * ph * pw);
    debug_assert_eq!(out.len(), c * h * w);
    let num_h = (h - ph) / sh + 1;
    let num_w = (w - pw) / sw + 1;
    debug_assert_eq!(l, num_h * num_w);
    if let Some(k) = kernel {
        debug_assert_eq!(k.len(), ph * pw);
    }

    // One work item per (channel, row): `out` rows are contiguous W-sized
    // chunks and no two items touch the same memory.
    out.par_chunks_mut(w).enumerate().for_each(|(row, out_row)| {
        let ch = row / h;
        let y = row % h;
        let (i_lo, i_hi) = covering(y, ph, num_h, sh);
        let ch_base = ch * ph * pw;
        for (x, slot) in out_row.iter_mut().enumerate() {
            let (j_lo, j_hi) = covering(x, pw, num_w, sw);
            let mut acc = T::ZERO;
            // Descending patch index = ascending kernel offset (dy, dx): the
            // same per-pixel summation order as ATen's CPU col2im, so results
            // are bit-identical to `F.fold`.
            for i in (i_lo..=i_hi).rev() {
                let dy = y - i * sh;
                for j in (j_lo..=j_hi).rev() {
                    let dx = x - j * sw;
                    let p = i * num_w + j;
                    let v = patches[p * (c * ph * pw) + ch_base + dy * pw + dx];
                    acc = acc
                        + match kernel {
                            Some(k) => v * k[dy * pw + dx],
                            None => v,
                        };
                }
            }
            *slot = acc;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only generic conversion so fixtures can be written once.
    trait FromF64 {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }

    /// Naive scatter reference: for each patch in descending order, add it into
    /// `out`. Per-pixel arrival order is descending patch index (= ascending
    /// kernel offset, matching ATen col2im's per-pixel order) — the same order
    /// the gather kernel sums in — so results must be bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn reference<T: Scalar>(
        patches: &[T],
        out: &mut [T],
        l: usize,
        c: usize,
        ph: usize,
        pw: usize,
        h: usize,
        w: usize,
        sh: usize,
        sw: usize,
        kernel: Option<&[T]>,
    ) {
        let num_w = (w - pw) / sw + 1;
        for p in (0..l).rev() {
            let (i, j) = (p / num_w, p % num_w);
            for ch in 0..c {
                for dy in 0..ph {
                    for dx in 0..pw {
                        let v = patches[((p * c + ch) * ph + dy) * pw + dx];
                        let v = match kernel {
                            Some(k) => v * k[dy * pw + dx],
                            None => v,
                        };
                        out[(ch * h + i * sh + dy) * w + j * sw + dx] =
                            out[(ch * h + i * sh + dy) * w + j * sw + dx] + v;
                    }
                }
            }
        }
    }

    /// Deterministic pseudo-random value in [-0.5, 0.5).
    fn prand(p: usize, c: usize, y: usize, x: usize) -> f64 {
        ((p * 131 + c * 17 + y * 7 + x) % 97) as f64 / 97.0 - 0.5
    }

    struct Geom {
        l: usize,
        c: usize,
        ph: usize,
        pw: usize,
        h: usize,
        w: usize,
        sh: usize,
        sw: usize,
    }

    /// Geometry with exact coverage from grid counts.
    fn case(c: usize, ph: usize, pw: usize, sh: usize, sw: usize, num_h: usize, num_w: usize) -> Geom {
        Geom {
            l: num_h * num_w,
            c,
            ph,
            pw,
            h: (num_h - 1) * sh + ph,
            w: (num_w - 1) * sw + pw,
            sh,
            sw,
        }
    }

    fn run<T: Scalar + FromF64 + PartialEq + std::fmt::Debug>(g: &Geom, with_kernel: bool) {
        let &Geom { l, c, ph, pw, h, w, sh, sw } = g;
        let patches: Vec<T> = (0..l * c * ph * pw)
            .map(|idx| {
                let dx = idx % pw;
                let dy = (idx / pw) % ph;
                let ch = (idx / (pw * ph)) % c;
                let p = idx / (pw * ph * c);
                T::from_f64(prand(p, ch, dy, dx))
            })
            .collect();
        let kernel: Vec<T> = (0..ph * pw)
            .map(|i| T::from_f64(0.25 + 0.5 * ((i % 5) as f64) / 5.0))
            .collect();
        let k = if with_kernel { Some(&kernel[..]) } else { None };

        let mut out = vec![T::ZERO; c * h * w];
        let mut expected = vec![T::ZERO; c * h * w];
        fold_add(&patches, &mut out, l, c, ph, pw, h, w, sh, sw, k);
        reference(&patches, &mut expected, l, c, ph, pw, h, w, sh, sw, k);
        assert_eq!(
            out, expected,
            "mismatch: c={c} ph={ph} pw={pw} sh={sh} sw={sw} h={h} w={w} kernel={with_kernel}"
        );
    }

    #[test]
    fn overlap_square_f64() {
        let g = case(3, 4, 4, 2, 2, 8, 8);
        run::<f64>(&g, false);
        run::<f64>(&g, true);
    }

    #[test]
    fn overlap_square_f32() {
        let g = case(3, 4, 4, 2, 2, 8, 8);
        run::<f32>(&g, false);
        run::<f32>(&g, true);
    }

    #[test]
    fn rectangular_everything() {
        let g = case(3, 4, 6, 3, 5, 5, 4); // 16x21
        run::<f64>(&g, false);
        run::<f64>(&g, true);
    }

    #[test]
    fn stride_does_not_divide_patch() {
        let g = case(1, 5, 4, 3, 2, 4, 6); // 14x14
        run::<f64>(&g, false);
        run::<f64>(&g, true);
    }

    #[test]
    fn stride_one_max_overlap() {
        let g = case(3, 4, 4, 1, 1, 5, 5); // 8x8, 25 patches
        run::<f64>(&g, false);
        run::<f64>(&g, true);
    }

    #[test]
    fn single_patch_no_overlap() {
        let g = case(3, 4, 4, 4, 4, 1, 1); // 4x4, one patch
        run::<f64>(&g, false);
        run::<f64>(&g, true);
    }
}
