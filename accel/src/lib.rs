//! Python bindings for the patchcraft native accelerator.
//!
//! Raw-pointer interface: the Python caller owns the buffers and guarantees
//! they are contiguous, correctly sized, CPU-resident, correctly typed
//! (`dtype`), and alive for the duration of the (synchronous) call. No torch
//! coupling: this module never imports or links against it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod kernel;

/// Bumped on any breaking change to the native signature. patchcraft checks
/// it before calling in; a mismatch means a silent pure-torch fallback.
const ABI_VERSION: u32 = 1;

/// fold_add(patches_ptr, out_ptr, l, c, ph, pw, h, w, sh, sw, kernel_ptr, dtype)
#[pyfunction]
// Explicit signature: pyo3 treats `Option<_>` args as optional-with-default,
// which would make the required trailing `dtype` ambiguous. Listing every
// parameter keeps all of them positional-and-required while `kernel_ptr`
// still accepts Python `None` (mapped to `None`) or an int (`Some(ptr)`).
#[pyo3(signature = (patches_ptr, out_ptr, l, c, ph, pw, h, w, sh, sw, kernel_ptr, dtype))]
#[allow(clippy::too_many_arguments)]
fn fold_add(
    py: Python<'_>,
    patches_ptr: usize,
    out_ptr: usize,
    l: usize,
    c: usize,
    ph: usize,
    pw: usize,
    h: usize,
    w: usize,
    sh: usize,
    sw: usize,
    kernel_ptr: Option<usize>,
    dtype: &str,
) -> PyResult<()> {
    let patches_len = l
        .checked_mul(c)
        .and_then(|v| v.checked_mul(ph))
        .and_then(|v| v.checked_mul(pw))
        .ok_or_else(|| PyValueError::new_err("patches size overflow"))?;
    let out_len = c
        .checked_mul(h)
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| PyValueError::new_err("output size overflow"))?;
    let kernel_len = ph
        .checked_mul(pw)
        .ok_or_else(|| PyValueError::new_err("kernel size overflow"))?;

    match dtype {
        "f32" => unsafe {
            // SAFETY: the Python caller guarantees the pointers reference
            // contiguous, correctly sized, aligned, live buffers for the
            // duration of this synchronous call, and `out` is exclusively
            // owned. Slices are built before the GIL release so no raw
            // pointers cross into the closure.
            let patches = std::slice::from_raw_parts(patches_ptr as *const f32, patches_len);
            let out = std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len);
            let k = kernel_ptr.map(|p| std::slice::from_raw_parts(p as *const f32, kernel_len));
            py.allow_threads(|| kernel::fold_add(patches, out, l, c, ph, pw, h, w, sh, sw, k));
        },
        "f64" => unsafe {
            // SAFETY: same contract as the f32 arm.
            let patches = std::slice::from_raw_parts(patches_ptr as *const f64, patches_len);
            let out = std::slice::from_raw_parts_mut(out_ptr as *mut f64, out_len);
            let k = kernel_ptr.map(|p| std::slice::from_raw_parts(p as *const f64, kernel_len));
            py.allow_threads(|| kernel::fold_add(patches, out, l, c, ph, pw, h, w, sh, sw, k));
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "dtype must be \"f32\" or \"f64\", got {other:?}"
            )));
        }
    }
    Ok(())
}

#[pymodule]
fn patchcraft_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("_ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(fold_add, m)?)?;
    Ok(())
}
