//! Python bindings for the patchcraft native accelerator.
//!
//! Built into the `patchcraft` wheel itself as `patchcraft._accel_native`.
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
    // A slice may not be built from a null or a misaligned pointer, whatever
    // its length. The Python caller declines both: a data_ptr() of 0 is what
    // torch.func.functionalize and FakeTensorMode hand out, and a tensor viewed
    // from a byte buffer at an offset can start off its dtype's alignment.
    // This is the second line, for any caller that reaches the function
    // directly.
    let align = match dtype {
        "f32" => std::mem::align_of::<f32>(),
        "f64" => std::mem::align_of::<f64>(),
        other => {
            return Err(PyValueError::new_err(format!(
                "dtype must be \"f32\" or \"f64\", got {other:?}"
            )));
        }
    };
    if patches_ptr == 0 || out_ptr == 0 || kernel_ptr == Some(0) {
        return Err(PyValueError::new_err("null data pointer"));
    }
    if patches_ptr % align != 0
        || out_ptr % align != 0
        || kernel_ptr.is_some_and(|p| p % align != 0)
    {
        return Err(PyValueError::new_err("misaligned data pointer"));
    }

    match dtype {
        "f32" => unsafe {
            // SAFETY: the checks above rule out null and misaligned pointers,
            // and the Python caller guarantees they reference contiguous,
            // correctly sized, live buffers for the duration of this
            // synchronous call, and that `out` is exclusively owned. Slices
            // are built before the GIL release so no raw pointers cross into
            // the closure.
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
        _ => unreachable!("dtype was checked above"),
    }
    Ok(())
}

#[pymodule]
fn _accel_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("_ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(fold_add, m)?)?;
    Ok(())
}
