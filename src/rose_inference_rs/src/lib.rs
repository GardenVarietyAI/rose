use pyo3::prelude::*;
use pyo3::types::PyDict;

// Alias the shim so `tokio` remains the real crate name.
use pyo3_async_runtimes::tokio as pyo3_tokio;
use pyo3_async_runtimes::tokio::future_into_py;

#[pymodule]
fn rose_inference_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build a Tokio runtime and register it
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let rt = ::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        // Hand the runtime to pyo3-async-runtimes
        let _ = pyo3_tokio::init_with_runtime(Box::leak(Box::new(rt)));
    });

    m.add_class::<InferenceServer>()?;
    Ok(())
}

#[pyclass]
pub struct InferenceServer;

#[pymethods]
impl InferenceServer {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (request, on_event))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        request: PyObject,
        on_event: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = request;
        let cb = on_event.clone_ref(py);

        future_into_py(py, async move {
            Python::with_gil(|py| {
                let d = PyDict::new(py);
                d.set_item("type", "InputTokensCounted").ok();
                d.set_item("input_tokens", 0u32).ok();
                let _ = cb.bind(py).call1((d,));
            });
            Python::with_gil(|py| {
                let d = PyDict::new(py);
                d.set_item("type", "Token").ok();
                d.set_item("token", "hello").ok();
                d.set_item("token_id", 0u32).ok();
                d.set_item("position", 0u32).ok();
                let _ = cb.bind(py).call1((d,));
            });
            Python::with_gil(|py| {
                let d = PyDict::new(py);
                d.set_item("type", "Complete").ok();
                d.set_item("input_tokens", 0u32).ok();
                d.set_item("output_tokens", 1u32).ok();
                d.set_item("total_tokens", 1u32).ok();
                d.set_item("finish_reason", "stop").ok();
                let _ = cb.bind(py).call1((d,));
            });
            Ok(Python::with_gil(|py| py.None()))
        })
    }
}
