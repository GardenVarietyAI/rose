use candle_core::{DType, Device, Tensor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub device: Device,
    pub weights_dtype: DType,
    pub compute_dtype: DType,
    pub kv_cache_dtype: DType,
}

impl DeviceConfig {
    pub fn from_string(device_str: Option<&str>) -> PyResult<Self> {
        let device_str = device_str.unwrap_or("auto");
        tracing::info!("Device selection: requested='{}'", device_str);

        let device = match device_str {
            "cpu" => {
                tracing::info!("Using CPU device");
                Device::Cpu
            }
            "cuda" => {
                if let Ok(d) = Device::new_cuda(0) {
                    tracing::info!("Using CUDA device");
                    d
                } else {
                    return Err(PyRuntimeError::new_err("CUDA device not available"));
                }
            }
            "metal" => {
                if let Ok(d) = Device::new_metal(0) {
                    tracing::info!("Using Metal device");
                    d
                } else {
                    return Err(PyRuntimeError::new_err("Metal device not available"));
                }
            }
            _ => {
                // Auto selection: try Metal first, then CPU
                if let Ok(d) = Device::new_metal(0) {
                    tracing::info!("Auto-selected Metal device");
                    d
                } else {
                    tracing::info!("Metal not available, using CPU device");
                    Device::Cpu
                }
            }
        };

        let dtype_config = Self::detect_dtypes(&device);
        tracing::info!(
            "Device config: device={:?}, weights={:?}, compute={:?}, kv_cache={:?}",
            Self::device_kind(&device),
            dtype_config.weights_dtype,
            dtype_config.compute_dtype,
            dtype_config.kv_cache_dtype
        );

        Ok(dtype_config)
    }

    pub fn detect_dtypes(device: &Device) -> Self {
        let has_f16 = Self::supports_dtype(device, DType::F16);
        let has_bf16 = Self::supports_dtype(device, DType::BF16);

        match device {
            Device::Cpu => Self {
                device: device.clone(),
                weights_dtype: DType::F32,
                compute_dtype: DType::F32,
                kv_cache_dtype: DType::F32,
            },
            Device::Cuda(_) => {
                let weights = if has_bf16 {
                    DType::BF16
                } else if has_f16 {
                    DType::F16
                } else {
                    DType::F32
                };
                let compute = if has_f16 { DType::F16 } else { DType::F32 };
                let kv = if has_f16 { DType::F16 } else { DType::F32 };
                Self {
                    device: device.clone(),
                    weights_dtype: weights,
                    compute_dtype: compute,
                    kv_cache_dtype: kv,
                }
            }
            Device::Metal(_) => {
                let compute = if has_f16 { DType::F16 } else { DType::F32 };
                Self {
                    device: device.clone(),
                    weights_dtype: if has_f16 { DType::F16 } else { DType::F32 },
                    compute_dtype: compute,
                    kv_cache_dtype: compute,
                }
            }
        }
    }

    fn supports_dtype(device: &Device, dt: DType) -> bool {
        // Capability probe
        Tensor::zeros(&[1], dt, device).is_ok()
    }

    pub fn device_kind(device: &Device) -> String {
        match device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{id:?}"),
            Device::Metal(id) => format!("metal:{id:?}"),
        }
    }
}
