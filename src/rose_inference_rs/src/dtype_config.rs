use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone, PartialEq)]
pub struct DTypeConfig {
    pub weights_dtype: DType,
    pub compute_dtype: DType,
    pub kv_cache_dtype: DType,
}

impl DTypeConfig {
    pub fn auto_detect(device: &Device) -> Self {
        let has_f16 = Self::supports(device, DType::F16);
        let has_bf16 = Self::supports(device, DType::BF16);

        match device {
            Device::Cpu => Self {
                weights_dtype: if has_bf16 { DType::BF16 } else { DType::F32 },
                compute_dtype: DType::F32,
                kv_cache_dtype: if has_bf16 { DType::BF16 } else { DType::F32 },
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
                    weights_dtype: weights,
                    compute_dtype: compute,
                    kv_cache_dtype: kv,
                }
            }
            Device::Metal(_) => {
                let compute = if has_f16 { DType::F16 } else { DType::F32 };
                Self {
                    weights_dtype: if has_f16 { DType::F16 } else { DType::F32 },
                    compute_dtype: compute,
                    kv_cache_dtype: compute,
                }
            }
        }
    }

    fn supports(device: &Device, dt: DType) -> bool {
        // Capability probe
        Tensor::zeros(&[1], dt, device).is_ok()
    }
}
