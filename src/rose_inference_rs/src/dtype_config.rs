use candle_core::{Device, DType};

/// Configuration for dtype usage across the inference pipeline
#[derive(Debug, Clone, PartialEq)]
pub struct DTypeConfig {
    /// Data type for storing model weights (F32, F16, BF16)
    pub weights_dtype: DType,
    /// Data type for computation operations (typically F32 for stability)
    pub compute_dtype: DType,
    /// Data type for KV cache storage (F32, F16, BF16, or quantized)
    pub kv_cache_dtype: DType,
}

impl DTypeConfig {
    /// Auto-detect optimal configuration based on device capabilities
    pub fn auto_detect(device: &Device) -> Self {
        match device {
            Device::Cpu => Self {
                weights_dtype: DType::BF16,
                compute_dtype: DType::F32,
                kv_cache_dtype: DType::F32,
            },
            Device::Cuda(_) => Self {
                weights_dtype: DType::BF16,
                compute_dtype: DType::F32,
                kv_cache_dtype: DType::F16,
            },
            Device::Metal(_) => Self {
                weights_dtype: DType::F32, // Use F32 for Metal compatibility
                compute_dtype: DType::F32,
                kv_cache_dtype: DType::F32,
            },
        }
    }

    /// Validate that the configuration is supported on the given device
    pub fn validate(&self, device: &Device) -> Result<(), String> {
        // For now, be conservative and only allow known-good combinations
        match device {
            Device::Metal(_) => {
                if self.compute_dtype != DType::F32 {
                    return Err("Metal backend currently supports F32 for compute operations".to_string());
                }
                if self.weights_dtype != DType::F32 {
                    return Err("Metal backend currently supports F32 for weights".to_string());
                }
            },
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_config() {
        let cpu_config = DTypeConfig::auto_detect(&Device::Cpu);
        assert_eq!(cpu_config.weights_dtype, DType::BF16);
        assert_eq!(cpu_config.compute_dtype, DType::F32);
        assert_eq!(cpu_config.kv_cache_dtype, DType::F32);
    }

    #[test]
    fn test_validate_config() {
        let config = DTypeConfig::auto_detect(&Device::Cpu);
        assert!(config.validate(&Device::Cpu).is_ok());
    }
}
