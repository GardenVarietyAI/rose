use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TensorKey {
    shape: Vec<usize>,
    dtype: DType,
    device_id: String,
}

impl TensorKey {
    fn new(shape: &[usize], dtype: DType, device: &Device) -> Self {
        let device_id = match device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{:?}", id),
            Device::Metal(id) => format!("metal:{:?}", id),
        };
        Self {
            shape: shape.to_vec(),
            dtype,
            device_id,
        }
    }
}

pub struct TensorPool {
    pool: Arc<Mutex<HashMap<TensorKey, Vec<Tensor>>>>,
    max_per_key: usize,
}

impl TensorPool {
    pub fn new(max_per_key: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(HashMap::new())),
            max_per_key,
        }
    }

    #[allow(dead_code)]
    pub fn get_or_create(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let key = TensorKey::new(shape, dtype, device);

        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(tensors) = pool.get_mut(&key) {
                if let Some(tensor) = tensors.pop() {
                    return Ok(tensor);
                }
            }
        }

        Tensor::zeros(shape, dtype, device).map_err(Into::into)
    }

    pub fn return_tensor(&self, tensor: Tensor) {
        let shape = tensor.shape().dims();
        let dtype = tensor.dtype();
        let device = tensor.device();
        let key = TensorKey::new(shape, dtype, device);

        let mut pool = self.pool.lock().unwrap();
        let tensors = pool.entry(key).or_insert_with(Vec::new);

        if tensors.len() < self.max_per_key {
            tensors.push(tensor);
        }
    }

    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();
    }
}

static GLOBAL_TENSOR_POOL: std::sync::OnceLock<TensorPool> = std::sync::OnceLock::new();

pub fn get_tensor_pool() -> &'static TensorPool {
    GLOBAL_TENSOR_POOL.get_or_init(|| TensorPool::new(8))
}

#[allow(dead_code)]
pub fn get_pooled_zeros(shape: &[usize], dtype: DType, device: &Device) -> anyhow::Result<Tensor> {
    get_tensor_pool().get_or_create(shape, dtype, device)
}

pub fn return_to_pool(tensor: Tensor) {
    get_tensor_pool().return_tensor(tensor);
}
