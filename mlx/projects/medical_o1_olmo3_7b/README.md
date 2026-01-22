# medical_o1

Project folder for medical fine-tuning experiments.

#
## Run

```bash
uv run --script mlx/medical_o1_olmo3_7b.py
```

Outputs are written under `mlx/artifacts/<run>/{data,adapters,models}` where `<run>` defaults to `medical_o1_<date>_<unix_timestamp>`.

## Config

- Training config (LoRA targets, LR schedule, etc): `mlx/projects/medical_o1/train_config.yaml`
