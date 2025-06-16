## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient RAM/VRAM
   - Check CUDA installation for GPU usage
   - Verify model files are downloaded

2. **Fine-Tuning OOM**
   - Reduce batch size
   - Enable LoRA (default)
   - Use smaller models

3. **ChromaDB Connection**
   - Service automatically falls back to local storage
   - No action needed unless you specifically need server mode

### Getting Help

1. Check service logs: `tail -f service.log`
2. Verify service health: `curl http://localhost:8004/health`
3. List available models: `poetry run rose models list`
