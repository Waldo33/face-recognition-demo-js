# ONNX Models

Put model files in this folder before running the demo:

- `version-RFB-320.onnx` (UltraFace detector, lightweight)
- `w600k_mbf.onnx` (InsightFace MobileFace embedding model, 512-d)

Example download commands:

```bash
mkdir -p public/models

curl -L "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/6fd293d22b523ec88959f104b8eef5395e3adfbc/version-RFB-320.onnx?download=true" \
  -o public/models/version-RFB-320.onnx

curl -L "https://huggingface.co/deepghs/insightface/resolve/4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb/buffalo_s/w600k_mbf.onnx?download=true" \
  -o public/models/w600k_mbf.onnx
```

If any URL changes, update `MODEL_PATHS` in `src/main.js`.
