# Face Identification Demo (JS + ONNXRuntime Web)

Demo web app for face identification in browser:

- detects a face in image/frame
- converts face to embedding (vector)
- stores embedding + metadata (`personId`, `firstName`, `lastName`) locally on device
- identifies known people by cosine similarity

## Stack

- `onnxruntime-web` for ONNX inference in browser
- Vite + vanilla JavaScript
- local storage in browser (`localStorage`)

## Selected Models

- Detector: `UltraFace version-RFB-320` (very light, good for realtime demo)
- Embeddings: `InsightFace w600k_mbf` (MobileFace-based, 512-d embeddings, good speed/quality tradeoff)

## Quick Start

1. Install dependencies:

```bash
npm install
```

2. Download ONNX models:

```bash
mkdir -p public/models

curl -L "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/6fd293d22b523ec88959f104b8eef5395e3adfbc/version-RFB-320.onnx?download=true" \
  -o public/models/version-RFB-320.onnx

curl -L "https://huggingface.co/deepghs/insightface/resolve/4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb/buffalo_s/w600k_mbf.onnx?download=true" \
  -o public/models/w600k_mbf.onnx
```

3. Run the app:

```bash
npm run dev
```

4. Open local URL shown by Vite.

## Usage

1. Start camera or upload image.
2. Capture a frame.
3. Fill `Person ID`, `First Name`, `Last Name`.
4. Click `Detect Face + Save Embedding`.
5. For identification, click `Detect Face + Identify`.
6. Adjust threshold if needed (default `0.52`).

## Notes

- Data is stored only in browser cache on current device.
- This is a demo and uses face crop without landmark alignment.
- For production usage, use stronger liveness checks, encrypted storage, and legal/privacy controls.
