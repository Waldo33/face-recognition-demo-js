# Face Identification Demo (JS + ONNXRuntime Web)

Demo web app for face identification in browser:

- detects a face in image/frame
- converts face to embedding (vector)
- stores embedding + metadata (`personId`, `firstName`, `lastName`) locally on device
- identifies known people by cosine similarity

## Stack

- `onnxruntime-web` for ONNX inference in browser
- Vite + vanilla JavaScript
- local storage in browser (`IndexedDB`, embeddings stored as binary `ArrayBuffer`)

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

## Build and Deploy to GitHub Pages

1. Push project to GitHub repository (branch `main`).
2. In repository settings open `Settings -> Pages`.
3. Set `Build and deployment -> Source` to `GitHub Actions`.
4. Workflow `.github/workflows/deploy-pages.yml` will build Vite project and deploy `dist` automatically on each push to `main`.
5. Final URL format:
   `https://<your-user>.github.io/<repo-name>/`

Notes:

- `vite.config.js` sets correct `base` automatically in GitHub Actions (`/<repo-name>/`), and uses `/` locally.
- ONNX models must be present in `public/models` before build.

## Usage

1. Case `Automatic Recognition`:
   Start camera, set threshold and interval, click `Start Auto Identify`.
2. Case `Add New Person`:
   Capture camera frame or upload photo, fill metadata, click `Detect Face + Save Sample`.
3. Repeated save with the same `Person ID` adds more face samples (up to 12) to improve matching.

## Notes

- Data is stored only in browser `IndexedDB` on current device.
- Service Worker caches app resources and ONNX models for offline reuse after first successful online load.
- This is a demo and uses face crop without landmark alignment.
- For production usage, use stronger liveness checks, encrypted storage, and legal/privacy controls.
