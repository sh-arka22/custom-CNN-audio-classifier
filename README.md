i# Audio CNN Classifier

A production-ready audio classification stack that trains a residual CNN on the ESC-50 dataset, serves real-time predictions via Modal, and exposes intermediate feature maps for a Next.js visualization UI.

## Key Features

- **Residual CNN backbone** tailored for Mel-spectrogram inputs with ResNet-34–style depth.
- **Automated ESC-50 ingestion** and augmentation (Mel front-end, SpecAugment, MixUp).
- **Remote training & inference with Modal** (GPU-enabled, persistent volumes for checkpoints/logs).
- **Interactive visualization app** (Next.js + Tailwind) for feature maps, spectrograms, and waveforms.
- **TensorBoard logging** for losses, accuracy, learning-rate schedules.

## Repository Layout

- `model.py` – Residual network definition and feature-map extraction hooks.
- `train.py` – Modal training job, dataset pipeline, logging, and checkpointing.
- `main.py` – Modal inference service exposing a FastAPI endpoint.
- `requirements.txt` – Python dependencies for training/inference workloads.
- `audio-cnn-visualisation/` – Next.js UI that consumes the API payload for visual analytics.
- `clap.wav` – Sample clip used by the local inference smoke test.
- `tensorboard_logs/` – Local TensorBoard runs (remote jobs log to `/models/tensorboard_logs` on Modal volumes).

## Model Architecture

The classifier is a single-channel 2D CNN operating on Mel-spectrograms.

| Stage   | Configuration                                                                 | Details                                                                     |
|---------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Input   | `Conv2d(1→64, 7×7, stride=2)` → BN → ReLU → `MaxPool(3×3, stride=2)`         | Projects waveform-derived spectrograms into a 64-channel feature grid.     |
| Layer 1 | 3 × `ResidualBlock(64→64)`                                                    | Identity shortcuts, stride 1.                                               |
| Layer 2 | `ResidualBlock(64→128, stride=2)` + 3 × `ResidualBlock(128→128)`             | First block downsamples using a 1×1 projection shortcut.                    |
| Layer 3 | `ResidualBlock(128→256, stride=2)` + 5 × `ResidualBlock(256→256)`            | Deepest feature extractor; richest spatial representations.                 |
| Layer 4 | `ResidualBlock(256→512, stride=2)` + 2 × `ResidualBlock(512→512)`            | Compresses to high-level semantics.                                         |
| Head    | AdaptiveAvgPool → Flatten → Dropout(0.5) → `Linear(512→N_CLASSES)`           | Global pooling enables variable spectrogram lengths.                        |

Each `ResidualBlock` contains two 3×3 convolutions with BatchNorm, ReLU activations, and optional projection shortcuts. When `return_feature_maps=True`, intermediate tensors are captured for visualization:

- Per-block pre-activation (`*.conv`) and post-activation (`*.relu`) maps.
- Stage-level outputs (`layer{n}`) after every residual stack.

## Data & Training Pipeline

- **Dataset**: ESC-50 (5,000 environmental audio clips across 50 classes). The Modal image downloads and caches the dataset under `/opt/esc50-data`.
- **Transforms**:
  - Train: `MelSpectrogram` (128 Mel bins, **44.1 kHz**), amplitude scaling to dB, frequency masking, time masking.
  - Eval: Same Mel front-end without masking.
  - MixUp with Beta(0.2, 0.2) applied 30% of the time to improve generalization.
- **Optimization**:
  - Loss: CrossEntropy with 0.1 label smoothing.
  - Optimizer: AdamW (LR 5e-4, weight decay 0.01).
  - Scheduler: OneCycleLR (max LR 2e-3 across 100 epochs).
- **Logging**: Scalars for train/validation loss, validation accuracy, learning rate, written to TensorBoard.
- **Checkpointing**: Best model stored at `/models/best_model.pth` with associated metadata.

## Inference Service

- `main.py` spins up a Modal class (`AudioClassifier`) backed by the saved checkpoint.
- Incoming audio is base64-encoded WAV; the service:
  1. Decodes and resamples to **44.1 kHz** with librosa if needed.
  2. Generates Mel-spectrograms via the same preprocessing as training.
  3. Runs the CNN on GPU, returning **top-3 predictions** with confidences.
  4. Aggregates feature maps for downstream visualization (channel-mean, NaN-safe).
  5. Includes waveform excerpts (sub-sampled to ~8k points) for plotting.
- **API surface**: FastAPI `POST /` endpoint exposed through Modal’s serve URL.

The `app.local_entrypoint()` in `main.py` demonstrates a smoke test using `clap.wav` against the live endpoint.

## Visualization App

The Next.js project under `audio-cnn-visualisation/` is a separate workspace that:

- Fetches inference responses and renders probability bars, spectrogram matrices, and feature-map heatmaps.
- Utilizes Tailwind, Radix UI primitives, and custom components (`FeatureMap.tsx`, `Waveform.tsx`, `ColorScale.tsx`).
- Assumes the Modal inference endpoint is reachable; configure the base URL via environment variables or client config.

## Prerequisites

- Python 3.10+ with `venv` or `conda`.
- Node.js 18+ and npm (or pnpm) for the visualization app.
- Modal CLI (`pip install modal`) with an authenticated account.
- System libraries: `ffmpeg`, `libsndfile1` (Modal image installs them; install locally if running inference off-Modal).
- GPU access is optional locally but leveraged on Modal (`H200` for training, `A10G` for inference).

## Setup & Installation

### Python Environment

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install modal  # if not already included

### Modal Authentication

    modal token new

Follow the browser prompt to link your account. This is required before launching training/inference jobs.

### Frontend (Visualization)

    cd audio-cnn-visualisation
    npm install
    npm run dev  # http://localhost:3000

## Usage

### Train on Modal

    modal run train.py::train

- Downloads ESC-50 into a shared Modal volume (`esc50-data`).
- Writes checkpoints and TensorBoard runs to `/models`.
- Cancel/re-run safely; volumes retain previous downloads and best model state.

### Inspect Training Logs

    # Sync logs locally if desired:
    modal volume get esc-model tensorboard_logs ./tensorboard_logs_from_modal
    tensorboard --logdir ./tensorboard_logs_from_modal

### Serve Inference

    modal serve main.py::AudioClassifier

Modal provides a public URL for the FastAPI endpoint.

Use the `app.local_entrypoint()` helper to send a test request:

    modal run main.py::main

The script prints top predictions and waveform stats for `clap.wav`.

### Run the Visualization App

- Ensure the Modal inference endpoint URL is available to the frontend (e.g., via `.env` or API client config).
- Start the Next.js dev server (`npm run dev`) and load the UI to explore predictions and feature heatmaps.

## Development Tips

- Keep `requirements.txt` in sync with any local dependency additions; Modal builds the image from this file.
- Update `audio-cnn-visualisation/README.md` once the frontend wiring to the API is finalized.
- Use `modal deploy` if you want persistent endpoints/jobs without keeping the CLI session open.

## Troubleshooting

- **Missing ESC-50 data locally**: The Modal image handles downloads; for offline experimentation, mirror the same commands in `train.py`.
- **Libsndfile errors**: Install `libsndfile1` locally (`apt-get install libsndfile1` or equivalent).
- **CUDA unavailable**: Both scripts fall back to CPU automatically, but training will be slow.

## License
(MIT Copy right).

---

