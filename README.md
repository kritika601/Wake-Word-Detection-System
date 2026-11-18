# Marvin Wake Word Detection on QCC748

An on-device wake word detection running entirely on a QCC748 (RISC-V) embedded board — no cloud, no host PC required at runtime. Say **"Marvin"** and a GPIO-connected LED lights up, ready to trigger further application logic.

Built with **TensorFlow Lite Micro (TFLM)**, integrated from scratch into the QCCSDK-QCC74x SDK, with a custom continuous-capture audio pipeline and MFCC feature extraction matching `librosa`'s output closely enough to reuse a model trained entirely off-device.

---

## How it works

```
PDM Mic
  │
  ▼
AUADC (32-bit PDM capture) ──ping-pong DMA──► continuous, gapless capture
  │
  ▼
Downsample 32kHz → 16kHz
  │
  ▼
Rolling 2s circular history buffer
  │
  ▼
Energy-based window search (finds the most likely 1s speech window)
  │
  ▼
Simple audio cleanup (DC offset removal, high-pass, low-pass, light AGC)
  │
  ▼
MFCC feature extraction (13 coefficients × 32 frames, matches librosa defaults)
  │
  ▼
TensorFlow Lite Micro interpreter (int8 quantized CNN)
  │
  ▼
3-class softmax: Marvin / Unknown / Background noise
  │
  ▼
Confidence > threshold + class == Marvin  →  LED ON (GPIO)
```

The system listens **continuously** — recording never stops. A background DMA ping-pong buffer keeps the microphone capturing at all times, while a separate detection cycle periodically pulls the most recent second of audio out of a rolling history buffer and runs it through the model. No fixed "listening window" the user has to time their speech against.

---

## Model

- **Architecture**: small CNN (2× Conv2D + MaxPool2D, followed by Dense layers), trained in Keras/TensorFlow.
- **Input**: 13×32 MFCC features (matches `librosa.feature.mfcc` defaults: `n_fft=2048`, `hop_length=512`, `n_mels=128`, `n_mfcc=13`, `sr=16000`).
- **Classes**: `0 = Marvin`, `1 = Unknown` (other spoken words), `2 = Background noise`.
- **Training data**: Google Speech Commands dataset — "marvin" as the positive class, a random sample of ~30 other words as the negative "unknown" class, and background noise clips as the third class.
- **Quantization**: full integer (int8) post-training quantization via a representative dataset, for efficient on-device inference.
- **Deployment**: converted to `.tflite`, exported as a C byte array (`model_data.c`/`.h`), and run through TensorFlow Lite Micro's interpreter directly on-device.

The training notebook is designed to run in Google Colab and exports everything needed for on-device integration: the quantized model, the mel filterbank matrix, and the DCT-II matrix (both baked as C arrays so the MFCC computation can be replicated in C++ without needing `librosa` on-device).

---

## Hardware

- **Board**: QCC748M / QCC743 (RISC-V, QCCSDK-QCC74x)
- **Microphone**: PDM mic, captured via AUADC peripheral
- **Speaker**: PWM output via AUDAC (used for debug playback only — hear exactly what audio the model receives)
- **Indicator**: LED on a GPIO pin, lights up on wake word detection

---

## Repository structure

```
├── main.c                  # Audio capture, rolling buffer, detection state machine, GPIO
├── tinyml.cpp / tinyml.h    # TFLM interpreter setup and inference
├── audio_features.cpp/.h    # MFCC extraction, simple audio cleanup filters
├── model_data.c/.h          # Quantized model, exported as a C byte array
├── mel_filterbank.h         # Precomputed mel filterbank matrix (Python-exported)
├── dct_matrix.h             # Precomputed DCT-II matrix (Python-exported)
├── third_party/
│   └── tensorflow/          # TensorFlow Lite Micro runtime (trimmed reference build)
│   └── kissfft/              # FFT implementation used for MFCC computation
├── CMakeLists.txt
└── training/
    └── wakeUp_word.ipynb    # Model training notebook (Google Colab)
```

---

## Building

Requires the QCCSDK-QCC74x SDK and its bundled RISC-V toolchain.

```
cd <your project path>
make BOARD=qcc743dk
```

Flash the resulting `.elf` using your standard QCCSDK-QCC74x flashing workflow.

---

## Key implementation notes

- **TensorFlow Lite Micro** is not officially ported to the QCC74x SDK — the runtime was manually pulled from the [tflite-micro](https://github.com/tensorflow/tflite-micro) repository, trimmed to just the interpreter/reference-kernel source tree, and wired into this SDK's CMake build as a set of plain source files.
- **MFCC extraction is hand-implemented in C++** (FFT via KissFFT, mel filterbank + DCT applied via precomputed matrices exported from Python) to closely match `librosa`'s output, since the model was trained entirely on `librosa`-generated features.
- **Continuous capture** uses a ping-pong (double-buffered) DMA scheme on the AUADC peripheral — one buffer is always being filled while the other is processed, so the microphone is never idle.
- **Audio cleanup** uses simple, cheap time-domain filters (DC offset removal, one-pole high-pass ~100Hz, one-pole low-pass ~4kHz, light AGC) rather than spectral subtraction, which was found to be too aggressive for this signal's low SNR and ended up suppressing speech along with noise.

---

## Status / known limitations

- Trained purely on the public Google Speech Commands dataset — no samples of the deployment microphone/room/voice were used in training, so accuracy is meaningfully better on clear, close-mic speech than on quiet or distant utterances. Retraining with device-captured audio would likely improve real-world reliability.
- Detection cadence and confidence threshold are tunable (`DETECTION_INTERVAL_MS`, confidence threshold in `main.c`) and may need adjustment for different environments.

---


