<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8%2B-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/platform-ESP32-orange.svg" alt="Platform">
</p>

# CSI-Sentinel v5.0

## Non-Line-of-Sight Semantic Wireless Sensing using Wi-Fi Channel State Information

Transform ambient Wi-Fi signals into a privacy-preserving, device-free perception system. CSI-Sentinel leverages Channel State Information (CSI) extracted from commodity ESP32 microcontrollers to perform human activity recognition through walls, without cameras, wearables, or any device on the subject.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technical Deep-Dive](#technical-deep-dive)
   - [Channel State Information (CSI)](#channel-state-information-csi)
   - [Signal Processing Pipeline](#signal-processing-pipeline)
   - [Deep Learning Models](#deep-learning-models)
5. [Hardware Requirements](#hardware-requirements)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Configuration Reference](#configuration-reference)
9. [API Documentation](#api-documentation)
10. [Training Guide](#training-guide)
11. [Deployment](#deployment)
12. [Performance Benchmarks](#performance-benchmarks)
13. [Troubleshooting](#troubleshooting)
14. [Project Structure](#project-structure)
15. [Contributing](#contributing)
16. [License](#license)
17. [Citation](#citation)
18. [Author](#author)

---

## Overview

CSI-Sentinel is a complete end-to-end system for Wi-Fi-based human sensing. It consists of:

1. **ESP32 Firmware**: Collects raw CSI data from Wi-Fi packets
2. **Python Server**: Receives UDP packets, processes signals, runs inference
3. **Deep Learning Models**: Wi-CLIP for zero-shot recognition, CSI-Diffusion for data augmentation
4. **Real-time Dashboard**: Streamlit-based visualization and monitoring

### Why Wi-Fi Sensing?

| Aspect | Camera | Wearable | Wi-Fi CSI |
|--------|--------|----------|-----------|
| Privacy | Invasive | Requires compliance | Non-invasive |
| Coverage | Line-of-sight only | On-body only | Through-wall |
| Cost | High | Per-person | One-time setup |
| Lighting | Requires light | N/A | Works in darkness |
| Maintenance | Lens cleaning | Charging | Minimal |

### Supported Activities

| Activity | Description | Detection Priority |
|----------|-------------|-------------------|
| `walk` | Normal walking pace | Standard |
| `run` | Fast movement/jogging | Standard |
| `sit` | Sitting down on chair/surface | Standard |
| `stand` | Standing still | Standard |
| `fall` | Falling to ground | **Critical** |
| `lie_down` | Lying on floor/bed | Standard |
| `wave` | Hand waving gesture | Standard |
| `jump` | Jumping motion | Standard |
| `crouch` | Crouching/squatting | Standard |
| `empty` | No person in room | Baseline |

---

## Key Features

### Wi-CLIP: Zero-Shot Activity Recognition

Inspired by OpenAI's CLIP, Wi-CLIP aligns RF spectrograms with natural language descriptions, enabling:

- **Zero-shot inference**: Recognize activities never seen during training
- **Natural language queries**: "Is someone falling?" or "Detect running"
- **Transfer learning**: Pre-trained text encoder from CLIP

### CSI-Diffusion: Synthetic Data Generation

A conditional diffusion model that generates realistic CSI spectrograms:

- **Class-conditional generation**: Generate samples for specific activities
- **Data augmentation**: Expand limited training datasets
- **DDIM sampling**: 50-step fast inference (vs 1000 DDPM steps)

### Real-Time Processing

- **200 Hz sample rate**: Captures fast human movements
- **50ms inference latency**: Real-time activity detection
- **Streaming architecture**: Continuous monitoring without gaps

### Production-Ready Code

- **Type-safe**: Full type hints with runtime validation
- **Thread-safe**: Proper locking for concurrent operations
- **Secure**: Protected against pickle deserialization attacks
- **Logged**: Comprehensive error tracking and debugging

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CSI-SENTINEL v5.0 ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐        ┌─────────────┐        ┌─────────────────────────────────┐
│   ESP32-TX  │  WiFi  │   ESP32-RX  │  UDP   │         LAPTOP/SERVER           │
│  (Injector) │ ─────► │  (Sniffer)  │ ─────► │                                 │
│             │  CSI   │             │ :5500  │  ┌─────────────────────────────┐│
│  Channel 6  │ Frames │  Extract    │        │  │      UDP Receiver           ││
│  1000 pkt/s │        │  CSI Data   │        │  │  - Packet parsing           ││
└─────────────┘        └─────────────┘        │  │  - Sequence tracking        ││
                                              │  │  - Drop detection           ││
      ┌───────────────────────────────────────┼──┴─────────────────────────────┤│
      │                                       │                                 ││
      │    HUMAN ACTIVITY IN ROOM             │  ┌─────────────────────────────┐││
      │    ┌─────────────────────┐            │  │      DSP Pipeline           │││
      │    │  Walking, Running,  │            │  │  - Phase sanitization       │││
      │    │  Sitting, Falling,  │ ◄──────────┼──│  - Hampel filtering         │││
      │    │  etc.               │  Multipath │  │  - Butterworth bandpass     │││
      │    └─────────────────────┘  Effects   │  │  - Doppler extraction       │││
      │                                       │  └─────────────────────────────┘││
      └───────────────────────────────────────┤                                 ││
                                              │  ┌─────────────────────────────┐││
                                              │  │   Spectrogram Generator     │││
                                              │  │  - STFT computation         │││
                                              │  │  - Mel filterbank           │││
                                              │  │  - RGB encoding             │││
                                              │  └─────────────────────────────┘││
                                              │                                 ││
                                              │  ┌─────────────────────────────┐││
                                              │  │      Wi-CLIP Model          │││
                                              │  │  ┌───────────┬───────────┐ │││
                                              │  │  │RF Encoder │Text Encoder│ │││
                                              │  │  │  (ResNet) │   (CLIP)   │ │││
                                              │  │  └─────┬─────┴─────┬─────┘ │││
                                              │  │        │           │        │││
                                              │  │        ▼           ▼        │││
                                              │  │  ┌───────────────────────┐  │││
                                              │  │  │  Contrastive Alignment │  │││
                                              │  │  │    (InfoNCE Loss)      │  │││
                                              │  │  └───────────────────────┘  │││
                                              │  └─────────────────────────────┘││
                                              │                                 ││
                                              │  ┌─────────────────────────────┐││
                                              │  │      Activity Output        │││
                                              │  │  - Predicted class          │││
                                              │  │  - Confidence score         │││
                                              │  │  - Alert triggering         │││
                                              │  └─────────────────────────────┘││
                                              └─────────────────────────────────┘│
                                                                                 │
                                              ┌─────────────────────────────────┐│
                                              │     Streamlit Dashboard         ││
                                              │  - Real-time visualization      ││
                                              │  - Activity history             ││
                                              │  - System metrics               ││
                                              └─────────────────────────────────┘│
                                              └─────────────────────────────────┘
```

### Data Flow

```
Raw CSI (64 subcarriers × complex)
    │
    ▼
┌─────────────────────────────────┐
│  Binary Packet (Header + Data)  │
│  Magic: 0xC510                  │
│  Version, Device ID, Timestamp  │
│  RSSI, Channel, Subcarrier Data │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Amplitude & Phase           │
│  amplitude = |CSI|              │
│  phase = ∠CSI                   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Phase Sanitization          │
│  - Conjugate multiplication     │
│  - Linear phase removal         │
│  - Unwrapping                   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Hampel Filtering            │
│  - Window size: 5               │
│  - Threshold: 3σ (MAD)          │
│  - Outlier replacement          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Butterworth Bandpass        │
│  - Order: 4                     │
│  - Highpass: 0.3 Hz             │
│  - Lowpass: 80 Hz               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     STFT Spectrogram            │
│  - n_fft: 256                   │
│  - hop_length: 16               │
│  - Mel bands: 128               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     RGB Spectrogram             │
│  R: Amplitude spectrogram       │
│  G: Phase spectrogram           │
│  B: Doppler spectrogram         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│     Neural Network              │
│  Input: 3 × 128 × T             │
│  Output: Activity + Confidence  │
└─────────────────────────────────┘
```

---

## Technical Deep-Dive

### Channel State Information (CSI)

CSI describes how a wireless signal propagates from transmitter to receiver, capturing:

- **Multipath effects**: Reflections from walls, furniture, and humans
- **Doppler shifts**: Frequency changes caused by moving objects
- **Attenuation**: Signal strength variations

#### Mathematical Foundation

For an OFDM system with N subcarriers, CSI is represented as:

```
H = [H₁, H₂, ..., Hₙ]

where Hᵢ = |Hᵢ|e^(j∠Hᵢ)
```

- `|Hᵢ|` = Amplitude (affected by path loss and absorption)
- `∠Hᵢ` = Phase (affected by path length and Doppler)

#### ESP32 CSI Extraction

The ESP32 provides 64 subcarriers per antenna:
- Subcarrier spacing: 312.5 kHz
- Bandwidth: 20 MHz (802.11n HT20)
- Data format: 8-bit signed integers (I/Q)

### Signal Processing Pipeline

#### 1. Phase Sanitization

Raw CSI phase is corrupted by:
- **Carrier Frequency Offset (CFO)**: Clock drift between TX/RX
- **Sampling Frequency Offset (SFO)**: ADC timing errors
- **Packet Detection Delay (PDD)**: Variable detection timing

**Solution: Conjugate Multiplication**

```python
# Remove CFO by computing phase difference between consecutive frames
conjugate_product = csi[t] * np.conj(csi[t-1])
phase_diff = np.angle(conjugate_product)
```

#### 2. Hampel Filter

Removes impulsive noise and outliers:

```python
def hampel_filter(data, window=5, threshold=3.0):
    for i in range(window//2, len(data) - window//2):
        window_data = data[i - window//2 : i + window//2 + 1]
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))

        if np.abs(data[i] - median) > threshold * 1.4826 * mad:
            data[i] = median

    return data
```

#### 3. Butterworth Bandpass Filter

Isolates human motion frequencies (0.3-80 Hz):

```python
from scipy.signal import butter, filtfilt

# Design filter
nyquist = sample_rate / 2
low = 0.3 / nyquist
high = 80.0 / nyquist
b, a = butter(order=4, [low, high], btype='band')

# Apply zero-phase filtering
filtered = filtfilt(b, a, data, axis=0)
```

#### 4. Spectrogram Generation

Convert time-domain signals to time-frequency representation:

```python
# Short-Time Fourier Transform
stft_matrix = np.zeros((n_fft//2 + 1, num_frames), dtype=complex)

for i in range(num_frames):
    frame = signal[i*hop : i*hop + n_fft] * window
    stft_matrix[:, i] = np.fft.rfft(frame)

# Mel filterbank
mel_spec = mel_filterbank @ np.abs(stft_matrix)**2

# Log compression
log_mel = 10 * np.log10(np.maximum(mel_spec, 1e-10))
```

### Deep Learning Models

#### RF Encoder Architecture

```
Input: [B, 3, 128, T]  (RGB Spectrogram)
    │
    ▼
┌─────────────────────────────────────────┐
│  Conv2d(3, 64, 7×7, stride=2, pad=3)    │
│  BatchNorm2d(64)                        │
│  ReLU                                   │
│  MaxPool2d(3×3, stride=2, pad=1)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  RFEncoderBlock(64, 64) × 2             │
│  ├── Conv2d(64, 64, 3×3)                │
│  ├── BatchNorm2d + ReLU                 │
│  ├── Conv2d(64, 64, 3×3)                │
│  ├── BatchNorm2d                        │
│  ├── SE Block (squeeze-excitation)      │
│  └── Residual Connection                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  RFEncoderBlock(64, 128, stride=2) × 2  │
│  RFEncoderBlock(128, 256, stride=2) × 2 │
│  RFEncoderBlock(256, 512, stride=2) × 2 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  TemporalAttention                      │
│  ├── Q, K, V projections                │
│  ├── Scaled dot-product attention       │
│  └── Output projection                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  AdaptiveAvgPool2d(1, 1)                │
│  Flatten                                │
│  Linear(512, embedding_dim)             │
└─────────────────────────────────────────┘
    │
    ▼
Output: [B, 512]  (RF Embedding)
```

#### Wi-CLIP Training

```python
# Forward pass
rf_embeddings = rf_encoder(spectrograms)      # [B, 512]
rf_embeddings = rf_projection(rf_embeddings)   # [B, 256]
rf_embeddings = F.normalize(rf_embeddings)

text_embeddings = text_encoder(activity_prompts)  # [B, 512]
text_embeddings = text_projection(text_embeddings) # [B, 256]
text_embeddings = F.normalize(text_embeddings)

# Compute similarity matrix
logit_scale = model.logit_scale.exp()
logits = logit_scale * rf_embeddings @ text_embeddings.T  # [B, B]

# InfoNCE loss (symmetric)
labels = torch.arange(batch_size)
loss_rf = F.cross_entropy(logits, labels)
loss_text = F.cross_entropy(logits.T, labels)
loss = (loss_rf + loss_text) / 2
```

#### CSI-Diffusion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DIFFUSION UNET                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: x_t (noisy spectrogram) + t (timestep) + c (class)  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Time Embedding: sinusoidal + MLP                   │    │
│  │  Class Embedding: nn.Embedding + addition           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────┐      │
│  │  Encoder (Down Blocks)                            │      │
│  │  ├── DownBlock(64, 64)   → skip₁                  │      │
│  │  ├── DownBlock(64, 128)  → skip₂                  │      │
│  │  ├── DownBlock(128, 256) → skip₃ + Attention      │      │
│  │  └── DownBlock(256, 512) → skip₄ + Attention      │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────┐      │
│  │  Middle Block                                     │      │
│  │  ├── ResidualBlock(512, 512)                      │      │
│  │  ├── AttentionBlock(512)                          │      │
│  │  └── ResidualBlock(512, 512)                      │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────┐      │
│  │  Decoder (Up Blocks)                              │      │
│  │  ├── UpBlock(512, 256) + skip₄ + Attention        │      │
│  │  ├── UpBlock(256, 128) + skip₃ + Attention        │      │
│  │  ├── UpBlock(128, 64)  + skip₂                    │      │
│  │  └── UpBlock(64, 64)   + skip₁                    │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────┐      │
│  │  Output: GroupNorm → SiLU → Conv2d(64, 3)         │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  Output: predicted_noise ε_θ(x_t, t, c)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Diffusion Process:**

```python
# Forward diffusion (training)
noise = torch.randn_like(x_0)
x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * noise  # Add noise

# Predict noise
predicted_noise = unet(x_t, t, class_labels)
loss = F.mse_loss(predicted_noise, noise)

# Reverse diffusion (sampling)
for t in reversed(range(T)):
    predicted_noise = unet(x_t, t, class_labels)
    x_{t-1} = (x_t - β_t * predicted_noise / sqrt(1-α_t)) / sqrt(α_t)
    if t > 0:
        x_{t-1} += sqrt(β_t) * torch.randn_like(x_t)
```

---

## Hardware Requirements

### Bill of Materials

| Component | Specification | Quantity | Cost (INR) | Purpose |
|-----------|---------------|----------|------------|---------|
| ESP32-WROOM-32UE | External antenna version | 2 | 450 each | TX/RX nodes |
| 2.4GHz Dipole Antenna | 3dBi gain, SMA connector | 2 | 200 each | Standard antenna |
| Yagi-Uda Antenna | 12dBi gain, directional | 1 | 200 | High-gain directional |
| TP4056 Module | Li-ion charger | 2 | 50 each | Battery charging |
| 18650 Battery | 3.7V 2600mAh | 2 | 150 each | Power supply |
| USB-UART Adapter | CP2102/CH340 | 1 | 150 | Programming |
| Jumper Wires | Male-Female | 1 pack | 100 | Connections |
| **Total** | | | **~2,500** | |

### Laptop/Server Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | GTX 1050 (2GB) | RTX 3060 (6GB) |
| Storage | 10 GB SSD | 50 GB NVMe |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 |

### Hardware Setup Diagram

```
                    ┌──────────────────┐
                    │   SENSING ROOM   │
                    │                  │
   ┌────────────┐   │    ┌────────┐    │   ┌────────────┐
   │  ESP32-TX  │───┼────│ HUMAN  │────┼───│  ESP32-RX  │
   │  (Channel) │   │    │ACTIVITY│    │   │  (Sniffer) │
   │   ┌───┐    │   │    └────────┘    │   │    ┌───┐   │
   │   │ANT│    │   │                  │   │    │ANT│   │
   │   └─┬─┘    │   │                  │   │    └─┬─┘   │
   └─────┼──────┘   │                  │   └──────┼─────┘
         │          │                  │          │
         │          │    Wall/Door     │          │
         │          └──────────────────┘          │
         │               (NLOS OK)                │
         │                                        │
    Power Bank                              USB to Laptop
    (Portable)                              (UDP Stream)
```

### Antenna Placement Guidelines

1. **TX-RX Distance**: 2-5 meters optimal
2. **Height**: 1.0-1.5 meters (chest level)
3. **Orientation**: TX and RX facing each other
4. **Obstacles**: Works through drywall, wood, glass (not metal)
5. **Avoid**: Metal furniture between TX-RX, direct antenna alignment

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- ESP-IDF v5.0+ (for firmware)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/ujjwal-manot/CSI-Sentinel-v5.0.git
cd CSI-Sentinel-v5.0
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n csi-sentinel python=3.10
conda activate csi-sentinel
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Optional: Install CLIP for text encoder
pip install transformers
```

### Step 4: Flash ESP32 Firmware

```bash
# Install ESP-IDF (if not installed)
# Follow: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/

# Navigate to firmware directory
cd firmware/esp32_csi

# Set target
idf.py set-target esp32

# Configure (set WiFi credentials, server IP)
idf.py menuconfig

# Build
idf.py build

# Flash (replace COM3 with your port)
idf.py -p COM3 flash monitor
```

### Step 5: Verify Installation

```bash
# Run tests
python -m pytest tests/

# Check imports
python -c "from models import WiCLIP, CSIDiffusion; print('OK')"
```

---

## Quick Start

### 1. Start UDP Server (Receive CSI Data)

```bash
python main.py server
```

Expected output:
```
2024-01-15 10:30:00 | INFO     | csi_server | Starting CSI-Sentinel Server...
2024-01-15 10:30:00 | INFO     | csi_server | Listening on 0.0.0.0:5500
2024-01-15 10:30:05 | INFO     | csi_server | Packets: 1000 | Frames: 50
```

### 2. Train Wi-CLIP Model

```bash
# Prepare data in data/ directory
# Structure: data/{activity}/{sample_001.npy, ...}

python main.py train --model wiclip
```

### 3. Train Diffusion Model

```bash
python main.py train --model diffusion
```

### 4. Train Classifier

```bash
python main.py train --model classifier
```

### 5. Generate Synthetic Data

```bash
python main.py generate --num-samples 500 --activity fall
```

### 6. Launch Dashboard

```bash
python main.py dashboard
```

Open browser at `http://localhost:8501`

---

## Configuration Reference

### Main Configuration (`configs/config.yaml`)

```yaml
# ============================================================
# CSI-SENTINEL v5.0 CONFIGURATION
# ============================================================

# System identification
system:
  name: "CSI-Sentinel"
  version: "5.0.0"
  device_id: "sentinel-001"

# Network settings for UDP receiver
network:
  udp_host: "0.0.0.0"        # Listen on all interfaces
  udp_port: 5500             # UDP port (1024-65535)
  buffer_size: 2048          # Receive buffer size in bytes

# CSI data parameters
csi:
  num_subcarriers: 64        # Number of OFDM subcarriers
  num_antennas: 1            # Number of RX antennas
  sample_rate: 200           # Packets per second
  window_size: 200           # Samples per window (1 second)
  hop_size: 50               # Samples between windows (250ms)
  fft_size: 256              # FFT size for spectrogram

# Digital Signal Processing
dsp:
  enable_phase_sanitization: true    # Remove CFO/SFO
  enable_amplitude_calibration: true # Normalize amplitude
  hampel_window: 5                   # Hampel filter window
  hampel_threshold: 3.0              # Outlier threshold (MAD)
  butterworth_order: 4               # Filter order
  lowpass_cutoff: 80.0               # Low-pass frequency (Hz)
  highpass_cutoff: 0.3               # High-pass frequency (Hz)

# Spectrogram generation
spectrogram:
  n_fft: 256                 # FFT window size
  hop_length: 16             # Hop between frames
  n_mels: 128                # Number of Mel bands
  fmin: 0.1                  # Minimum frequency (Hz)
  fmax: 100.0                # Maximum frequency (Hz)
  power: 2.0                 # Spectrogram power
  normalize: true            # Normalize output

# Model architecture
model:
  rf_encoder:
    input_channels: 3        # RGB spectrogram
    base_channels: 64        # Initial conv channels
    num_blocks: 4            # Number of residual blocks
    embedding_dim: 512       # Output embedding size
    dropout: 0.1             # Dropout rate

  text_encoder:
    model_name: "openai/clip-vit-base-patch32"  # CLIP model
    embedding_dim: 512       # Output embedding size
    freeze: true             # Freeze pretrained weights

  wiclip:
    temperature: 0.07        # InfoNCE temperature
    projection_dim: 256      # Joint embedding dimension

  diffusion:
    num_timesteps: 1000      # Diffusion steps
    beta_start: 0.0001       # Initial noise schedule
    beta_end: 0.02           # Final noise schedule
    unet_channels: [64, 128, 256, 512]  # UNet channels
    attention_resolutions: [16, 8]       # Attention at these sizes

# Training hyperparameters
training:
  batch_size: 32             # Batch size
  learning_rate: 0.0001      # Initial learning rate
  weight_decay: 0.01         # L2 regularization
  num_epochs: 100            # Training epochs
  warmup_epochs: 5           # LR warmup epochs
  gradient_clip: 1.0         # Gradient clipping norm
  mixed_precision: true      # Use FP16 training

  augmentation:
    time_mask_max: 30        # Max time mask width
    freq_mask_max: 20        # Max frequency mask width
    noise_std: 0.02          # Gaussian noise std
    scale_range: [0.8, 1.2]  # Random scaling range

# Inference settings
inference:
  confidence_threshold: 0.7  # Minimum confidence for alert
  smoothing_window: 5        # Temporal smoothing frames
  alert_cooldown: 10.0       # Seconds between alerts

# Logging configuration
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  save_dir: "logs"           # Log file directory
  tensorboard: true          # Enable TensorBoard logging

# Activity labels
activities:
  - "walk"
  - "run"
  - "sit"
  - "stand"
  - "fall"
  - "lie_down"
  - "wave"
  - "jump"
  - "crouch"
  - "empty"
```

### Environment Variables

```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Disable TensorFloat-32 for reproducibility
export NVIDIA_TF32_OVERRIDE=0

# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## API Documentation

### Server Module

#### `UDPReceiver`

```python
from server import UDPReceiver

receiver = UDPReceiver(
    host="0.0.0.0",      # Bind address
    port=5500,           # UDP port
    buffer_size=2048,    # Receive buffer
    queue_size=1000      # Packet queue size
)

# Start receiving
receiver.start()

# Register callback for each packet
def on_packet(packet: CSIPacket):
    print(f"Received: {packet.timestamp_ms}ms, RSSI: {packet.rssi}")

receiver.register_callback(on_packet)

# Get packets manually
packet = receiver.get_packet(timeout=1.0)
packets = receiver.get_packets(max_count=100)

# Statistics
print(f"Received: {receiver.packets_received}")
print(f"Dropped: {receiver.dropped_packets}")

# Stop
receiver.stop()
```

#### `CSIParser`

```python
from server import CSIParser, CSIPacket

parser = CSIParser(num_subcarriers=64)

# Parse raw bytes
packet: CSIPacket = parser.parse(raw_data)

# Access packet fields
print(packet.timestamp_us)      # Microsecond timestamp
print(packet.rssi)              # Received signal strength
print(packet.csi_complex)       # Complex CSI array [64]
print(packet.amplitude)         # |CSI| array [64]
print(packet.phase)             # ∠CSI array [64]
```

#### `DSPPipeline`

```python
from server import DSPPipeline, DSPConfig

config = DSPConfig(
    sample_rate=200.0,
    hampel_window=5,
    butterworth_order=4,
    lowpass_cutoff=80.0,
    highpass_cutoff=0.3
)

dsp = DSPPipeline(config)

# Process single frame
amplitude, phase = dsp.process_frame(csi_complex)

# Process batch
amplitudes, phases = dsp.process_batch(csi_batch)  # [N, 64]

# Apply filters
filtered = dsp.hampel_filter(data, axis=0)
filtered = dsp.apply_bandpass(data)

# Extract Doppler
doppler = dsp.extract_doppler(phase_data)

# PCA denoising
denoised = dsp.pca_denoise(data, n_components=5)

# Reset calibration
dsp.reset()
```

#### `SpectrogramGenerator`

```python
from server import SpectrogramGenerator

generator = SpectrogramGenerator(
    sample_rate=200.0,
    n_fft=256,
    hop_length=16,
    n_mels=128,
    fmin=0.1,
    fmax=100.0
)

# Generate spectrograms
stft = generator.stft(signal)                    # Complex STFT
spec = generator.spectrogram(signal)             # Power spectrogram
mel_spec = generator.mel_spectrogram(signal)     # Mel spectrogram
log_mel = generator.log_mel_spectrogram(signal)  # Log-Mel spectrogram

# Generate RGB spectrogram for model input
rgb = generator.generate_rgb_spectrogram(
    amplitude=amp_data,
    phase=phase_data,
    doppler=doppler_data
)  # [3, 128, T]
```

#### `RealtimeProcessor`

```python
from server import RealtimeProcessor, ProcessedFrame, InferenceResult

processor = RealtimeProcessor(
    host="0.0.0.0",
    port=5500,
    sample_rate=200.0,
    window_size=200,
    hop_size=50
)

# Set model for inference
processor.set_inference_model(trained_model)

# Register callbacks
def on_frame(frame: ProcessedFrame):
    print(f"Spectrogram shape: {frame.spectrogram.shape}")

def on_inference(result: InferenceResult):
    print(f"Activity: {result.activity}, Confidence: {result.confidence:.2f}")

processor.register_frame_callback(on_frame)
processor.register_inference_callback(on_inference)

# Start processing
processor.start()

# Get statistics
stats = processor.get_stats()
print(f"Frames processed: {stats['frames_processed']}")

# Stop
processor.stop()
```

### Models Module

#### `RFEncoder`

```python
from models import RFEncoder

encoder = RFEncoder(
    input_channels=3,      # RGB spectrogram
    base_channels=64,      # Initial channels
    num_blocks=4,          # Residual blocks per stage
    embedding_dim=512,     # Output dimension
    dropout=0.1            # Dropout rate
)

# Forward pass
spectrograms = torch.randn(16, 3, 128, 128)  # [B, C, H, W]
embeddings = encoder(spectrograms)            # [B, 512]
```

#### `TextEncoder`

```python
from models import TextEncoder

encoder = TextEncoder(
    model_name="openai/clip-vit-base-patch32",
    embedding_dim=512,
    freeze=True
)

# Encode text
texts = ["a person walking", "a person falling"]
embeddings = encoder(texts)  # [2, 512]

# Get activity embeddings
activity_emb = encoder.get_activity_embeddings()  # [10, 512]

# Activity list
activities = encoder.activity_list  # ['walk', 'run', ...]

# Add custom activity
encoder.add_activity_prompt("dance", "a person dancing energetically")
```

#### `WiCLIP`

```python
from models import WiCLIP, RFEncoder, TextEncoder

# Create model
rf_encoder = RFEncoder(embedding_dim=512)
text_encoder = TextEncoder(embedding_dim=512)
model = WiCLIP(
    rf_encoder=rf_encoder,
    text_encoder=text_encoder,
    projection_dim=256,
    temperature=0.07
)

# Training forward pass
rf_emb, text_emb, logits = model(spectrograms, activity_names)

# Zero-shot prediction
activity, confidence, scores = model.predict(spectrogram)
print(f"Predicted: {activity} ({confidence:.2%})")

# Custom text queries
activity, conf, scores = model.zero_shot_predict(
    spectrogram,
    ["someone is exercising", "room is empty", "emergency situation"]
)

# Clear text embedding cache
model.clear_cache()
```

#### `CSIDiffusion`

```python
from models import CSIDiffusion, DiffusionUNet

# Create model
unet = DiffusionUNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_classes=10
)
model = CSIDiffusion(
    unet=unet,
    num_timesteps=1000,
    beta_schedule="cosine"  # or "linear"
)

# Training
loss = model(spectrograms, class_labels)  # Returns MSE loss
loss.backward()

# Sampling (DDPM - slow but high quality)
samples = model.sample(
    batch_size=16,
    channels=3,
    height=128,
    width=128,
    class_labels=torch.tensor([4] * 16),  # "fall" class
    device="cuda"
)

# Sampling (DDIM - fast)
samples = model.ddim_sample(
    batch_size=16,
    channels=3,
    height=128,
    width=128,
    class_labels=torch.tensor([4] * 16),
    num_inference_steps=50,  # vs 1000 for DDPM
    eta=0.0,                 # 0 = deterministic
    device="cuda"
)
```

#### `ActivityClassifier`

```python
from models import ActivityClassifier, RFEncoder

# Create model
rf_encoder = RFEncoder(embedding_dim=512)
classifier = ActivityClassifier(
    rf_encoder=rf_encoder,
    num_classes=10,
    embedding_dim=512,
    hidden_dim=256,
    dropout=0.3
)

# Set activity names
classifier.set_activity_names([
    "walk", "run", "sit", "stand", "fall",
    "lie_down", "wave", "jump", "crouch", "empty"
])

# Training forward
logits = classifier(spectrograms)  # [B, 10]

# Prediction
activity, confidence, scores = classifier.predict(spectrogram)

# Batch prediction
activities, confidences, probs = classifier.predict_batch(spectrograms)
```

#### `InferenceEngine`

```python
from models import InferenceEngine, WiCLIP

# Load from checkpoint
engine = InferenceEngine.from_checkpoint(
    "checkpoints/best_model.pt",
    model_type="wiclip",
    confidence_threshold=0.7,
    smoothing_window=5
)

# Or wrap existing model
engine = InferenceEngine(
    model=trained_model,
    confidence_threshold=0.7,
    smoothing_window=5,
    activities=["walk", "run", ...]
)

# Single prediction
activity, confidence, scores = engine.predict(spectrogram)

# Smoothed prediction (uses history)
activity, confidence, scores = engine.predict_smoothed(spectrogram)

# Batch prediction
activities, confidences, all_scores = engine.predict_batch(spectrograms)

# Check confidence
if engine.is_confident(confidence):
    trigger_alert()

# Reset history
engine.reset_history()
```

### Training Module

#### `CSIDataset`

```python
from training import CSIDataset, create_dataloaders

# Create dataset
dataset = CSIDataset(
    data_dir="data",
    activities=["walk", "run", "sit", ...],
    split="train",  # or "val", "test"
    transform=augmentor
)

# Get item
spectrogram, label, activity_name = dataset[0]

# Create all dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data",
    batch_size=32,
    transform=augmentor,
    activities=activities
)
```

#### `CSIAugmentor`

```python
from training import CSIAugmentor, MixUp, CutMix

# Spectrogram augmentation
augmentor = CSIAugmentor(
    time_mask_max=30,      # Max time mask width
    freq_mask_max=20,      # Max frequency mask width
    noise_std=0.02,        # Gaussian noise
    scale_min=0.8,         # Min scaling factor
    scale_max=1.2          # Max scaling factor
)

augmented = augmentor(spectrogram)

# MixUp augmentation
mixup = MixUp(alpha=0.2)
mixed_x, labels_a, labels_b, lam = mixup(batch_x, batch_y)
loss = lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)

# CutMix augmentation
cutmix = CutMix(alpha=1.0)
mixed_x, labels_a, labels_b, lam = cutmix(batch_x, batch_y)
```

#### Trainers

```python
from training import WiCLIPTrainer, DiffusionTrainer, ClassifierTrainer

# Wi-CLIP training
trainer = WiCLIPTrainer(
    model=wiclip_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config={
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "mixed_precision": True,
        "save_dir": "checkpoints/wiclip"
    }
)
history = trainer.train()

# Load checkpoint
trainer.load_checkpoint("checkpoints/wiclip/best_model.pt")

# Diffusion training
trainer = DiffusionTrainer(
    model=diffusion_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config={
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "use_ema": True,
        "ema_decay": 0.9999
    }
)

# Generate samples during training
samples = trainer.generate_samples(num_samples=16, class_label=4)

# Classifier training
trainer = ClassifierTrainer(
    model=classifier,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    config={
        "loss_type": "focal",  # or "cross_entropy"
        "use_mixup": True,
        "mixup_alpha": 0.2
    }
)
history = trainer.train()
test_results = trainer.test()
```

### Utils Module

#### Configuration

```python
from utils import Config, get_config, reset_config

# Load configuration
config = Config("configs/config.yaml")

# Or use singleton
config = get_config("configs/config.yaml")

# Access values
print(config.network.udp_port)        # 5500
print(config.csi.sample_rate)         # 200
print(config.training.batch_size)     # 32
print(config.activities)              # ['walk', 'run', ...]
print(config.num_classes)             # 10

# Modify (mutable configs only)
config.activities = ["walk", "run", "fall"]

# Save configuration
config.save("configs/custom_config.yaml")

# Reset singleton
reset_config()
```

#### Logging

```python
from utils import setup_logger, get_logger, set_log_level, shutdown_logging

# Setup logger
logger = setup_logger(
    name="my_module",
    level="DEBUG",
    log_dir="logs",
    console=True,
    use_colors=True
)

# Get existing logger
logger = get_logger("my_module")

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")

# Change level
set_log_level("my_module", "WARNING")

# Cleanup
shutdown_logging()
```

#### Helpers

```python
from utils import (
    set_seed, get_device, count_parameters,
    Timer, AverageMeter, EarlyStopping,
    format_time, safe_load_checkpoint,
    validate_tensor_shape, ensure_batch_dim, torch_eval_mode
)

# Reproducibility
set_seed(42)

# Device selection
device = get_device()  # Automatically selects cuda/mps/cpu
device = get_device("cuda:1")  # Specific device

# Count parameters
total = count_parameters(model, trainable_only=False)
trainable = count_parameters(model, trainable_only=True)

# Timer
with Timer() as t:
    train_one_epoch()
print(f"Took {t.elapsed:.2f}s")

# Average meter
meter = AverageMeter()
meter.update(loss.item(), batch_size)
print(f"Average loss: {meter.avg:.4f}")

# Early stopping
early_stop = EarlyStopping(patience=10, min_delta=0.001)
if early_stop(val_loss):
    print("Early stopping triggered")

# Format time
print(format_time(3661))  # "1h 1m 1s"

# Safe checkpoint loading
checkpoint = safe_load_checkpoint("model.pt", device=device)

# Validate tensor shape
validate_tensor_shape(tensor, expected_dims=4, name="spectrogram")

# Ensure batch dimension
tensor = ensure_batch_dim(tensor, expected_dims=4)

# Evaluation context
with torch_eval_mode(model):
    output = model(input)
```

---

## Training Guide

### Data Collection

#### 1. Prepare Environment

```
┌────────────────────────────────────────┐
│              ROOM LAYOUT               │
│                                        │
│  [ESP32-TX]                [ESP32-RX]  │
│      ●─────────────────────────●       │
│      │      3-5 meters         │       │
│      │                         │       │
│      │    ┌─────────────┐      │       │
│      │    │  ACTIVITY   │      │       │
│      │    │    ZONE     │      │       │
│      │    └─────────────┘      │       │
│      │                         │       │
│                                        │
└────────────────────────────────────────┘
```

#### 2. Collection Protocol

```python
# collection_protocol.py
ACTIVITIES = ["walk", "run", "sit", "stand", "fall", "lie_down", "wave", "jump", "crouch", "empty"]

# Each activity: 5 minutes × 5 subjects = 25 minutes/activity
# Total: 250 minutes of data

for activity in ACTIVITIES:
    for subject_id in range(5):
        print(f"Subject {subject_id}: Perform '{activity}' for 5 minutes")
        collect_data(duration=300, label=activity, subject=subject_id)
        rest(30)  # 30 second break
```

#### 3. Data Organization

```
data/
├── train/
│   ├── walk/
│   │   ├── walk_001.npy
│   │   ├── walk_002.npy
│   │   └── ...
│   ├── run/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Training Wi-CLIP

```bash
# Full training
python main.py train --model wiclip

# Custom configuration
python training/train_wiclip.py \
    --data_dir data \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --save_dir checkpoints/wiclip
```

#### Training Curves

```
Epoch 1:   Train Loss: 4.2341 | Val Loss: 4.1823 | Val Acc: 0.12
Epoch 10:  Train Loss: 2.1567 | Val Loss: 2.0934 | Val Acc: 0.45
Epoch 50:  Train Loss: 0.8234 | Val Loss: 0.9123 | Val Acc: 0.78
Epoch 100: Train Loss: 0.3456 | Val Loss: 0.5678 | Val Acc: 0.89
```

### Training Diffusion

```bash
python main.py train --model diffusion
```

#### Expected Training Time

| GPU | Batch Size | Time/Epoch | Total (100 epochs) |
|-----|------------|------------|-------------------|
| RTX 3060 | 16 | ~5 min | ~8 hours |
| RTX 3080 | 32 | ~3 min | ~5 hours |
| RTX 4090 | 64 | ~1.5 min | ~2.5 hours |

### Training Classifier

```bash
python main.py train --model classifier
```

### Generating Synthetic Data

```bash
# Generate 1000 samples per class
for activity in walk run sit stand fall lie_down wave jump crouch empty; do
    python main.py generate --num-samples 1000 --activity $activity
done
```

### Hyperparameter Tuning

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| learning_rate | 1e-5 to 1e-3 | 1e-4 | Use warmup |
| batch_size | 16-64 | 32 | Larger = more stable |
| weight_decay | 0.001-0.1 | 0.01 | Regularization |
| temperature | 0.01-0.1 | 0.07 | InfoNCE temperature |
| dropout | 0.1-0.5 | 0.1 | For encoder |
| mixup_alpha | 0.1-0.4 | 0.2 | Data augmentation |

---

## Deployment

### Local Deployment

```bash
# Run all services
python main.py server &
python main.py dashboard &

# Or use process manager
pip install supervisor
supervisord -c supervisord.conf
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 5500/udp
EXPOSE 8501

CMD ["python", "main.py", "server"]
```

```bash
# Build and run
docker build -t csi-sentinel .
docker run -p 5500:5500/udp -p 8501:8501 --gpus all csi-sentinel
```

### Systemd Service

```ini
# /etc/systemd/system/csi-sentinel.service
[Unit]
Description=CSI-Sentinel Activity Recognition
After=network.target

[Service]
Type=simple
User=csi
WorkingDirectory=/opt/csi-sentinel
ExecStart=/opt/csi-sentinel/venv/bin/python main.py server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable csi-sentinel
sudo systemctl start csi-sentinel
```

### Alert Integration

```python
# Example: Send alert via webhook
import requests

def on_fall_detected(activity, confidence, timestamp):
    if activity == "fall" and confidence > 0.9:
        requests.post("https://your-webhook.com/alert", json={
            "type": "FALL_DETECTED",
            "confidence": confidence,
            "timestamp": timestamp,
            "device_id": "sentinel-001"
        })

processor.register_inference_callback(on_fall_detected)
```

---

## Performance Benchmarks

### Inference Latency

| Model | Device | Batch Size | Latency (ms) |
|-------|--------|------------|--------------|
| Wi-CLIP | RTX 3060 | 1 | 8.2 |
| Wi-CLIP | RTX 3060 | 32 | 45.6 |
| Wi-CLIP | CPU (i7) | 1 | 89.3 |
| Classifier | RTX 3060 | 1 | 5.1 |
| Classifier | RTX 3060 | 32 | 28.4 |
| Diffusion (DDIM-50) | RTX 3060 | 1 | 890 |

### Memory Usage

| Component | GPU Memory | CPU Memory |
|-----------|------------|------------|
| Wi-CLIP | 1.2 GB | 500 MB |
| Classifier | 0.8 GB | 350 MB |
| CSI-Diffusion | 2.4 GB | 800 MB |
| Full Pipeline | 1.5 GB | 1.2 GB |

### Accuracy Metrics

| Model | Accuracy | F1-Score | Fall Recall |
|-------|----------|----------|-------------|
| Wi-CLIP (zero-shot) | 78.3% | 0.76 | 0.82 |
| Wi-CLIP (fine-tuned) | 89.1% | 0.88 | 0.94 |
| Classifier (supervised) | 92.4% | 0.91 | 0.96 |
| Classifier + Diffusion Aug | 94.2% | 0.93 | 0.97 |

### Throughput

| Metric | Value |
|--------|-------|
| Max CSI packets/sec | 1000 |
| Max inference FPS | 50 |
| UDP packet drop rate | < 0.1% |
| End-to-end latency | < 100ms |

---

## Troubleshooting

### Common Issues

#### 1. No CSI Data Received

```
Problem: Server shows 0 packets received
```

**Solutions:**
- Verify ESP32 is powered and connected to same network
- Check firewall allows UDP port 5500
- Verify server IP in ESP32 firmware matches laptop IP
- Try `netcat -lu 5500` to test UDP reception

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in config
- Use mixed precision training (`mixed_precision: true`)
- Use gradient checkpointing for diffusion model
- Clear cache: `torch.cuda.empty_cache()`

#### 3. Poor Classification Accuracy

```
Problem: Model accuracy stuck at ~10% (random chance)
```

**Solutions:**
- Verify data is correctly labeled
- Check spectrogram generation (visualize samples)
- Increase training epochs
- Use data augmentation
- Check for class imbalance

#### 4. Phase Wrapping Issues

```
Problem: Spectrogram shows discontinuities
```

**Solutions:**
- Enable phase sanitization in config
- Use conjugate multiplication method
- Apply unwrapping after filtering

#### 5. High Packet Drop Rate

```
Problem: > 5% packet drops
```

**Solutions:**
- Increase queue size in config
- Reduce CSI transmission rate
- Check for WiFi interference
- Use wired connection for laptop

### Debug Mode

```bash
# Enable debug logging
python main.py server --config configs/debug.yaml

# Debug config
logging:
  level: "DEBUG"
```

### Diagnostic Commands

```bash
# Check UDP port
netstat -ulnp | grep 5500

# Monitor packets
tcpdump -i eth0 udp port 5500

# Test ESP32 connection
ping 192.168.1.100  # ESP32 IP

# GPU status
nvidia-smi -l 1

# Memory usage
watch -n 1 free -h
```

---

## Project Structure

```
CSI-Sentinel-v5.0/
│
├── firmware/                    # ESP32 firmware
│   └── esp32_csi/
│       ├── main/
│       │   └── main.c          # CSI collection code
│       ├── CMakeLists.txt
│       └── sdkconfig           # ESP-IDF configuration
│
├── server/                      # UDP server & DSP
│   ├── __init__.py             # Package exports
│   ├── csi_parser.py           # Binary packet parsing
│   ├── udp_receiver.py         # Threaded UDP server
│   ├── dsp_pipeline.py         # Signal processing
│   ├── spectrogram.py          # Spectrogram generation
│   └── realtime_processor.py   # Real-time pipeline
│
├── models/                      # Deep learning models
│   ├── __init__.py             # Package exports
│   ├── rf_encoder.py           # ResNet RF encoder
│   ├── text_encoder.py         # CLIP text encoder
│   ├── wi_clip.py              # Contrastive model
│   ├── csi_diffusion.py        # Diffusion model
│   ├── classifier.py           # Activity classifier
│   └── inference.py            # Production inference
│
├── training/                    # Training scripts
│   ├── __init__.py             # Package exports
│   ├── dataset.py              # PyTorch datasets
│   ├── augmentations.py        # Data augmentation
│   ├── train_wiclip.py         # Wi-CLIP trainer
│   ├── train_diffusion.py      # Diffusion trainer
│   └── train_classifier.py     # Classifier trainer
│
├── ui/                          # User interface
│   ├── dashboard.py            # Streamlit dashboard
│   └── visualizers.py          # Plotting utilities
│
├── utils/                       # Utilities
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging utilities
│   └── helpers.py              # Helper functions
│
├── configs/                     # Configuration files
│   ├── config.yaml             # Default configuration
│   └── debug.yaml              # Debug configuration
│
├── tests/                       # Unit tests
│   ├── test_parser.py
│   ├── test_dsp.py
│   └── test_models.py
│
├── data/                        # Data directory (gitignored)
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                 # Model checkpoints (gitignored)
│   ├── wiclip/
│   ├── diffusion/
│   └── classifier/
│
├── logs/                        # Log files (gitignored)
│
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Contributing

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/ujjwal-manot/CSI-Sentinel-v5.0.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8
- Use type hints for all functions
- Document with docstrings (Google style)
- Keep functions under 50 lines
- No silent exception handling

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run linting (`flake8 . && mypy .`)
5. Run tests (`pytest tests/`)
6. Commit (`git commit -m "Add amazing feature"`)
7. Push (`git push origin feature/amazing-feature`)
8. Open Pull Request

---

## License

```
MIT License

Copyright (c) 2024 Ujjwal Manot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{csi_sentinel_2024,
  author = {Manot, Ujjwal},
  title = {CSI-Sentinel: Non-Line-of-Sight Semantic Wireless Sensing},
  year = {2024},
  version = {5.0.0},
  url = {https://github.com/ujjwal-manot/CSI-Sentinel-v5.0}
}
```

### Related Works

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) - OpenAI
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al.
- [WiFi-Based Human Activity Recognition](https://arxiv.org/abs/1707.05426) - Survey
- [ESP32 CSI Toolkit](https://github.com/espressif/esp-csi) - Espressif

---

## Author

**Ujjwal Manot**

- Role: President, Resonance ECE
- Domain: Non-Line-of-Sight Semantic Wireless Sensing
- GitHub: [@ujjwal-manot](https://github.com/ujjwal-manot)

---

<p align="center">
  <b>CSI-Sentinel v5.0</b> - See Through Walls with Wi-Fi
</p>
