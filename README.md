# CSI-Sentinel v5.0

**Non-Line-of-Sight Semantic Wireless Sensing using Wi-Fi CSI**

Transform ambient Wi-Fi signals into a privacy-preserving, device-free perception system. CSI-Sentinel uses Channel State Information (CSI) from ESP32 chips to classify activities through walls using Deep Learning.

## Features

- **Wi-CLIP**: Zero-shot activity recognition via text-RF alignment
- **CSI-Diffusion**: Generative model for synthetic training data
- **Through-Wall Sensing**: NLOS activity detection using multipath
- **Privacy-Preserving**: No cameras, no wearables
- **Low-Cost Hardware**: < Rs. 3,000 total (ESP32-UE + antennas)

## Architecture

```
ESP32-TX ----[WiFi CSI]----> ESP32-RX ----[UDP]----> Laptop GPU
                                                         |
                                                    [DSP Pipeline]
                                                         |
                                                    [Wi-CLIP Model]
                                                         |
                                                    [Activity Label]
```

## Installation

```bash
git clone https://github.com/yourusername/CSI-Sentinel-v5.0.git
cd CSI-Sentinel-v5.0
pip install -r requirements.txt
pip install -e .
```

## Hardware Setup

### Components

| Component | Cost (INR) |
|-----------|------------|
| 2x ESP32-WROOM-32UE | 800-900 |
| 2x 2.4GHz Dipole Antennas | 400-500 |
| 1x DIY Yagi-Uda Antenna | 150-200 |
| Power Supply (TP4056 + 18650) | 500-600 |
| **Total** | **< 3,000** |

### Firmware Flashing

```bash
cd firmware/esp32_csi
idf.py set-target esp32
idf.py build
idf.py -p COM3 flash monitor
```

## Usage

### Start the Server

```bash
python main.py server
```

### Train Models

```bash
python main.py train --model wiclip
python main.py train --model diffusion
python main.py train --model classifier
```

### Generate Synthetic Data

```bash
python main.py generate --num-samples 1000 --activity fall
```

### Launch Dashboard

```bash
python main.py dashboard
```

Or directly:

```bash
streamlit run ui/dashboard.py
```

## Project Structure

```
CSI-Sentinel-v5.0/
├── firmware/           # ESP32 CSI collection firmware
│   └── esp32_csi/
├── server/             # UDP receiver & DSP pipeline
│   ├── csi_parser.py
│   ├── dsp_pipeline.py
│   ├── spectrogram.py
│   └── realtime_processor.py
├── models/             # Deep learning models
│   ├── rf_encoder.py
│   ├── text_encoder.py
│   ├── wi_clip.py
│   ├── csi_diffusion.py
│   └── classifier.py
├── training/           # Training scripts
│   ├── dataset.py
│   ├── augmentations.py
│   ├── train_wiclip.py
│   └── train_diffusion.py
├── ui/                 # Streamlit dashboard
│   ├── dashboard.py
│   └── visualizers.py
├── utils/              # Configuration & helpers
├── configs/            # YAML configuration
├── main.py             # CLI entry point
└── requirements.txt
```

## Technical Details

### DSP Pipeline

1. **Phase Sanitization**: Conjugate multiplication for CFO removal
2. **Hampel Filtering**: Outlier removal
3. **Butterworth Bandpass**: 0.3-80 Hz human motion band
4. **STFT**: Short-Time Fourier Transform for spectrograms

### Wi-CLIP Model

- **RF Encoder**: ResNet-style CNN with SE blocks
- **Text Encoder**: CLIP text transformer
- **Contrastive Loss**: InfoNCE with temperature scaling

### Activities Supported

- Walk, Run, Sit, Stand
- Fall (critical detection)
- Lie Down, Wave, Jump, Crouch
- Empty Room

## Configuration

Edit `configs/config.yaml`:

```yaml
network:
  udp_host: 0.0.0.0
  udp_port: 5500

csi:
  sample_rate: 200
  window_size: 200

inference:
  confidence_threshold: 0.7
```

## Author

**Ujjwal Manot**
Domain: Non-Line-of-Sight Semantic Wireless Sensing

## License

MIT License
