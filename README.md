# RVC to ONNX Conversion Tool

This project provides tools for converting PyTorch RVC (Retrieval-based Voice Conversion) models to ONNX format.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd momentum-conversion-tool
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Convert RVC Model to ONNX

```bash
python convert_rvc_official.py <model.pth> [output.onnx] [--hidden-channels 256|768]
```

Examples:

```bash
# Basic conversion (auto-detects output name)
python convert_rvc_official.py model.pth

# Specify output file
python convert_rvc_official.py model.pth output.onnx

# Specify hidden channels (deprecated - auto-detected)
python convert_rvc_official.py model.pth output.onnx --hidden-channels 768
```

## Model Information

The converter supports both RVC v1 and v2 models:
- **v1**: 256-dimensional phoneme embeddings
- **v2**: 768-dimensional phoneme embeddings

The version is automatically detected from the model weights.

### ONNX Model Inputs

- `phone`: Phoneme/Hubert embeddings (batch, seq_len, 256 or 768)
- `phone_lengths`: Sequence lengths (batch,)
- `pitch`: F0 pitch in Hz (batch, seq_len)
- `pitchf`: Continuous F0 (batch, seq_len)
- `ds`: Speaker ID (batch,)
- `rnd`: Random noise (batch, 192, seq_len)

### ONNX Model Output

- `audio`: Converted audio waveform

## Notes

- The ONNX model requires proper preprocessing (phoneme extraction, F0 extraction)

