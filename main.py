#!/usr/bin/env python3
"""
RVC to ONNX Converter using Official RVC Architecture
Based on RVC-Project/Retrieval-based-Voice-Conversion-WebUI
"""

import warnings
import torch
import sys
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from rvc_infer.models_onnx import SynthesizerTrnMsNSFsidM


def convert_rvc_to_onnx(model_path, output_path, hidden_channels=256):
    """
    Convert RVC .pth model to ONNX using official architecture

    Args:
        model_path: Path to .pth model file
        output_path: Path for output .onnx file
        hidden_channels: (Deprecated - auto-detected from model) Hidden channels parameter
    """
    print("=" * 60)
    print("RVC to ONNX Converter (Official Architecture)")
    print("=" * 60)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load checkpoint
    print("\nLoading checkpoint...")
    cpt = torch.load(model_path, map_location="cpu")

    # Update n_spk (number of speakers) from actual embedding weights
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    print(f"✓ Checkpoint loaded")
    print(f"  Sample rate: {cpt.get('sr', 'unknown')}")
    print(f"  F0: {cpt.get('f0', 'unknown')}")
    print(f"  Config: {cpt['config']}")

    config = cpt["config"]

    # Add version parameter if not present
    if len(config) == 18:
        if "enc_p.emb_phone.weight" in cpt["weight"]:
            phone_dim = cpt["weight"]["enc_p.emb_phone.weight"].shape[1]
            version = "v1" if phone_dim == 256 else "v2"
        else:
            version = "v1"  # Default
            phone_dim = 256
        config.append(version)
        print(f"  Detected version: {version}")
    else:
        version = config[-1] if len(config) > 18 else "v1"
        # Determine phone_dim from existing config or weights
        if "enc_p.emb_phone.weight" in cpt["weight"]:
            phone_dim = cpt["weight"]["enc_p.emb_phone.weight"].shape[1]
        else:
            phone_dim = 768 if version == "v2" else 256
        print(f"  Version: {version}")

    # Create model
    print("\nCreating model with official architecture...")
    net_g = SynthesizerTrnMsNSFsidM(*config, is_half=False)

    # Load weights
    print("Loading weights...")
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()
    print("✓ Model created and weights loaded")

    # Create dummy inputs for ONNX export
    print("\nPreparing ONNX export...")
    test_phone = torch.rand(1, 200, phone_dim)  # hidden unit (256 for v1, 768 for v2)
    test_phone_lengths = torch.tensor([200]).long()
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # F0 in Hz
    test_pitchf = torch.rand(1, 200)  # nsf F0
    test_ds = torch.LongTensor([0])  # speaker ID
    test_rnd = torch.rand(1, 192, 200)  # random noise

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]

    # Export to ONNX using legacy API
    print("Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            net_g,
            (
                test_phone,
                test_phone_lengths,
                test_pitch,
                test_pitchf,
                test_ds,
                test_rnd,
            ),
            output_path,
            export_params=True,
            dynamic_axes={
                "phone": [1],
                "pitch": [1],
                "pitchf": [1],
                "rnd": [2],
            },
            do_constant_folding=False,
            opset_version=16,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            # Use legacy exporter to avoid newer PyTorch issues
            dynamo=False,
        )

    print("✓ ONNX export complete!")
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nONNX model saved to: {output_path}")
    print(f"\nModel version: {version}")
    print("\nInput format:")
    print(f"  - phone: (batch, seq_len, {phone_dim}) - phoneme/hubert embeddings")
    print("  - phone_lengths: (batch,) - sequence lengths")
    print("  - pitch: (batch, seq_len) - F0 pitch in Hz")
    print("  - pitchf: (batch, seq_len) - continuous F0")
    print("  - ds: (batch,) - speaker ID")
    print("  - rnd: (batch, 192, seq_len) - random noise")
    print("\nOutput:")
    print("  - audio: converted audio waveform")
    print("\n⚠ NOTE: This model requires proper preprocessing:")
    print("  1. Extract phoneme/hubert embeddings from audio")
    print("  2. Extract F0 pitch using CREPE/RMVPE")
    print("  3. Format inputs correctly")
    print("\nFor complete inference pipeline, use RVC-WebUI.")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.pth> [output.onnx] [--hidden-channels 256|768]")
        print("\nExamples:")
        print(f"  {sys.argv[0]} model.pth")
        print(f"  {sys.argv[0]} model.pth output.onnx")
        print(f"  {sys.argv[0]} model.pth output.onnx --hidden-channels 768")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: File not found: {model_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_path = Path(sys.argv[2])
    else:
        output_path = model_path.with_suffix('.onnx')

    # Parse hidden_channels
    hidden_channels = 256
    if '--hidden-channels' in sys.argv:
        idx = sys.argv.index('--hidden-channels')
        if idx + 1 < len(sys.argv):
            hidden_channels = int(sys.argv[idx + 1])

    try:
        convert_rvc_to_onnx(str(model_path), str(output_path), hidden_channels)
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
