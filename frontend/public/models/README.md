# ONNX Models

This directory contains the ONNX models for super resolution.

## Current Models

- `srgan.onnx` - SRGAN model (copy of srragan_psnr_nobn_2.onnx)
- `srresnet.onnx` - SRResNet model (copy of srragan_psnr_nobn_2.onnx)

## Setup

To set up the models, run:

```bash
# From the frontend directory
cp public/srragan_psnr_nobn_2.onnx public/models/srgan.onnx
cp public/srragan_psnr_nobn_2.onnx public/models/srresnet.onnx
```

Or use the setup script:

```bash
./scripts/setup-models.sh
```

## Model Details

- Input size: Flexible (fully convolutional model)
- Tile size: 96x96 (for very fast CPU processing)
- Overlap: 6px
- Output: 4x super resolution
- Format: ONNX
- Framework: PyTorch SR models

## Performance Notes

- Model is fully convolutional - accepts flexible input sizes
- 96x96 tiles with 6px overlap for extremely fast processing
- WebGL acceleration is attempted but falls back to CPU if not available
- Processing time varies based on image size and hardware capabilities
- Expect very fast processing with small tiles
