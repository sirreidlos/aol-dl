# ONNX Runtime Integration Setup

## Overview
The frontend has been integrated with ONNX runtime to perform super-resolution inference directly in the browser.

## Key Features
1. **Tile-based Processing**: Large images are processed in tiles to handle memory constraints
2. **Progress Tracking**: Shows real-time progress and current tile being processed
3. **Visual Feedback**: Green overlay shows the current tile being processed on the original image
4. **Error Handling**: Graceful fallback if ONNX model fails to load or process

## Setup Instructions

### 1. Install Dependencies
```bash
cd frontend
npm install onnxruntime-web
```

### 2. Verify ONNX Model
The ONNX model should be located at `public/srragan_psnr_nobn_2.onnx`. If it's missing, copy it from the project root:
```bash
cp ../srragan_psnr_nobn_2.onnx public/
```

### 3. Start Development Server
```bash
npm run dev
```

## How It Works

1. **Model Initialization**: When the app loads, it initializes the ONNX model using WebGL for acceleration
2. **Image Upload**: When you upload an image, it's converted to the format expected by the model
3. **Tile Processing**: For images larger than 512x512, the image is split into overlapping tiles
4. **Inference**: Each tile is processed through the ONNX model
5. **Reconstruction**: Processed tiles are stitched back together to form the final upscaled image
6. **Display**: The result is shown alongside the original for comparison

## UI Features During Processing

- **Progress Bar**: Shows overall processing progress (0-100%)
- **Tile Overlay**: Green rectangle shows the current tile being processed
- **Tile Counter**: Shows "Processing tile X of Y"
- **Status Indicator**: Yellow dot during processing, green when complete

## Model Selection

Currently, the app uses the SRGAN model (`srragan_psnr_nobn_2.onnx`) for both "SRResNet" and "SRGAN" options. The model names in the UI have been updated to reflect this.

## Troubleshooting

- **Model Loading Issues**: Check browser console for errors. Ensure the ONNX model file is accessible
- **Performance**: For very large images, processing may take time. The tile-based approach helps manage memory
- **Browser Compatibility**: Requires WebGL support for optimal performance

## Technical Details

- **Input Size**: 512x512 pixels (images are automatically resized)
- **Tile Overlap**: 64 pixels to avoid seam artifacts
- **Output**: 4x upscaling (512x512 → 2048x2048)
- **Backend**: WebGL acceleration via onnxruntime-web
