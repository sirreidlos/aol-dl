# SR Model Comparison Frontend

A React + TypeScript frontend for comparing Super Resolution model outputs (SRResNet vs SRGAN).

## Features

- **Model Selection**: Choose between SRResNet, SRGAN, or compare both models
- **Multiple View Modes**:
  - **Side by Side**: View original and processed images next to each other
  - **Slider**: Interactive slider to reveal/hide the processed result
  - **Sync Zoom**: Synchronized zooming and panning across both images
- **Drag & Drop Upload**: Easy image upload with drag and drop support
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
npm run preview
```

## Backend Integration

The frontend expects a Python FastAPI backend with the following endpoints:

- `POST /api/upscale/resnet` - Upscale image using SRResNet
- `POST /api/upscale/gan` - Upscale image using SRGAN  
- `POST /api/upscale/both` - Upscale using both models

Each endpoint should accept a multipart form with an `image` field and return:

```json
{
  "success": true,
  "originalUrl": "...",
  "processedUrl": "...",
  "model": "resnet|gan",
  "processingTime": 1.234
}
```

## Project Structure

```
frontend/
├── public/
│   └── vite.svg
├── src/
│   ├── api/
│   │   └── index.ts          # API utilities
│   ├── components/
│   │   ├── BothModelsView.tsx    # View for comparing both models
│   │   ├── ImageUpload.tsx       # Image upload component
│   │   ├── ModelSelector.tsx     # Model selection buttons
│   │   ├── SideBySideView.tsx    # Side-by-side comparison
│   │   ├── SliderView.tsx        # Slider comparison
│   │   ├── SyncZoomView.tsx      # Synchronized zoom view
│   │   └── ViewModeToggle.tsx    # View mode selector
│   ├── types/
│   │   └── index.ts          # TypeScript type definitions
│   ├── App.tsx               # Main application component
│   ├── index.css             # Global styles
│   └── main.tsx              # Application entry point
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Contributors

Group Project

