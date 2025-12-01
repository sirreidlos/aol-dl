import React, { useRef, useEffect, useCallback } from 'react';
import { TileInfo } from '../utils/onnxService';

interface InferenceProgressProps {
  inputImageData: globalThis.ImageData | null;
  outputImageData: globalThis.ImageData | null;
  currentTile: TileInfo | null;
  progress: number;
  scale: number;
  onOutputUpdate?: (updateFn: (imageData: globalThis.ImageData) => void) => void;
}

export const InferenceProgress: React.FC<InferenceProgressProps> = ({
  inputImageData,
  outputImageData,
  currentTile,
  progress,
  scale,
  onOutputUpdate
}) => {
  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputImageDataRef = useRef<globalThis.ImageData | null>(null);
  
  // Expose the update function to parent
  useEffect(() => {
    if (onOutputUpdate) {
      onOutputUpdate((imageData: globalThis.ImageData) => {
        if (!outputCanvasRef.current) return;
        
        const canvas = outputCanvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        console.log('Direct callback update, dimensions:', imageData.width, 'x', imageData.height);
        
        // Set canvas size
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        
        // Draw the output image
        ctx.putImageData(imageData, 0, 0);
        
        console.log('Direct callback update completed');
      });
    }
  }, [onOutputUpdate]);

  // Update input canvas with tile highlighting
  useEffect(() => {
    if (!inputCanvasRef.current || !inputImageData) return;

    const canvas = inputCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = inputImageData.width;
    canvas.height = inputImageData.height;

    // Draw the input image
    ctx.putImageData(inputImageData, 0, 0);

    // Highlight current tile if processing
    if (currentTile) {
      const { x1, y1, x2, y2 } = currentTile;
      
      // Draw semi-transparent green overlay
      ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

      // Draw red border
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }, [inputImageData, currentTile]);

  // Update output canvas with real-time patches
  useEffect(() => {
    console.log('Output canvas useEffect triggered, outputImageData:', outputImageData ? 'present' : 'null');
    if (!outputCanvasRef.current || !outputImageData) return;

    const canvas = outputCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    console.log('Drawing to output canvas, dimensions:', outputImageData.width, 'x', outputImageData.height);
    
    // Set canvas size
    canvas.width = outputImageData.width;
    canvas.height = outputImageData.height;

    // Draw the output image
    ctx.putImageData(outputImageData, 0, 0);
    
    console.log('Output canvas updated');
  }, [outputImageData]);

  // Update ref when outputImageData changes
  useEffect(() => {
    outputImageDataRef.current = outputImageData;
  }, [outputImageData]);

  // Direct update function to bypass React batching
  const updateOutputCanvas = useCallback(() => {
    const currentImageData = outputImageDataRef.current;
    if (!outputCanvasRef.current || !currentImageData) return;

    const canvas = outputCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    console.log('Direct canvas update, dimensions:', currentImageData.width, 'x', currentImageData.height);
    
    // Set canvas size
    canvas.width = currentImageData.width;
    canvas.height = currentImageData.height;

    // Draw the output image
    ctx.putImageData(currentImageData, 0, 0);
    
    console.log('Direct canvas update completed');
  }, []);

  // Call direct update when outputImageData prop changes
  useEffect(() => {
    if (outputImageData) {
      // Use setTimeout to ensure this runs after React's render cycle
      setTimeout(updateOutputCanvas, 0);
    }
  }, [outputImageData, updateOutputCanvas]);

  // Don't render if not processing
  if (!inputImageData || progress === 100) {
    return null;
  }

  return (
    <div className="inference-progress">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Image */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Input Image</h3>
            {inputImageData && (
              <span className="text-sm text-gray-600">
                {inputImageData.width}×{inputImageData.height}
              </span>
            )}
          </div>
          
          <div className="relative bg-gray-100 rounded-lg overflow-hidden">
            {inputImageData ? (
              <canvas
                ref={inputCanvasRef}
                className="w-full h-auto"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
              />
            ) : (
              <div className="aspect-square flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p>No input image</p>
                </div>
              </div>
            )}
            
            {currentTile && (
              <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded-md text-sm">
                <div>Tile {currentTile.current}/{currentTile.total}</div>
                <div className="text-xs opacity-75">
                  ({currentTile.x1}, {currentTile.y1}) → ({currentTile.x2}, {currentTile.y2})
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Output Image */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Output Image ({scale}× Super Resolution)</h3>
            {outputImageData && (
              <span className="text-sm text-gray-600">
                {outputImageData.width}×{outputImageData.height}
              </span>
            )}
          </div>
          
          <div className="relative bg-gray-100 rounded-lg overflow-hidden">
            {outputImageData ? (
              <canvas
                ref={outputCanvasRef}
                className="w-full h-auto"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
              />
            ) : (
              <div className="aspect-square flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <p>Output will appear here</p>
                </div>
              </div>
            )}
            
            {progress > 0 && progress < 100 && (
              <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded-md text-sm">
                <div>Processing...</div>
                <div className="text-xs opacity-75">{Math.round(progress)}%</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
