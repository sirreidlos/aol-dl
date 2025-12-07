import React, { useRef, useEffect, useCallback } from 'react';
import { TileInfo } from '../utils/onnxService';

interface OverlayProgressProps {
  inputImageData: globalThis.ImageData | null;
  currentTile: TileInfo | null;
  progress: number;
  scale: number;
  onTileUpdate?: (updateFn: (tile: TileInfo, tileImageData: globalThis.ImageData) => void) => void;
}

export const OverlayProgress: React.FC<OverlayProgressProps> = ({
  inputImageData,
  currentTile,
  progress,
  scale,
  onTileUpdate
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const outputImageDataRef = useRef<globalThis.ImageData | null>(null);
  
  // Function to draw just the background (input image)
  const redrawBackground = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (!inputImageData) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Create a temporary canvas for the input image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputImageData.width;
    tempCanvas.height = inputImageData.height;
    const tempCtx = tempCanvas.getContext('2d');
    if (tempCtx) {
      tempCtx.putImageData(inputImageData, 0, 0);
      
      // Draw the input image resized to output dimensions
      ctx.drawImage(
        tempCanvas,
        0, 0, inputImageData.width, inputImageData.height,
        0, 0, width, height
      );
    }
  }, [inputImageData]);

  // Function to redraw everything (background + all processed tiles + current overlay)
  const redrawEverything = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!canvasRef.current || !inputImageData) return;

    const canvas = canvasRef.current;
    
    // Draw background
    redrawBackground(ctx, canvas.width, canvas.height);
    
    // Draw all processed tiles
    processedTilesDataRef.current.forEach((tileImageData, tileKey) => {
      const [x1, y1] = tileKey.split('-').map(Number);
      const scaledX1 = x1 * scale;
      const scaledY1 = y1 * scale;
      
      ctx.putImageData(tileImageData, scaledX1, scaledY1);
    });
    
    // Draw current tile overlay if we have one (outline only, no green fill)
    if (currentTile) {
      const { x1, y1, x2, y2 } = currentTile;
      
      // Scale tile coordinates to output dimensions
      const scaledX1 = x1 * scale;
      const scaledY1 = y1 * scale;
      const scaledX2 = x2 * scale;
      const scaledY2 = y2 * scale;
      
      // Draw red border only (no green fill)
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
    }
  }, [inputImageData, scale, currentTile, redrawBackground]);

  // Track processed tiles to avoid redrawing background
  const processedTilesRef = useRef<Set<string>>(new Set());
  const processedTilesDataRef = useRef<Map<string, globalThis.ImageData>>(new Map());

  // Expose the update function to parent
  useEffect(() => {
    if (onTileUpdate) {
      onTileUpdate((tile: TileInfo, tileImageData: globalThis.ImageData) => {
        if (!canvasRef.current) return;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        console.log('Tile overlay update, tile:', tile.current, 'dimensions:', tileImageData.width, 'x', tileImageData.height);
        
        const tileKey = `${tile.x1}-${tile.y1}`;
        const isFirstTile = processedTilesRef.current.size === 0;
        
        // Only redraw background for the first tile
        if (isFirstTile) {
          redrawBackground(ctx, canvas.width, canvas.height);
        }
        
        // Store the tile data if this tile hasn't been processed before
        if (!processedTilesRef.current.has(tileKey)) {
          // Store the tile data
          processedTilesDataRef.current.set(tileKey, tileImageData);
          processedTilesRef.current.add(tileKey);
          
          // Redraw everything
          redrawEverything(ctx);
        }
        
        console.log('Tile overlay update completed');
      });
    }
  }, [onTileUpdate, scale, currentTile, redrawBackground]);

  
  // Update canvas when any relevant data changes
  useEffect(() => {
    if (!canvasRef.current || !inputImageData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas to output dimensions
    const outputWidth = inputImageData.width * scale;
    const outputHeight = inputImageData.height * scale;
    canvas.width = outputWidth;
    canvas.height = outputHeight;

    // Draw the background (input image)
    redrawBackground(ctx, outputWidth, outputHeight);
    
    // Mark that background has been drawn
    outputImageDataRef.current = new ImageData(new Uint8ClampedArray(outputWidth * outputHeight * 4), outputWidth, outputHeight);
  }, [inputImageData, scale, redrawBackground]);

  // Draw current tile overlay when it changes
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Redraw everything (background + processed tiles + current overlay)
    redrawEverything(ctx);
  }, [currentTile, redrawEverything]);
  
  // Don't render if not processing
  if (!inputImageData || progress === 100) {
    return null;
  }

  return (
    <div className="overlay-progress">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-pearl">Processing ({scale}×)</h3>
          {inputImageData && (
            <span className="text-xs text-silver">
              {inputImageData.width * scale}×{inputImageData.height * scale}
            </span>
          )}
        </div>
        
        <div className="relative bg-obsidian/50 rounded-lg overflow-auto flex justify-center items-center p-3 border border-white/10">
          <canvas
            ref={canvasRef}
            className="max-w-full h-auto rounded"
            style={{ maxHeight: '450px' }}
          />
          
          {currentTile && (
            <div className="absolute top-2 left-2 bg-void/90 text-pearl px-2 py-1 rounded text-xs border border-white/10">
              <div>Tile {currentTile.current}/{currentTile.total}</div>
            </div>
          )}
          
          {progress > 0 && progress < 100 && (
            <div className="absolute top-2 right-2 bg-void/90 text-pearl px-2 py-1 rounded text-xs border border-white/10">
              <div>{Math.round(progress)}%</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
