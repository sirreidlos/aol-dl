import React, { useEffect, useRef } from 'react';

interface TileOverlayProps {
  imageSrc: string;
  tile: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    current: number;
    total: number;
  } | null;
}

export const TileOverlay: React.FC<TileOverlayProps> = ({ imageSrc, tile }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!imageRef.current || !canvasRef.current || !tile) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;
    
    // Set canvas size to match image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the tile highlight
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Scale tile coordinates to canvas size
    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;
    
    const x1 = tile.x1 * scaleX;
    const y1 = tile.y1 * scaleY;
    const x2 = tile.x2 * scaleX;
    const y2 = tile.y2 * scaleY;

    // Draw rectangle around current tile
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Add semi-transparent fill
    ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

  }, [tile]);

  if (!tile) return null;

  return (
    <div className="absolute inset-0 pointer-events-none">
      <img
        ref={imageRef}
        src={imageSrc}
        alt="Original"
        className="w-full h-auto object-contain opacity-0"
        onLoad={() => {
          // Trigger canvas redraw when image loads
          if (canvasRef.current && tile) {
            const event = new Event('resize');
            window.dispatchEvent(event);
          }
        }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ mixBlendMode: 'normal' }}
      />
      <div className="absolute top-3 right-3 z-20">
        <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-green-500/20 backdrop-blur-sm border border-green-500/30 text-green-400">
          Processing tile {tile.current} of {tile.total}
        </span>
      </div>
    </div>
  );
};
