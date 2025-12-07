import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Move } from 'lucide-react';
import { ZoomState, DualComparisonProps } from '../types';
import { TileOverlay } from './TileOverlay';

interface BothModelsViewProps extends DualComparisonProps {
  processingTile?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    current: number;
    total: number;
  } | null;
}

export const BothModelsView: React.FC<BothModelsViewProps> = ({
  originalImage,
  leftImage,
  rightImage,
  leftModelName,
  rightModelName,
  processingTile,
}) => {
  const [zoom, setZoom] = useState<ZoomState>({ scale: 1, x: 50, y: 50 });
  const [isHovering, setIsHovering] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setZoom((prev) => ({ ...prev, x, y }));
  }, []);

  const handleZoomIn = useCallback(() => {
    setZoom((prev) => ({ ...prev, scale: Math.min(prev.scale + 0.5, 4) }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom((prev) => ({ ...prev, scale: Math.max(prev.scale - 0.5, 1) }));
  }, []);

  // Use native event listener with passive: false to properly prevent page scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY > 0 ? -0.2 : 0.2;
      setZoom((prev) => ({
        ...prev,
        scale: Math.max(1, Math.min(4, prev.scale + delta)),
      }));
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, []);

  const getTransformStyle = () => ({
    transform: `scale(${zoom.scale})`,
    transformOrigin: `${zoom.x}% ${zoom.y}%`,
    transition: isHovering ? 'transform-origin 0.1s ease-out' : 'transform 0.3s ease-out, transform-origin 0.1s ease-out',
  });

  const ImageCard: React.FC<{ image: string; label: string; isOriginal?: boolean; size?: 'small' | 'normal' }> = ({
    image,
    label,
    isOriginal = false,
    size = 'normal',
  }) => (
    <div className="relative group">
      <div
        className={`
          absolute -inset-0.5 rounded-xl opacity-50 transition-opacity duration-300
          ${isOriginal ? 'bg-gradient-to-r from-slate to-obsidian' : 'bg-gradient-to-r from-accent/30 to-accent-bright/30 group-hover:opacity-75'}
        `}
      />
      <div
        className={`
          relative bg-obsidian rounded-xl overflow-hidden border
          ${isOriginal ? 'border-white/5' : 'border-accent/20'}
        `}
      >
        <div className="absolute top-2 left-2 z-10">
          <span className="px-2 py-1 rounded-full text-xs font-semibold bg-white/90 backdrop-blur-sm border border-black/10 text-black shadow-md">
            {label}
          </span>
        </div>
        <div className="overflow-hidden">
          <img
            src={image}
            alt={label}
            className={`w-full h-auto object-contain ${size === 'small' ? 'max-h-40' : ''}`}
            style={getTransformStyle()}
            draggable={false}
          />
          {isOriginal && processingTile && <TileOverlay imageSrc={image} tile={processingTile} />}
        </div>
      </div>
    </div>
  );

  return (
    <div className="animate-fade-in">
      {/* Zoom Controls */}
      <div className="flex items-center justify-center gap-4 mb-4">
        <button
          onClick={handleZoomOut}
          disabled={zoom.scale <= 1}
          className="p-2 rounded-lg bg-obsidian border border-white/10 text-silver hover:text-pearl hover:border-white/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <ZoomOut className="w-5 h-5" />
        </button>
        
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-obsidian/50 border border-white/5">
          <Move className="w-4 h-4 text-silver" />
          <span className="text-sm font-medium text-pearl">{Math.round(zoom.scale * 100)}%</span>
        </div>
        
        <button
          onClick={handleZoomIn}
          disabled={zoom.scale >= 4}
          className="p-2 rounded-lg bg-obsidian border border-white/10 text-silver hover:text-pearl hover:border-white/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <ZoomIn className="w-5 h-5" />
        </button>
      </div>

      <p className="text-center text-xs text-silver mb-4">
        Hover to pan • Scroll to zoom • All images synchronized
      </p>

      {/* Three-way Comparison */}
      <div
        ref={containerRef}
        className="space-y-4"
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        {/* Original - smaller and centered */}
        <div className="flex justify-center">
          <div className="w-1/2">
            <ImageCard image={originalImage} label="Original" isOriginal size="small" />
          </div>
        </div>
        
        {/* Models side by side below */}
        <div className="grid grid-cols-2 gap-4">
          <ImageCard image={leftImage} label={leftModelName} />
          <ImageCard image={rightImage} label={rightModelName} />
        </div>
      </div>

      {/* Position Indicator */}
      <div className="mt-4 flex justify-center">
        <div className="relative w-32 h-20 rounded-lg bg-obsidian/50 border border-white/10 overflow-hidden">
          <div
            className="absolute w-6 h-4 rounded border-2 border-accent bg-accent/20 transition-all duration-100"
            style={{
              left: `${(zoom.x / 100) * (128 - 24)}px`,
              top: `${(zoom.y / 100) * (80 - 16)}px`,
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-xs text-silver/50">
            viewport
          </div>
        </div>
      </div>
    </div>
  );
};
