import React, { useState, useCallback, useRef } from 'react';
import { ZoomIn, ZoomOut, Move } from 'lucide-react';
import { ComparisonProps, ZoomState } from '../types';

interface SyncZoomViewProps extends ComparisonProps {}

export const SyncZoomView: React.FC<SyncZoomViewProps> = ({
  originalImage,
  processedImage,
  modelName,
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

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.2 : 0.2;
    setZoom((prev) => ({
      ...prev,
      scale: Math.max(1, Math.min(4, prev.scale + delta)),
    }));
  }, []);

  const getTransformStyle = () => ({
    transform: `scale(${zoom.scale})`,
    transformOrigin: `${zoom.x}% ${zoom.y}%`,
    transition: isHovering ? 'transform-origin 0.1s ease-out' : 'transform 0.3s ease-out, transform-origin 0.1s ease-out',
  });

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
        Hover over images to pan • Scroll to zoom • Both images stay synchronized
      </p>

      {/* Comparison Grid */}
      <div
        ref={containerRef}
        className="grid grid-cols-2 gap-4"
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        onWheel={handleWheel}
      >
        {/* Original Image */}
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-slate to-obsidian rounded-xl opacity-50" />
          <div className="relative bg-obsidian rounded-xl overflow-hidden border border-white/5">
            <div className="absolute top-3 left-3 z-10">
              <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-void/80 backdrop-blur-sm border border-white/10 text-silver">
                Original
              </span>
            </div>
            <div className="overflow-hidden">
              <img
                src={originalImage}
                alt="Original"
                className="w-full h-auto object-contain"
                style={getTransformStyle()}
                draggable={false}
              />
            </div>
          </div>
        </div>

        {/* Processed Image */}
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-accent/30 to-accent-bright/30 rounded-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300" />
          <div className="relative bg-obsidian rounded-xl overflow-hidden border border-accent/20">
            <div className="absolute top-3 left-3 z-10">
              <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-accent/20 backdrop-blur-sm border border-accent/30 text-accent-bright">
                {modelName}
              </span>
            </div>
            <div className="overflow-hidden">
              <img
                src={processedImage}
                alt={`${modelName} Result`}
                className="w-full h-auto object-contain"
                style={getTransformStyle()}
                draggable={false}
              />
            </div>
          </div>
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

