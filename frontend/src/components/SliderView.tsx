import React, { useState, useRef, useCallback, useEffect } from 'react';
import { ComparisonProps } from '../types';

interface SliderViewProps extends ComparisonProps {}

export const SliderView: React.FC<SliderViewProps> = ({
  originalImage,
  processedImage,
  modelName,
}) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPosition(percentage);
    },
    []
  );

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    },
    [isDragging, handleMove]
  );

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (isDragging && e.touches[0]) {
        handleMove(e.touches[0].clientX);
      }
    },
    [isDragging, handleMove]
  );

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('touchmove', handleTouchMove);
    document.addEventListener('touchend', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp, handleTouchMove]);

  return (
    <div className="animate-fade-in">
      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-xl border border-white/10 cursor-col-resize select-none"
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
      >
        {/* Original Image (Full) */}
        <img
          src={originalImage}
          alt="Original"
          className="w-full h-auto block"
          draggable={false}
        />

        {/* Processed Image (Clipped) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
        >
          <img
            src={processedImage}
            alt={`${modelName} Result`}
            className="w-full h-auto block"
            draggable={false}
          />
        </div>

        {/* Slider Line */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white/90 shadow-lg shadow-black/50"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        >
          {/* Slider Handle */}
          <div
            className={`
              absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
              w-10 h-10 rounded-full bg-white shadow-xl
              flex items-center justify-center
              transition-transform duration-150
              ${isDragging ? 'scale-110' : 'scale-100'}
            `}
          >
            <div className="flex gap-0.5">
              <div className="w-0.5 h-4 bg-slate/50 rounded-full" />
              <div className="w-0.5 h-4 bg-slate/50 rounded-full" />
            </div>
          </div>
        </div>

        {/* Labels */}
        <div className="absolute top-3 left-3 z-10">
          <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-void/80 backdrop-blur-sm border border-white/10 text-silver">
            Original
          </span>
        </div>
        <div className="absolute top-3 right-3 z-10">
          <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-accent/20 backdrop-blur-sm border border-accent/30 text-accent-bright">
            {modelName}
          </span>
        </div>
      </div>

      {/* Slider control below */}
      <div className="mt-4 px-4">
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPosition}
          onChange={(e) => setSliderPosition(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-silver mt-2">
          <span>Original</span>
          <span>{Math.round(sliderPosition)}%</span>
          <span>{modelName}</span>
        </div>
      </div>
    </div>
  );
};

