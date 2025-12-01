import React from 'react';
import { ComparisonProps } from '../types';
import { TileOverlay } from './TileOverlay';

interface SideBySideViewProps extends ComparisonProps {
  processingTile?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    current: number;
    total: number;
  } | null;
}

export const SideBySideView: React.FC<SideBySideViewProps> = ({
  originalImage,
  processedImage,
  modelName,
  processingTile,
}) => {
  return (
    <div className="grid grid-cols-2 gap-4 animate-fade-in">
      {/* Original Image */}
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-slate to-obsidian rounded-xl opacity-50" />
        <div className="relative bg-obsidian rounded-xl overflow-hidden border border-white/5">
          <div className="absolute top-3 left-3 z-10">
            <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-void/80 backdrop-blur-sm border border-white/10 text-silver">
              Original
            </span>
          </div>
          <img
            src={originalImage}
            alt="Original"
            className="w-full h-auto object-contain"
          />
          {processingTile && <TileOverlay imageSrc={originalImage} tile={processingTile} />}
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
          <img
            src={processedImage}
            alt={`${modelName} Result`}
            className="w-full h-auto object-contain"
          />
        </div>
      </div>
    </div>
  );
};

