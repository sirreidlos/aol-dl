import React from 'react';
import { Columns, SlidersHorizontal, ZoomIn } from 'lucide-react';
import { ViewMode } from '../types';

interface ViewModeToggleProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

const modes: { id: ViewMode; name: string; icon: React.ReactNode }[] = [
  {
    id: 'side-by-side',
    name: 'Side by Side',
    icon: <Columns className="w-4 h-4" />,
  },
  {
    id: 'slider',
    name: 'Slider',
    icon: <SlidersHorizontal className="w-4 h-4" />,
  },
  {
    id: 'sync-zoom',
    name: 'Sync Zoom',
    icon: <ZoomIn className="w-4 h-4" />,
  },
];

export const ViewModeToggle: React.FC<ViewModeToggleProps> = ({ viewMode, onViewModeChange }) => {
  return (
    <div className="flex flex-col gap-3">
      <label className="text-sm font-medium text-silver uppercase tracking-wider">
        View Mode
      </label>
      <div className="flex gap-2 p-1 bg-obsidian/80 rounded-lg border border-white/5">
        {modes.map((mode) => (
          <button
            key={mode.id}
            onClick={() => onViewModeChange(mode.id)}
            className={`
              flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-md
              text-sm font-medium transition-all duration-200
              ${
                viewMode === mode.id
                  ? 'bg-accent text-white shadow-lg shadow-accent/25'
                  : 'text-silver hover:text-pearl hover:bg-white/5'
              }
            `}
          >
            {mode.icon}
            <span className="hidden sm:inline">{mode.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

