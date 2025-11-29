import React from 'react';
import { Cpu, Sparkles, ChevronDown, Check, Layers, Square } from 'lucide-react';
import { SingleModelType, ModelSelection, CompareMode } from '../types';

interface ModelSelectorProps {
  selection: ModelSelection;
  onSelectionChange: (selection: ModelSelection) => void;
}

const models: { id: SingleModelType; name: string; description: string; icon: React.ReactNode }[] = [
  {
    id: 'resnet',
    name: 'SRResNet',
    description: 'Fast & reliable upscaling',
    icon: <Cpu className="w-4 h-4" />,
  },
  {
    id: 'gan',
    name: 'SRGAN',
    description: 'Photorealistic details',
    icon: <Sparkles className="w-4 h-4" />,
  },
];

interface ModelDropdownProps {
  label: string;
  value: SingleModelType;
  onChange: (value: SingleModelType) => void;
  variant?: 'default' | 'left' | 'right';
}

const ModelDropdown: React.FC<ModelDropdownProps> = ({ label, value, onChange, variant = 'default' }) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedModel = models.find((m) => m.id === value);

  return (
    <div className="flex flex-col gap-2" ref={dropdownRef}>
      <label className="text-xs font-medium text-silver uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`
            w-full flex items-center justify-between gap-3 p-3 rounded-xl
            bg-obsidian/80 border transition-all duration-200
            ${isOpen ? 'border-accent/50 ring-2 ring-accent/20' : 'border-white/10 hover:border-white/20'}
          `}
        >
          <div className="flex items-center gap-3">
            <div className={`p-1.5 rounded-lg ${variant === 'right' ? 'bg-accent/20' : 'bg-slate/50'} text-accent-bright`}>
              {selectedModel?.icon}
            </div>
            <div className="text-left">
              <div className="text-sm font-medium text-pearl">{selectedModel?.name}</div>
              <div className="text-xs text-silver">{selectedModel?.description}</div>
            </div>
          </div>
          <ChevronDown className={`w-4 h-4 text-silver transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute top-full left-0 right-0 mt-2 z-50 animate-fade-in">
            <div className="bg-obsidian border border-white/10 rounded-xl overflow-hidden shadow-xl shadow-black/50">
              {models.map((model) => (
                <button
                  key={model.id}
                  onClick={() => {
                    onChange(model.id);
                    setIsOpen(false);
                  }}
                  className={`
                    w-full flex items-center justify-between gap-3 p-3 transition-all duration-150
                    ${value === model.id ? 'bg-accent/10' : 'hover:bg-slate/30'}
                  `}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-1.5 rounded-lg ${value === model.id ? 'bg-accent/30 text-accent-bright' : 'bg-slate/50 text-silver'}`}>
                      {model.icon}
                    </div>
                    <div className="text-left">
                      <div className={`text-sm font-medium ${value === model.id ? 'text-pearl' : 'text-silver'}`}>
                        {model.name}
                      </div>
                      <div className="text-xs text-silver/70">{model.description}</div>
                    </div>
                  </div>
                  {value === model.id && (
                    <Check className="w-4 h-4 text-accent-bright" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export const ModelSelector: React.FC<ModelSelectorProps> = ({ selection, onSelectionChange }) => {
  const handleModeChange = (mode: CompareMode) => {
    onSelectionChange({ ...selection, mode });
  };

  return (
    <div className="space-y-4">
      {/* Mode Toggle */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-silver uppercase tracking-wider">
          Mode
        </label>
        <div className="flex gap-2 p-1 bg-obsidian/80 rounded-lg border border-white/5">
          <button
            onClick={() => handleModeChange('single')}
            className={`
              flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-md
              text-sm font-medium transition-all duration-200
              ${selection.mode === 'single'
                ? 'bg-accent text-white shadow-lg shadow-accent/25'
                : 'text-silver hover:text-pearl hover:bg-white/5'
              }
            `}
          >
            <Square className="w-4 h-4" />
            Single Model
          </button>
          <button
            onClick={() => handleModeChange('compare')}
            className={`
              flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-md
              text-sm font-medium transition-all duration-200
              ${selection.mode === 'compare'
                ? 'bg-accent text-white shadow-lg shadow-accent/25'
                : 'text-silver hover:text-pearl hover:bg-white/5'
              }
            `}
          >
            <Layers className="w-4 h-4" />
            Compare Models
          </button>
        </div>
      </div>

      {/* Model Selection */}
      {selection.mode === 'single' ? (
        <ModelDropdown
          label="Select Model"
          value={selection.single}
          onChange={(value) => onSelectionChange({ ...selection, single: value })}
          variant="default"
        />
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <ModelDropdown
            label="Left Side"
            value={selection.left}
            onChange={(value) => onSelectionChange({ ...selection, left: value })}
            variant="left"
          />
          <ModelDropdown
            label="Right Side"
            value={selection.right}
            onChange={(value) => onSelectionChange({ ...selection, right: value })}
            variant="right"
          />
        </div>
      )}

      {/* Helper text */}
      {selection.mode === 'compare' && selection.left === selection.right && (
        <p className="text-xs text-silver/70 text-center">
          Select different models to compare them
        </p>
      )}
    </div>
  );
};
