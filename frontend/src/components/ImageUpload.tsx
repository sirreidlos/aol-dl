import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploadProps {
  onImageUpload: (file: File, dataUrl: string) => void;
  currentImage: string | null;
  onClear: () => void;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUpload, currentImage, onClear }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const dataUrl = e.target?.result as string;
          onImageUpload(file, dataUrl);
        };
        reader.readAsDataURL(file);
      }
    },
    [onImageUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  if (currentImage) {
    return (
      <div className="relative group">
        <div className="relative overflow-hidden rounded-xl border border-white/10">
          <img
            src={currentImage}
            alt="Uploaded"
            className="w-full h-48 object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-void/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </div>
        <button
          onClick={onClear}
          className="absolute top-3 right-3 p-2 rounded-lg bg-void/80 border border-white/10 text-silver hover:text-red-400 hover:border-red-400/30 transition-all duration-200 opacity-0 group-hover:opacity-100"
        >
          <X className="w-4 h-4" />
        </button>
        <div className="absolute bottom-3 left-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <p className="text-xs text-silver truncate">Image loaded successfully</p>
        </div>
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        relative overflow-hidden rounded-xl border-2 border-dashed transition-all duration-300
        ${
          isDragging
            ? 'border-accent bg-accent/10 scale-[1.02]'
            : 'border-white/10 hover:border-white/20 bg-obsidian/30'
        }
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleInputChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
      />
      
      <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
        <div
          className={`
            p-4 rounded-2xl mb-4 transition-all duration-300
            ${isDragging ? 'bg-accent/20 scale-110' : 'bg-slate/30'}
          `}
        >
          {isDragging ? (
            <ImageIcon className="w-8 h-8 text-accent-bright" />
          ) : (
            <Upload className="w-8 h-8 text-silver" />
          )}
        </div>
        
        <h3 className="font-display font-semibold text-pearl mb-1">
          {isDragging ? 'Drop your image' : 'Upload an image'}
        </h3>
        <p className="text-sm text-silver">
          Drag & drop or click to browse
        </p>
        <p className="text-xs text-silver/50 mt-2">
          Supports JPG, PNG, WebP
        </p>
      </div>

      {/* Animated border gradient on drag */}
      {isDragging && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-accent/20 via-accent-bright/20 to-accent/20 animate-shimmer" />
        </div>
      )}
    </div>
  );
};

