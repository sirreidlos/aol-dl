export type SingleModelType = 'resnet' | 'gan';

export type CompareMode = 'single' | 'compare';

export type ViewMode = 'side-by-side' | 'slider' | 'sync-zoom';

export interface ModelSelection {
  mode: CompareMode;
  single: SingleModelType;
  left: SingleModelType;
  right: SingleModelType;
}

export interface ImageData {
  original: string | null;
  resnet: string | null;
  gan: string | null;
}

export interface ZoomState {
  scale: number;
  x: number;
  y: number;
}

export interface ComparisonProps {
  originalImage: string;
  processedImage: string;
  modelName: string;
}

export interface DualComparisonProps {
  originalImage: string;
  leftImage: string;
  rightImage: string;
  leftModelName: string;
  rightModelName: string;
}
