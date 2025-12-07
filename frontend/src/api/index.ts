/**
 * API utilities for connecting to the Python backend
 * 
 * In production, set up your FastAPI backend to expose endpoints like:
 * - GET /api/health - Check if models are loaded and ready
 * - POST /api/upscale/resnet - Upscale image using SRResNet
 * - POST /api/upscale/gan - Upscale image using SRGAN
 * - POST /api/upscale/both - Upscale using both models
 */

const API_BASE = '/api';

export interface ModelStatus {
  ready: boolean;
  models: {
    resnet: boolean;
    gan: boolean;
  };
  message?: string;
}

/**
 * Check if the backend and models are ready
 */
export async function checkModelStatus(): Promise<ModelStatus> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
    });

    if (!response.ok) {
      return {
        ready: false,
        models: { resnet: false, gan: false },
        message: 'Backend not responding',
      };
    }

    return response.json();
  } catch {
    return {
      ready: false,
      models: { resnet: false, gan: false },
      message: 'Cannot connect to backend',
    };
  }
}

interface UpscaleResponse {
  success: boolean;
  originalUrl: string;
  processedUrl: string;
  model: string;
  processingTime: number;
}

interface UpscaleBothResponse {
  success: boolean;
  originalUrl: string;
  resnetUrl: string;
  ganUrl: string;
  processingTime: number;
}

/**
 * Upload and upscale an image using the specified model
 */
export async function upscaleImage(
  file: File,
  model: 'resnet' | 'gan'
): Promise<UpscaleResponse> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_BASE}/upscale/${model}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to upscale image: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Upload and upscale an image using both models
 */
export async function upscaleImageBoth(file: File): Promise<UpscaleBothResponse> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_BASE}/upscale/both`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to upscale image: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Convert a File to base64 data URL
 */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

