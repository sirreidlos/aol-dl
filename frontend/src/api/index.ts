/**
 * API utilities for connecting to the Python backend
 * 
 * In production, set up your FastAPI backend to expose endpoints like:
 * - POST /api/upscale/resnet - Upscale image using SRResNet
 * - POST /api/upscale/gan - Upscale image using SRGAN
 * - POST /api/upscale/both - Upscale using both models
 */

const API_BASE = '/api';

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

