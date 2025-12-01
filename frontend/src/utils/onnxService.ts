export interface TileInfo {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  current: number;
  total: number;
}

export interface ProcessingCallbacks {
  onTileStart?: (tile: TileInfo) => void;
  onProgress?: (progress: number) => void;
  onComplete?: () => void;
  onTileUpdate?: (tile: TileInfo, tileOutput: ImageData) => void;
}

export class ONNXService {
  private worker: Worker | null = null;
  private initialized = false;
  private initPromise: Promise<void> | null = null;

  async initializeModel(modelPath: string = '/models/srgan.onnx'): Promise<void> {
    console.log('initializeModel called, initialized:', this.initialized, 'worker:', !!this.worker);
    
    if (this.initialized && this.worker) {
      console.log('Model already initialized');
      return;
    }
    
    // Return existing promise if initialization is in progress
    if (this.initPromise) {
      return this.initPromise;
    }
    
    this.initPromise = this.doInitialize(modelPath);
    return this.initPromise;
  }
  
  private async doInitialize(modelPath: string): Promise<void> {
    try {
      // Create worker with separate file
      this.worker = new Worker(new URL('../workers/onnxWorker.ts', import.meta.url), {
        type: 'module'
      });
      
      // Load model
      console.log('Initializing ONNX model from:', modelPath);
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error('Failed to load model');
      }
      const modelBuffer = await response.arrayBuffer();
      
      // Initialize model in worker
      await new Promise<void>((resolve, reject) => {
        if (!this.worker) throw new Error('Worker not created');
        
        this.worker.postMessage({ type: 'loadModel', data: modelBuffer });
        
        const handler = (e: MessageEvent) => {
          const { type, success, error } = e.data;
          if (type === 'modelLoaded') {
            this.worker?.removeEventListener('message', handler);
            if (success) {
              console.log('Worker initialized successfully');
              resolve();
            } else {
              reject(new Error(error));
            }
          }
        };
        
        this.worker.addEventListener('message', handler);
      });
      
      this.initialized = true;
      console.log('ONNX model initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ONNX model:', error);
      this.cleanup();
      throw error;
    }
  }

  private cleanup(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.initialized = false;
    this.initPromise = null;
  }

  async processImage(imageData: ImageData, callbacks?: ProcessingCallbacks): Promise<ImageData> {
    if (!this.worker || !this.initialized) {
      throw new Error('Model not initialized. Call initializeModel() first.');
    }

    // Constants from onnx.html
    const TILE_SIZE = 96;
    const TILE_OVERLAP = 6;
    const SCALE = 4;
    
    return new Promise((resolve, reject) => {
      try {
        const runTiledInference = async (imageData: ImageData): Promise<ImageData> => {
          const { width, height } = imageData;
          const outputWidth = width * SCALE;
          const outputHeight = height * SCALE;
          
          const outputImageData = new ImageData(outputWidth, outputHeight);
          const step = TILE_SIZE - TILE_OVERLAP;
          const tilesY = Math.ceil(height / step);
          const tilesX = Math.ceil(width / step);
          const totalTiles = tilesY * tilesX;
          
          let tileCount = 0;
          
          for (let y = 0; y < height; y += step) {
            for (let x = 0; x < width; x += step) {
              tileCount++;
              const percent = (tileCount / totalTiles) * 100;
              
              callbacks?.onProgress?.(percent);
              callbacks?.onTileStart?.({
                x1: x,
                y1: y,
                x2: Math.min(x + TILE_SIZE, width),
                y2: Math.min(y + TILE_SIZE, height),
                current: tileCount,
                total: totalTiles
              });
              
              const y1 = y;
              const x1 = x;
              const y2 = Math.min(y1 + TILE_SIZE, height);
              const x2 = Math.min(x1 + TILE_SIZE, width);
              
              // Extract patch
              const patch = this.extractPatch(imageData, x1, y1, x2, y2);
              
              // Run inference
              const srPatch = await this.runModelWorker(patch);
              
              // Place patch
              const patchDy1 = y1 * SCALE;
              const patchDx1 = x1 * SCALE;
              this.placePatch(outputImageData, srPatch, patchDx1, patchDy1);
              
              // Create tile-specific ImageData for the processed area
              const tileOutputWidth = (x2 - x1) * SCALE;
              const tileOutputHeight = (y2 - y1) * SCALE;
              const tileOutputData = new Uint8ClampedArray(tileOutputWidth * tileOutputHeight * 4);
              
              // Extract the processed tile area from the full output
              const extractDy1 = y1 * SCALE;
              const extractDx1 = x1 * SCALE;
              const fullOutputData = outputImageData.data;
              const fullOutputWidth = outputImageData.width;
              
              for (let y = 0; y < tileOutputHeight; y++) {
                for (let x = 0; x < tileOutputWidth; x++) {
                  const srcIdx = ((extractDy1 + y) * fullOutputWidth + (extractDx1 + x)) * 4;
                  const dstIdx = (y * tileOutputWidth + x) * 4;
                  
                  tileOutputData[dstIdx] = fullOutputData[srcIdx];
                  tileOutputData[dstIdx + 1] = fullOutputData[srcIdx + 1];
                  tileOutputData[dstIdx + 2] = fullOutputData[srcIdx + 2];
                  tileOutputData[dstIdx + 3] = fullOutputData[srcIdx + 3];
                }
              }
              
              const tileImageData = new ImageData(tileOutputData, tileOutputWidth, tileOutputHeight);
              
              // Send tile-specific update
              console.log('Sending tile update for tile', tileCount);
              callbacks?.onTileUpdate?.({
                x1, y1, x2, y2,
                current: tileCount,
                total: totalTiles
              }, tileImageData);
            }
          }
          
          return outputImageData;
        };
        
        runTiledInference(imageData)
          .then(result => {
            callbacks?.onComplete?.();
            resolve(result);
          })
          .catch(error => {
            reject(error);
          });
          
      } catch (e) {
        reject(e);
      }
    });
  }
  
  private extractPatch(imageData: ImageData, x1: number, y1: number, x2: number, y2: number) {
    const width = x2 - x1;
    const height = y2 - y1;
    const patch = new Float32Array(3 * height * width);
    
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        const srcIdx = (y * imageData.width + x) * 4;
        const localY = y - y1;
        const localX = x - x1;
        const baseIdx = localY * width + localX;
        
        patch[baseIdx] = (imageData.data[srcIdx] / 127.5) - 1.0;
        patch[height * width + baseIdx] = (imageData.data[srcIdx + 1] / 127.5) - 1.0;
        patch[2 * height * width + baseIdx] = (imageData.data[srcIdx + 2] / 127.5) - 1.0;
      }
    }
    
    return { data: patch, width, height };
  }
  
  private runModelWorker(patch: { data: Float32Array, width: number, height: number }) {
    return new Promise<{ data: Uint8ClampedArray, width: number, height: number }>((resolve, reject) => {
      const { data, width, height } = patch;
      
      if (!this.worker) {
        reject(new Error('Worker not available'));
        return;
      }
      
      this.worker.postMessage({
        type: 'runInference',
        data: { patchData: Array.from(data), width, height }
      });
      
      const handler = (e: MessageEvent) => {
        const { type, data, error } = e.data;
        
        if (type === 'inferenceResult') {
          this.worker?.removeEventListener('message', handler);
          
          const { outputData, outWidth, outHeight } = data;
          const rgbData = new Uint8ClampedArray(outHeight * outWidth * 3);
          
          for (let i = 0; i < outputData.length; i++) {
            const val = Math.max(-1, Math.min(1, outputData[i]));
            rgbData[i] = Math.round((val + 1) * 127.5);
          }
          
          resolve({ data: rgbData, width: outWidth, height: outHeight });
        } else if (type === 'inferenceError') {
          this.worker?.removeEventListener('message', handler);
          reject(new Error(error));
        }
      };
      
      this.worker.addEventListener('message', handler);
    });
  }
  
  private placePatch(outputImageData: ImageData, patch: { data: Uint8ClampedArray, width: number, height: number }, dx1: number, dy1: number) {
    const { data, width: patchWidth, height: patchHeight } = patch;
    const outWidth = outputImageData.width;
    const outData = outputImageData.data;
    
    const channelSize = patchWidth * patchHeight;
    
    for (let y = 0; y < patchHeight; y++) {
      for (let x = 0; x < patchWidth; x++) {
        const pixelIdx = y * patchWidth + x;
        
        const rIdx = pixelIdx;
        const gIdx = channelSize + pixelIdx;
        const bIdx = 2 * channelSize + pixelIdx;
        
        const outX = dx1 + x;
        const outY = dy1 + y;
        
        if (outX < outWidth && outY < outputImageData.height) {
          const outIdx = (outY * outWidth + outX) * 4;
          
          outData[outIdx] = data[rIdx];
          outData[outIdx + 1] = data[gIdx];
          outData[outIdx + 2] = data[bIdx];
          outData[outIdx + 3] = 255;
        }
      }
    }
  }
}

export const onnxService = new ONNXService();
