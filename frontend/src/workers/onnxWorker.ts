import { Tensor, InferenceSession } from 'onnxruntime-web';

// Configure ONNX Runtime for module worker
declare const ort: any;
if (typeof ort !== 'undefined') {
  (ort as any).env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  (ort as any).env.wasm.numThreads = 1;
}

let ortSession: InferenceSession | null = null;

// Worker message handler - simplified like onnx.html
self.onmessage = async function(e: MessageEvent) {
  const { type, data } = e.data;
  
  if (type === 'loadModel') {
    try {
      const modelBuffer = data;
      // Create session with WASM backend like onnx.html
      ortSession = await InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      self.postMessage({ type: 'modelLoaded', success: true });
    } catch (err) {
      self.postMessage({ 
        type: 'modelLoaded', 
        success: false, 
        error: (err as Error).message 
      });
    }
  }
  
  if (type === 'runInference') {
    try {
      if (!ortSession) {
        throw new Error('Model not loaded');
      }
      
      const { patchData, width, height } = data;
      
      // Create input tensor exactly like onnx.html
      const inputTensor = new Tensor('float32', patchData, [1, 3, height, width]);
      
      // Run inference
      const feeds = { [ortSession.inputNames[0]]: inputTensor };
      const results = await ortSession.run(feeds);
      
      // Get output
      const outputTensor = results[ortSession.outputNames[0]];
      const outputData = Array.from(outputTensor.data as Float32Array);
      const outHeight = outputTensor.dims[2];
      const outWidth = outputTensor.dims[3];
      
      self.postMessage({
        type: 'inferenceResult',
        data: { outputData, outWidth, outHeight }
      });
    } catch (err) {
      self.postMessage({
        type: 'inferenceError',
        error: (err as Error).message
      });
    }
  }
};
