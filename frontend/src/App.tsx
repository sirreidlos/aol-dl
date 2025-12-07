import { useState, useCallback, useRef, useEffect } from 'react';
import { Wand2, Github, Sparkles } from 'lucide-react';
import { ONNXService } from './utils/onnxService';
const onnxService = new ONNXService();
import { fileToImageData, imageDataToDataUrl } from './utils/imageUtils';
import {
  ModelSelector,
  ViewModeToggle,
  ImageUpload,
  SideBySideView,
  SliderView,
  SyncZoomView,
  BothModelsView,
  OverlayProgress,
  LoadingScreen,
} from './components';
import { ModelSelection, ViewMode, ImageData, SingleModelType } from './types';

// Demo images for testing - replace with actual API integration
const DEMO_IMAGES = {
  original: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&q=80',
  srgan: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
  srresnet: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
  srragan: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
  srranet: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
};

function App() {
  const [modelSelection, setModelSelection] = useState<ModelSelection>({
    mode: 'single',
    single: 'srresnet',
    left: 'srresnet',
    right: 'srgan',
  });
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [images, setImages] = useState<ImageData>({
    original: null,
    srgan: null,
    srresnet: null,
    srragan: null,
    srranet: null,
  });
  const [isProcessing, setIsProcessing] = useState(false);

  // Check if we're in compare mode with different models
  const isComparing = modelSelection.mode === 'compare' && modelSelection.left !== modelSelection.right;

  // Track processing progress (0-100%)
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentTile, setCurrentTile] = useState<{x1: number, y1: number, x2: number, y2: number, current: number, total: number} | null>(null);
  const [modelInitialized, setModelInitialized] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [modelsStatus, setModelsStatus] = useState({ resnet: false, gan: false });
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const processingRef = useRef(false);
  const initializingRef = useRef(false);
  
  // Track ImageData for progress visualization
  const [inputImageData, setInputImageData] = useState<globalThis.ImageData | null>(null);
  const outputImageDataRef = useRef<globalThis.ImageData | null>(null);
  const overlayOutputUpdateRef = useRef<(tile: any, tileImageData: globalThis.ImageData) => void>(() => {});

  // Initialize ONNX model when component mounts or model selection changes
  useEffect(() => {
    // Prevent multiple initializations
    if (initializingRef.current) {
      console.log('Already initializing model, skipping...');
      return;
    }

    console.log('Model selection changed, initializing model');
    const initModel = async () => {
      initializingRef.current = true;
      setModelLoadError(null);
      try {
        const selectedModel = modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left;
        const modelPath = getModelPath(selectedModel);
        console.log('Calling onnxService.initializeModel with:', modelPath);
        await onnxService.initializeModel(modelPath);
        console.log('ONNX model initialized successfully');
        console.log('Setting modelInitialized to true...');
        
        // Update model status based on which model was loaded
        setModelsStatus(prev => ({
          ...prev,
          resnet: selectedModel === 'srresnet' || selectedModel === 'srranet' ? true : prev.resnet,
          gan: selectedModel === 'srgan' || selectedModel === 'srragan' ? true : prev.gan,
        }));
        
        setModelInitialized(true);
        console.log('modelInitialized state set');
      } catch (error) {
        console.error('Failed to initialize ONNX model:', error);
        setModelLoadError(error instanceof Error ? error.message : 'Failed to load model');
      } finally {
        initializingRef.current = false;
      }
    };
    initModel();
  }, [modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left]);

  // Retry loading model
  const handleRetryModelLoad = useCallback(() => {
    setModelInitialized(false);
    setModelLoadError(null);
    setModelsStatus({ resnet: false, gan: false });
    initializingRef.current = false;
    // Trigger re-initialization by forcing a re-render
    const selectedModel = modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left;
    const initModel = async () => {
      initializingRef.current = true;
      try {
        const modelPath = getModelPath(selectedModel);
        await onnxService.initializeModel(modelPath);
        setModelsStatus(prev => ({
          ...prev,
          resnet: selectedModel === 'srresnet' || selectedModel === 'srranet' ? true : prev.resnet,
          gan: selectedModel === 'srgan' || selectedModel === 'srragan' ? true : prev.gan,
        }));
        setModelInitialized(true);
      } catch (error) {
        console.error('Failed to initialize ONNX model:', error);
        setModelLoadError(error instanceof Error ? error.message : 'Failed to load model');
      } finally {
        initializingRef.current = false;
      }
    };
    initModel();
  }, [modelSelection]);

  // Handle re-processing when model selection changes (if we already have an image)
  useEffect(() => {
    // Skip if no file, not initialized, or already processing
    if (!currentFile || !modelInitialized || processingRef.current) return;
    
    const reprocessIfNeeded = async () => {
      if (modelSelection.mode === 'single') {
        // In single mode, re-process if the selected model hasn't been processed yet
        if (!images[modelSelection.single]) {
          console.log('Re-processing for single model:', modelSelection.single);
          
          // Load the model first
          const modelPath = getModelPath(modelSelection.single);
          await onnxService.initializeModel(modelPath);
          
          await processImageWithModel(currentFile, modelSelection.single, true);
        }
      } else {
        // Compare mode - check if we need to process either model
        const needsLeft = !images[modelSelection.left];
        const needsRight = !images[modelSelection.right];
        
        if (needsLeft || needsRight) {
          console.log('Re-processing for compare mode, needsLeft:', needsLeft, 'needsRight:', needsRight);
          
          // Process left model if needed
          if (needsLeft) {
            console.log('Loading and processing left model:', modelSelection.left);
            const leftModelPath = getModelPath(modelSelection.left);
            await onnxService.initializeModel(leftModelPath);
            await processImageWithModel(currentFile, modelSelection.left, true);
            processingRef.current = false; // Reset for next model
          }
          
          // Process right model if needed (and different from left)
          if (needsRight && modelSelection.right !== modelSelection.left) {
            console.log('Loading and processing right model:', modelSelection.right);
            const rightModelPath = getModelPath(modelSelection.right);
            await onnxService.initializeModel(rightModelPath);
            await processImageWithModel(currentFile, modelSelection.right, true);
          }
        }
      }
    };
    
    reprocessIfNeeded();
  }, [modelSelection.mode, modelSelection.single, modelSelection.left, modelSelection.right, modelInitialized, currentFile]); // Trigger on any model change

  const getModelPath = (modelType: SingleModelType): string => {
    return `models/${modelType}.onnx`;
  };

  const processImageWithModel = useCallback(async (file: File, specificModel?: SingleModelType, skipOriginal?: boolean) => {
    if (processingRef.current) return;
    
    console.log('processImageWithModel called with modelInitialized:', modelInitialized, 'skipOriginal:', skipOriginal);
    
    console.log('Model is initialized, proceeding with image processing...');
    
    // Declare originalDataUrl outside try-catch to make it accessible
    let originalDataUrl: string | null = null;
    
    try {
      processingRef.current = true;
      setIsProcessing(true);
      setProcessingProgress(0);
      
      // Convert file to ImageData
      const imageData = await fileToImageData(file);
      
      // Only set original if not skipping (for compare mode second model)
      if (!skipOriginal) {
        originalDataUrl = URL.createObjectURL(file);
        setImages(prev => ({ ...prev, original: originalDataUrl }));
        setCurrentFile(file); // Store current file for re-processing
      }
      
      // Set input ImageData for progress visualization
      setInputImageData(imageData);
      outputImageDataRef.current = null; // Clear ref as well
      
      setProcessingProgress(30);
      
      // Determine which model to use for processing
      const targetModel = specificModel || (modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left);
      
      // Process with ONNX model
      const processedImageData = await onnxService.processImage(imageData, {
        onTileStart: (tile: any) => {
          setCurrentTile(tile);
        },
        onProgress: (progress: any) => {
          setProcessingProgress(progress);
        },
        onComplete: () => {
          setCurrentTile(null);
        },
        onTileUpdate: (tile: any, tileImageData: globalThis.ImageData) => {
          console.log('Received tile update for tile', tile.current, 'dimensions:', tileImageData.width, 'x', tileImageData.height);
          
          // Call overlay update with tile-specific data
          if (overlayOutputUpdateRef.current) {
            overlayOutputUpdateRef.current(tile, tileImageData);
          }
        }
      });
      
      const processedDataUrl = imageDataToDataUrl(processedImageData);
      
      setProcessingProgress(90);
      
      // Update state with processed image for the specific model
      setImages(prev => ({
        ...prev,
        [targetModel]: processedDataUrl,
      }));
      
      setProcessingProgress(100);
      
    } catch (error) {
      console.error('Error processing image:', error);
      
      // Fallback to original image if processing fails
      console.log('Falling back to original image');
      if (originalDataUrl) {
        const targetModel = specificModel || (modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left);
        setImages(prev => ({
          ...prev,
          [targetModel]: originalDataUrl,
        }));
      }
      
      setProcessingProgress(100);
      
      // Show error to user
      alert('Image processing failed. Showing original image instead. Error: ' + 
            (error instanceof Error ? error.message : 'Unknown error'));
      processingRef.current = false;
      setProcessingProgress(0);
      setCurrentTile(null);
    } finally {
      setIsProcessing(false);
      processingRef.current = false;
      setProcessingProgress(0);
      setCurrentTile(null);
    }
  }, [modelInitialized, modelSelection]); // Add dependencies to update when modelInitialized changes

  const handleImageUpload = useCallback(async (file: File) => {
    if (processingRef.current) return;
    
    if (modelSelection.mode === 'single') {
      // Single mode - just process with selected model
      await processImageWithModel(file, modelSelection.single, false);
    } else {
      // Compare mode - process both models sequentially
      console.log('Compare mode upload: processing both models sequentially');
      
      // Process the left model first
      console.log('Processing left model:', modelSelection.left);
      await processImageWithModel(file, modelSelection.left, false);
      
      // If right model is different, load it and process
      if (modelSelection.left !== modelSelection.right) {
        // Reset processing flag to allow second model
        processingRef.current = false;
        
        console.log('Loading right model:', modelSelection.right);
        const rightModelPath = getModelPath(modelSelection.right);
        await onnxService.initializeModel(rightModelPath);
        
        console.log('Processing right model:', modelSelection.right);
        // Skip setting original since it's already set from first model
        await processImageWithModel(file, modelSelection.right, true);
      }
    }
  }, [processImageWithModel, modelSelection, getModelPath]);

  const handleClearImage = useCallback(() => {
    setImages({ original: null, srgan: null, srresnet: null, srragan: null, srranet: null });
    setCurrentFile(null);
  }, []);

  const getModelImage = (model: SingleModelType): string | null => {
    return images[model];
  };

  const getModelName = (model: SingleModelType) => {
    const modelNames = {
      srgan: 'SRGAN',
      srresnet: 'SRResNet',
      srragan: 'SRRaGAN',
      srranet: 'SRRaNet'
    };
    return `${modelNames[model]} (ONNX)`;
  };

  const renderComparison = () => {
    if (!images.original) return null;

    // Compare mode with different models
    if (isComparing) {
      const leftImage = getModelImage(modelSelection.left);
      const rightImage = getModelImage(modelSelection.right);

      if (!leftImage || !rightImage) return null;

      return (
        <BothModelsView
          originalImage={images.original}
          leftImage={leftImage}
          rightImage={rightImage}
          leftModelName={getModelName(modelSelection.left)}
          rightModelName={getModelName(modelSelection.right)}
          processingTile={isProcessing ? currentTile : null}
        />
      );
    }

    // Single model mode OR compare mode with same model
    const selectedModel = modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left;
    const processedImage = getModelImage(selectedModel);

    if (!processedImage) return null;

    const props = {
      originalImage: images.original,
      processedImage,
      modelName: getModelName(selectedModel),
      processingTile: isProcessing ? currentTile : null,
    };

    switch (viewMode) {
      case 'side-by-side':
        return <SideBySideView {...props} />;
      case 'slider':
        return <SliderView {...props} />;
      case 'sync-zoom':
        return <SyncZoomView {...props} />;
      default:
        return <SideBySideView {...props} />;
    }
  };

  const getResultTitle = () => {
    if (isComparing) {
      return `${getModelName(modelSelection.left)} vs ${getModelName(modelSelection.right)}`;
    }
    const selectedModel = modelSelection.mode === 'single' ? modelSelection.single : modelSelection.left;
    return `${getModelName(selectedModel)} Result`;
  };

  const useDemoImages = async () => {
    try {
      setIsProcessing(true);
      setProcessingProgress(0);
      
      // Load demo image
      const response = await fetch(DEMO_IMAGES.original);
      const blob = await response.blob();
      const file = new File([blob], 'demo.jpg', { type: 'image/jpeg' });
      
      // Process the demo image
      await processImageWithModel(file);
    } catch (error) {
      console.error('Error loading demo image:', error);
      setImages({
        original: DEMO_IMAGES.original,
        srgan: DEMO_IMAGES.original, // Fallback to original if processing fails
        srresnet: DEMO_IMAGES.original,
        srragan: DEMO_IMAGES.original,
        srranet: DEMO_IMAGES.original,
      });
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

  // Show loading screen until model is ready
  if (!modelInitialized) {
    return (
      <LoadingScreen
        modelsStatus={modelsStatus}
        message={modelLoadError || undefined}
        onRetry={handleRetryModelLoad}
        hasError={!!modelLoadError}
      />
    );
  }

  return (
    <div className="min-h-screen bg-void grid-bg relative">
      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Background gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent/10 rounded-full blur-[128px]" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-bright/10 rounded-full blur-[128px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-accent/20 border border-accent/30">
              <Wand2 className="w-6 h-6 text-accent-bright" />
            </div>
            <div>
              <h1 className="font-display font-bold text-xl text-pearl">SR Compare</h1>
              <p className="text-xs text-silver">Super Resolution Model Comparison</p>
            </div>
          </div>
          <a
            href="https://github.com/sirreidlos/aol-dl"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg bg-obsidian/50 border border-white/5 text-silver hover:text-pearl hover:border-white/10 transition-all"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="glass rounded-2xl p-6 mb-8 animate-slide-up relative z-[9999]">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Upload Section */}
            <div className="lg:col-span-1">
              <label className="block text-sm font-medium text-silver uppercase tracking-wider mb-3">
                Input Image
              </label>
              <ImageUpload
                onImageUpload={handleImageUpload}
                currentImage={images.original}
                onClear={handleClearImage}
              />
              {!images.original && (
                <button
                  onClick={useDemoImages}
                  className="mt-3 w-full py-2 px-4 rounded-lg bg-slate/30 border border-white/5 text-sm text-silver hover:text-pearl hover:border-white/10 transition-all flex items-center justify-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Use Demo Image
                </button>
              )}
            </div>

            {/* Model & View Selection */}
            <div className="lg:col-span-2 space-y-6">
              <ModelSelector selection={modelSelection} onSelectionChange={setModelSelection} />
              {!isComparing && (
                <ViewModeToggle viewMode={viewMode} onViewModeChange={setViewMode} />
              )}
              {isComparing && (
                <p className="text-xs text-silver/70 text-center bg-slate/20 rounded-lg p-3 border border-white/5">
                  Comparing {getModelName(modelSelection.left)} (left) vs {getModelName(modelSelection.right)} (right)
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Processing State */}
        {isProcessing && (
          <div className="glass rounded-2xl p-12 text-center animate-fade-in">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/20 mb-4">
              <Wand2 className="w-8 h-8 text-accent-bright animate-pulse" />
            </div>
            <h3 className="font-display font-semibold text-xl text-pearl mb-2">
              {processingProgress === 0 ? 'Initializing AI model...' : 'Processing your image...'}
            </h3>
            <p className="text-silver">
              {processingProgress === 0 
                ? 'Loading the super resolution model. This may take a few seconds.'
                : 'Running super resolution models. This may take a few seconds.'
              }
            </p>
            <div className="mt-6 w-48 h-1 bg-slate/50 rounded-full mx-auto overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-accent to-accent-bright transition-all duration-300 ease-out" 
                style={{ 
                  width: `${processingProgress}%`,
                  backgroundSize: '200% 100%',
                  transitionProperty: 'width',
                  transitionDuration: '300ms',
                  transitionTimingFunction: 'ease-out'
                }} 
              />
            </div>
            <p className="text-xs text-silver mt-2">
              {processingProgress === 0 
                ? 'Loading model...'
                : processingProgress > 0 
                  ? `Processing: ${Math.round(processingProgress)}%`
                  : 'Initializing model...'
              }
            </p>
          </div>
        )}

        {/* Inference Progress Visualization */}
        {isProcessing && inputImageData && (
          <div className="glass rounded-2xl p-6 animate-slide-up" style={{ animationDelay: '100ms' }}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-display font-semibold text-lg text-pearl">
                {isProcessing ? 'Processing Progress' : 'Processing Complete'}
              </h2>
              <div className="flex items-center gap-2 text-sm text-silver">
                <span className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-500' : 'bg-green-500'} animate-pulse`} />
                {isProcessing ? 'Processing tiles...' : 'Processing complete'}
              </div>
            </div>
            <OverlayProgress
              inputImageData={inputImageData}
              currentTile={currentTile}
              progress={processingProgress}
              scale={4}
              onTileUpdate={(updateFn) => {
                overlayOutputUpdateRef.current = updateFn;
              }}
            />
          </div>
        )}

        {/* Comparison View */}
        {images.original && !isProcessing && (
          <div className="glass rounded-2xl p-6 animate-slide-up" style={{ animationDelay: '100ms' }}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-display font-semibold text-lg text-pearl">
                {getResultTitle()}
              </h2>
              <div className="flex items-center gap-2 text-sm text-silver">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                Processing complete
              </div>
            </div>
            {renderComparison()}
          </div>
        )}

        {/* Empty State */}
        {!isProcessing && !images.original && (
          <div className="glass rounded-2xl p-12 text-center animate-slide-up" style={{ animationDelay: '100ms' }}>
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-slate/30 mb-6">
              <Wand2 className="w-10 h-10 text-silver" />
            </div>
            <h3 className="font-display font-semibold text-2xl text-pearl mb-3">
              Ready to enhance your images
            </h3>
            <p className="text-silver max-w-md mx-auto">
              Upload an image to see how our super resolution models can upscale and enhance it. 
              Choose a single model or compare different models side by side.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 text-center">
          <p className="text-sm text-silver">
            Built with React + TypeScript • Powered by PyTorch SR Models
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
