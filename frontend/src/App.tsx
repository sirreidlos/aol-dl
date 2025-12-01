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
} from './components';
import { ModelSelection, ViewMode, ImageData } from './types';

// Demo images for testing - replace with actual API integration
const DEMO_IMAGES = {
  original: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&q=80',
  resnet: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
  gan: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=95',
};

function App() {
  const [modelSelection, setModelSelection] = useState<ModelSelection>({
    mode: 'single',
    single: 'resnet',
    left: 'resnet',
    right: 'gan',
  });
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [images, setImages] = useState<ImageData>({
    original: null,
    resnet: null,
    gan: null,
  });
  const [isProcessing, setIsProcessing] = useState(false);

  // Check if we're in compare mode with different models
  const isComparing = modelSelection.mode === 'compare' && modelSelection.left !== modelSelection.right;

  // Track processing progress (0-100%)
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentTile, setCurrentTile] = useState<{x1: number, y1: number, x2: number, y2: number, current: number, total: number} | null>(null);
  const [modelInitialized, setModelInitialized] = useState(false);
  const processingRef = useRef(false);
  
  // Track ImageData for progress visualization
  const [inputImageData, setInputImageData] = useState<globalThis.ImageData | null>(null);
  const outputImageDataRef = useRef<globalThis.ImageData | null>(null);
  const overlayOutputUpdateRef = useRef<(tile: any, tileImageData: globalThis.ImageData) => void>(() => {});

  // Initialize ONNX model when component mounts
  useEffect(() => {
    console.log('App mounted, starting model initialization');
    const initModel = async () => {
      try {
        console.log('Calling onnxService.initializeModel...');
        await onnxService.initializeModel();
        console.log('ONNX model initialized successfully');
        setModelInitialized(true);
      } catch (error) {
        console.error('Failed to initialize ONNX model:', error);
      }
    };
    initModel();
  }, []);

  const processImageWithModel = useCallback(async (file: File) => {
    if (processingRef.current) return;
    
    // Ensure model is initialized
    if (!modelInitialized) {
      try {
        await onnxService.initializeModel();
        setModelInitialized(true);
      } catch (error) {
        console.error('Failed to initialize ONNX model:', error);
        throw error;
      }
    }
    
    // Declare originalDataUrl outside try-catch to make it accessible
    let originalDataUrl: string | null = null;
    
    try {
      processingRef.current = true;
      setIsProcessing(true);
      setProcessingProgress(0);
      
      // Convert file to ImageData
      const imageData = await fileToImageData(file);
      originalDataUrl = URL.createObjectURL(file);
      setImages(prev => ({ ...prev, original: originalDataUrl }));
      
      // Set input ImageData for progress visualization
      setInputImageData(imageData);
      outputImageDataRef.current = null; // Clear ref as well
      
      setProcessingProgress(30);
      
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
      
      // Update state with processed image
      setImages(prev => ({
        ...prev,
        resnet: processedDataUrl, // Using resnet for ONNX model output
        gan: processedDataUrl, // Same for GAN for now
      }));
      
      setProcessingProgress(100);
      
    } catch (error) {
      console.error('Error processing image:', error);
      
      // Fallback to original image if processing fails
      console.log('Falling back to original image');
      if (originalDataUrl) {
        setImages(prev => ({
          ...prev,
          resnet: originalDataUrl,
          gan: originalDataUrl,
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
  }, []);

  const handleImageUpload = useCallback(async (file: File) => {
    await processImageWithModel(file);
  }, [processImageWithModel]);

  const handleClearImage = useCallback(() => {
    setImages({ original: null, resnet: null, gan: null });
  }, []);

  const getModelImage = (model: 'resnet' | 'gan') => {
    return model === 'resnet' ? images.resnet : images.gan;
  };

  const getModelName = (model: 'resnet' | 'gan') => {
    if (model === 'resnet') {
      return 'SRResNet (ONNX)';
    } else {
      return 'SRGAN (ONNX)';
    }
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
        resnet: DEMO_IMAGES.original, // Fallback to original if processing fails
        gan: DEMO_IMAGES.original,
      });
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

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
        <div className="glass rounded-2xl p-6 mb-8 animate-slide-up">
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
              {modelInitialized ? 'Ready to enhance your images' : 'Loading AI model...'}
            </h3>
            <p className="text-silver max-w-md mx-auto">
              {modelInitialized 
                ? 'Upload an image to see how our super resolution models can upscale and enhance it. Choose a single model or compare SRResNet and SRGAN side by side.'
                : 'Initializing the super resolution model. This may take a few seconds...'
              }
            </p>
            {!modelInitialized && (
              <div className="mt-4 w-32 h-1 bg-slate/50 rounded-full mx-auto overflow-hidden">
                <div className="h-full bg-gradient-to-r from-accent to-accent-bright animate-shimmer" style={{ backgroundSize: '200% 100%' }} />
              </div>
            )}
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
