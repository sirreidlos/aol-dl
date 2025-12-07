import React from 'react';
import { Loader2, Cpu, Sparkles, CheckCircle2, XCircle } from 'lucide-react';

interface LoadingScreenProps {
  modelsStatus: {
    resnet: boolean;
    gan: boolean;
  };
  message?: string;
  onRetry: () => void;
  hasError: boolean;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  modelsStatus,
  message,
  onRetry,
  hasError,
}) => {
  return (
    <div className="min-h-screen bg-void grid-bg flex items-center justify-center relative">
      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Background gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-accent/5 rounded-full blur-[128px]" />
        <div className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-accent-bright/5 rounded-full blur-[128px]" />
      </div>

      <div className="relative z-10 text-center max-w-md mx-auto px-6">
        {/* Logo/Icon */}
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-slate/30 border border-white/10 mb-8">
          {hasError ? (
            <XCircle className="w-10 h-10 text-red-400" />
          ) : (
            <Loader2 className="w-10 h-10 text-pearl animate-spin" />
          )}
        </div>

        {/* Title */}
        <h1 className="font-display font-bold text-2xl text-pearl mb-2">
          {hasError ? 'Connection Error' : 'Loading Models'}
        </h1>
        <p className="text-silver mb-8">
          {hasError
            ? message || 'Unable to connect to the backend server'
            : 'Please wait while we initialize the super resolution models...'}
        </p>

        {/* Model Status */}
        <div className="glass rounded-xl p-4 mb-6">
          <div className="space-y-3">
            {/* SRResNet Status */}
            <div className="flex items-center justify-between p-3 rounded-lg bg-obsidian/50">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${modelsStatus.resnet ? 'bg-green-500/20' : 'bg-slate/50'}`}>
                  <Cpu className={`w-4 h-4 ${modelsStatus.resnet ? 'text-green-400' : 'text-silver'}`} />
                </div>
                <span className="text-sm font-medium text-pearl">SRResNet</span>
              </div>
              {modelsStatus.resnet ? (
                <CheckCircle2 className="w-5 h-5 text-green-400" />
              ) : hasError ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : (
                <Loader2 className="w-5 h-5 text-silver animate-spin" />
              )}
            </div>

            {/* SRGAN Status */}
            <div className="flex items-center justify-between p-3 rounded-lg bg-obsidian/50">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${modelsStatus.gan ? 'bg-green-500/20' : 'bg-slate/50'}`}>
                  <Sparkles className={`w-4 h-4 ${modelsStatus.gan ? 'text-green-400' : 'text-silver'}`} />
                </div>
                <span className="text-sm font-medium text-pearl">SRGAN</span>
              </div>
              {modelsStatus.gan ? (
                <CheckCircle2 className="w-5 h-5 text-green-400" />
              ) : hasError ? (
                <XCircle className="w-5 h-5 text-red-400" />
              ) : (
                <Loader2 className="w-5 h-5 text-silver animate-spin" />
              )}
            </div>
          </div>
        </div>

        {/* Progress bar (only show when loading) */}
        {!hasError && (
          <div className="w-full h-1 bg-slate/50 rounded-full overflow-hidden mb-6">
            <div 
              className="h-full bg-gradient-to-r from-silver to-pearl animate-shimmer" 
              style={{ backgroundSize: '200% 100%' }} 
            />
          </div>
        )}

        {/* Retry button (only show on error) */}
        {hasError && (
          <button
            onClick={onRetry}
            className="px-6 py-3 rounded-xl bg-slate/50 border border-white/10 text-pearl font-medium hover:bg-slate/70 hover:border-white/20 transition-all"
          >
            Try Again
          </button>
        )}

        {/* Tip */}
        {!hasError && (
          <p className="text-xs text-silver/60">
            Tip: First load may take longer as models are loaded into memory
          </p>
        )}
      </div>
    </div>
  );
};

