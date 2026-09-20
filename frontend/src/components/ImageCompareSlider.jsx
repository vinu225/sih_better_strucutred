import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  ArrowLeftRight,
  Maximize2,
  Minimize2,
  MapPin,
  AlertCircle,
  RefreshCw,
  Layers,
  Sliders,
  Sparkles,
} from 'lucide-react';

/**
 * ImageCompareSlider - Interactive Before / After Split Slider.
 * Supports Split comparison view and lazy Difference Heatmap overlay view.
 */
export default function ImageCompareSlider({
  beforeUrl,
  afterUrl,
  diffUrl = null,
  beforeDate = null,
  afterDate = null,
  location = null,
  illustrative = false,
  initialView = 'split', // 'split' | 'diff'
  className = '',
}) {
  const [position, setPosition] = useState(50); // 0 to 100%
  const [viewMode, setViewMode] = useState(initialView); // 'split' | 'diff'
  const [diffOpacity, setDiffOpacity] = useState(0.75); // 0.0 to 1.0
  const [isDragging, setIsDragging] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Load states for primary comparison images
  const [beforeLoaded, setBeforeLoaded] = useState(false);
  const [afterLoaded, setAfterLoaded] = useState(false);
  const [beforeError, setBeforeError] = useState(false);
  const [afterError, setAfterError] = useState(false);

  // Lazy load state for difference map
  const [diffLoaded, setDiffLoaded] = useState(false);
  const [diffError, setDiffError] = useState(false);

  const [aspectRatio, setAspectRatio] = useState(null);
  const [loadTimedOut, setLoadTimedOut] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  const containerRef = useRef(null);

  useEffect(() => {
    if (initialView) {
      setViewMode(initialView);
    }
  }, [initialView]);

  // 8-second loading timeout for primary images
  useEffect(() => {
    setLoadTimedOut(false);
    if (beforeLoaded && afterLoaded) return;

    const timer = setTimeout(() => {
      if (!beforeLoaded || !afterLoaded) {
        setLoadTimedOut(true);
      }
    }, 8000);

    return () => clearTimeout(timer);
  }, [beforeLoaded, afterLoaded, reloadKey, beforeUrl, afterUrl]);

  const isLoading = (!beforeLoaded && !beforeError) || (!afterLoaded && !afterError);
  const hasError = beforeError || afterError || (loadTimedOut && isLoading);

  const handleBeforeLoad = (e) => {
    setBeforeLoaded(true);
    setBeforeError(false);
    if (e.target.naturalWidth && e.target.naturalHeight) {
      setAspectRatio(e.target.naturalWidth / e.target.naturalHeight);
    }
  };

  const handleAfterLoad = () => {
    setAfterLoaded(true);
    setAfterError(false);
  };

  const handleDiffLoad = () => {
    setDiffLoaded(true);
    setDiffError(false);
  };

  const handleDiffError = () => {
    setDiffError(true);
  };

  const handleRetry = () => {
    setBeforeError(false);
    setAfterError(false);
    setBeforeLoaded(false);
    setAfterLoaded(false);
    setLoadTimedOut(false);
    setDiffError(false);
    setReloadKey((prev) => prev + 1);
  };

  const updatePositionFromClientX = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clamped = Math.max(0, Math.min(rect.width, x));
    const newPos = (clamped / rect.width) * 100;
    setPosition(Math.round(newPos * 10) / 10);
  }, []);

  const handlePointerDown = (e) => {
    if (hasError || isLoading) return;
    if (
      e.target.closest('button') ||
      e.target.closest('input') ||
      e.target.closest('.no-drag')
    ) {
      return;
    }
    setIsDragging(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    updatePositionFromClientX(e.clientX);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    updatePositionFromClientX(e.clientX);
  };

  const handlePointerUp = (e) => {
    if (!isDragging) return;
    setIsDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };

  // Keyboard accessibility
  const handleKeyDown = (e) => {
    let delta = 0;
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft') delta = -step;
    else if (e.key === 'ArrowRight') delta = step;
    else if (e.key === 'Home') {
      e.preventDefault();
      setPosition(0);
      return;
    } else if (e.key === 'End') {
      e.preventDefault();
      setPosition(100);
      return;
    }

    if (delta !== 0) {
      e.preventDefault();
      setPosition((prev) => Math.max(0, Math.min(100, prev + delta)));
    }
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen().catch(() => {
          setIsFullscreen((prev) => !prev);
        });
      } else {
        setIsFullscreen((prev) => !prev);
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isFullscreen]);

  const beforeLabel = beforeDate ? `Before (${beforeDate})` : 'Before';
  const afterLabel =
    viewMode === 'diff'
      ? 'Difference Map'
      : afterDate
      ? `After (${afterDate})`
      : 'After';

  const containerStyle = isFullscreen
    ? { height: '100vh', width: '100vw' }
    : aspectRatio
    ? {
        aspectRatio: `${aspectRatio}`,
        maxHeight: 'min(60vh, 520px)',
      }
    : {
        height: '380px',
        maxHeight: 'min(60vh, 520px)',
      };

  return (
    <div key={reloadKey} className="space-y-2">
      {/* Top Controls Bar (View Toggle & Opacity Slider when Diff is Available) */}
      {diffUrl && (
        <div className="flex flex-wrap items-center justify-between gap-2 px-1">
          {/* View Mode Toggle Button Group */}
          <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200 shadow-2xs">
            <button
              type="button"
              onClick={() => setViewMode('split')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                viewMode === 'split'
                  ? 'bg-white text-brand-blue shadow-2xs'
                  : 'text-slate-600 hover:text-brand-dark'
              }`}
            >
              Before / After
            </button>
            <button
              type="button"
              onClick={() => setViewMode('diff')}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                viewMode === 'diff'
                  ? 'bg-white text-brand-blue shadow-2xs'
                  : 'text-slate-600 hover:text-brand-dark'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Difference Map</span>
            </button>
          </div>

          {/* Opacity Control & Legend in Diff Mode */}
          {viewMode === 'diff' && !diffError && (
            <div className="flex items-center gap-3 text-xs bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg">
              <div className="flex items-center gap-1.5">
                <Sliders className="w-3 h-3 text-slate-500" />
                <span className="text-[11px] text-slate-600 font-medium">Opacity:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={diffOpacity}
                  onChange={(e) => setDiffOpacity(parseFloat(e.target.value))}
                  className="w-16 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-blue"
                />
                <span className="text-[10px] font-mono text-slate-500 w-7">
                  {Math.round(diffOpacity * 100)}%
                </span>
              </div>

              {/* Heatmap Legend */}
              <div className="hidden sm:flex items-center gap-1.5 border-l border-slate-200 pl-3">
                <span className="text-[10px] text-slate-500">Change:</span>
                <div className="w-14 h-2 rounded-full bg-gradient-to-r from-yellow-300 via-orange-400 to-red-500" />
                <span className="text-[10px] text-slate-600 font-medium">Low → High</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Slider Canvas */}
      <div
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        style={containerStyle}
        className={`relative overflow-hidden rounded-2xl border border-brand-border bg-slate-950 select-none shadow-soft touch-pan-y group ${
          isFullscreen
            ? 'fixed inset-0 z-50 rounded-none border-none max-h-none'
            : 'w-full'
        } ${className}`}
      >
        {/* Loading Skeleton */}
        {isLoading && (
          <div className="absolute inset-0 bg-slate-900 flex flex-col items-center justify-center text-slate-400 z-30 animate-pulse">
            <div className="w-10 h-10 border-2 border-brand-blue border-t-transparent rounded-full animate-spin mb-3" />
            <p className="text-xs font-medium">Aligning and loading comparison views...</p>
          </div>
        )}

        {/* Error Fallback */}
        {hasError && (
          <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center z-30 text-slate-300">
            <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
            <p className="text-sm font-semibold text-white">Image could not be loaded</p>
            <p className="text-xs text-slate-400 mt-1 max-w-xs">
              One of the comparison images failed to load.
            </p>
            <button
              type="button"
              onClick={handleRetry}
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-blue hover:bg-blue-600 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* 1. Base Layer (Right of Divider) */}
        {viewMode === 'split' ? (
          <img
            src={afterUrl}
            alt="After satellite observation"
            draggable={false}
            onLoad={handleAfterLoad}
            onError={() => setAfterError(true)}
            className="absolute inset-0 w-full h-full object-cover object-center pointer-events-none"
          />
        ) : (
          /* Difference Map View: Base image with lazy diff.png overlay */
          <div className="absolute inset-0 w-full h-full pointer-events-none">
            <img
              src={beforeUrl}
              alt="Base observation for difference overlay"
              draggable={false}
              className="absolute inset-0 w-full h-full object-cover object-center"
            />
            {diffUrl && !diffError && (
              <img
                src={diffUrl}
                alt="Difference heatmap overlay"
                draggable={false}
                onLoad={handleDiffLoad}
                onError={handleDiffError}
                style={{ opacity: diffOpacity }}
                className="absolute inset-0 w-full h-full object-cover object-center"
              />
            )}
            {diffError && (
              <div className="absolute inset-0 bg-slate-950/75 flex items-center justify-center p-4 text-center">
                <span className="text-xs font-medium text-slate-300 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
                  Difference map not available
                </span>
              </div>
            )}
          </div>
        )}

        {/* 2. Top Layer: BEFORE Image (Left of Divider, Clipped) */}
        <div
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{
            clipPath: `inset(0 calc(100% - ${position}%) 0 0)`,
            WebkitClipPath: `inset(0 calc(100% - ${position}%) 0 0)`,
          }}
        >
          <img
            src={beforeUrl}
            alt="Before satellite observation"
            draggable={false}
            onLoad={handleBeforeLoad}
            onError={() => setBeforeError(true)}
            className="absolute inset-0 w-full h-full object-cover object-center pointer-events-none"
          />
        </div>

        {/* 3. Split Divider Line */}
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-white pointer-events-none z-20 shadow-[0_0_10px_rgba(0,0,0,0.5)]"
          style={{ left: `${position}%` }}
        />

        {/* 4. Draggable / Focusable Central Handle */}
        <div
          role="slider"
          tabIndex={0}
          aria-orientation="horizontal"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(position)}
          aria-label="Before and after comparison"
          onKeyDown={handleKeyDown}
          style={{ left: `${position}%` }}
          className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-white text-slate-700 shadow-lg border-2 border-slate-200/80 flex items-center justify-center cursor-ew-resize z-25 focus:outline-none focus:ring-4 focus:ring-brand-blue/40 transition-transform active:scale-95 group-hover:scale-105"
        >
          <ArrowLeftRight className="w-5 h-5 text-slate-700 pointer-events-none" />
        </div>

        {/* 5. Top-Left Pill: Before Badge */}
        <div className="absolute top-3 left-3 z-20 pointer-events-none flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/65 backdrop-blur-xs text-white text-xs font-semibold border border-white/20 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-slate-300" />
            {beforeLabel}
          </span>
          {illustrative && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-cyan-950/80 backdrop-blur-xs text-cyan-300 text-[10px] font-semibold border border-cyan-400/30 shadow-sm">
              <Sparkles className="w-2.5 h-2.5" />
              <span>Illustrative sample</span>
            </span>
          )}
        </div>

        {/* 6. Top-Right Pill: After / Difference Badge */}
        <div className="absolute top-3 right-3 z-20 pointer-events-none">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/65 backdrop-blur-xs text-white text-xs font-semibold border border-white/20 shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                viewMode === 'diff' ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'
              }`}
            />
            {afterLabel}
          </span>
        </div>

        {/* 7. Bottom-Left: Location Pill (If Provided and non-null) */}
        {location && (
          <div className="absolute bottom-3 left-3 z-20 pointer-events-none">
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/65 backdrop-blur-xs text-white text-[11px] font-medium border border-white/20 shadow-sm">
              <MapPin className="w-3 h-3 text-cyan-400" />
              <span className="truncate max-w-[200px]">{location}</span>
            </span>
          </div>
        )}

        {/* 8. Bottom-Right: Fullscreen Button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            toggleFullscreen();
          }}
          title={isFullscreen ? 'Exit Fullscreen' : 'View Fullscreen'}
          className="no-drag absolute bottom-3 right-3 z-20 p-2 rounded-xl bg-black/65 hover:bg-black/85 backdrop-blur-xs text-white border border-white/20 shadow-sm transition-all hover:scale-105 cursor-pointer focus:outline-none focus:ring-2 focus:ring-white/40"
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
