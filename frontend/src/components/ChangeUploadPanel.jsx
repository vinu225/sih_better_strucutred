import React, { useRef, useState, useEffect } from 'react';
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Trash2,
  ArrowRightLeft,
  ArrowLeftRight,
} from 'lucide-react';
import { processUploadedFile } from '../demo/changeDemo';

export default function ChangeUploadPanel({
  beforeImage,
  afterImage,
  setBeforeImage,
  setAfterImage,
  beforeDate,
  setBeforeDate,
  afterDate,
  setAfterDate,
  isReversed = false,
  onSwapSlots,
}) {
  const beforeInputRef = useRef(null);
  const afterInputRef = useRef(null);
  const [beforeDragOver, setBeforeDragOver] = useState(false);
  const [afterDragOver, setAfterDragOver] = useState(false);
  const [beforeWarning, setBeforeWarning] = useState(null);
  const [afterWarning, setAfterWarning] = useState(null);
  const [aspectRatioMismatch, setAspectRatioMismatch] = useState(false);

  // Check aspect ratio divergence (> 10%)
  useEffect(() => {
    if (beforeImage?.aspectRatio && afterImage?.aspectRatio) {
      const diff =
        Math.abs(beforeImage.aspectRatio - afterImage.aspectRatio) /
        beforeImage.aspectRatio;
      setAspectRatioMismatch(diff > 0.1);
    } else {
      setAspectRatioMismatch(false);
    }
  }, [beforeImage?.aspectRatio, afterImage?.aspectRatio]);

  const processFile = async (file, isBefore) => {
    const setWarning = isBefore ? setBeforeWarning : setAfterWarning;
    const setImage = isBefore ? setBeforeImage : setAfterImage;

    setWarning(null);
    if (!file) return;

    const lowerName = file.name.toLowerCase();
    if (
      lowerName.endsWith('.tif') ||
      lowerName.endsWith('.tiff') ||
      lowerName.endsWith('.npy') ||
      lowerName.endsWith('.npz')
    ) {
      setWarning(
        'Preview not available for this format in demo mode. Please use a PNG or JPG.'
      );
      return;
    }

    if (!file.type.startsWith('image/')) {
      setWarning('Please upload a valid image file (.png, .jpg, .webp).');
      return;
    }

    try {
      const processed = await processUploadedFile(file);
      setImage({
        file,
        name: file.name,
        url: processed.dataUrl,
        aspectRatio: processed.aspectRatio,
        width: processed.width,
        height: processed.height,
        fingerprint: processed.fingerprint,
      });
    } catch (err) {
      console.error('File load failed:', err);
      setWarning('Failed to process image file.');
    }
  };

  const handleRemove = (isBefore) => {
    const setter = isBefore ? setBeforeImage : setAfterImage;
    setter(null);
  };

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft mb-5">
      {/* Header with full width layout */}
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-xl bg-brand-blue-tint text-brand-blue flex items-center justify-center shrink-0">
          <ArrowRightLeft className="w-5 h-5" />
        </div>
        <div>
          <h2 className="font-display font-normal text-[22px] sm:text-[24px] tracking-[-0.01em] leading-[1.15] text-brand-dark">
            Upload Images for Comparison
          </h2>
          <p className="text-[12px] text-brand-muted mt-0.5">
            Upload two images from different time periods
          </p>
        </div>
      </div>

      {/* Hidden File Inputs */}
      <input
        type="file"
        ref={beforeInputRef}
        onChange={(e) =>
          e.target.files?.[0] && processFile(e.target.files[0], true)
        }
        accept="image/png,image/jpeg,image/webp,.tif,.tiff,.npy"
        className="hidden"
      />
      <input
        type="file"
        ref={afterInputRef}
        onChange={(e) =>
          e.target.files?.[0] && processFile(e.target.files[0], false)
        }
        accept="image/png,image/jpeg,image/webp,.tif,.tiff,.npy"
        className="hidden"
      />

      {/* Dual Upload Slots Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
        {/* SLOT 1: BEFORE */}
        <div className="flex flex-col">
          <div
            onDrop={(e) => {
              e.preventDefault();
              setBeforeDragOver(false);
              if (e.dataTransfer.files?.[0])
                processFile(e.dataTransfer.files[0], true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setBeforeDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setBeforeDragOver(false);
            }}
            className={`relative rounded-2xl border-2 transition-all duration-200 p-4 flex flex-col items-center justify-center min-h-[160px] text-center ${
              beforeImage
                ? 'border-slate-200 bg-slate-50/70'
                : beforeDragOver
                ? 'border-brand-blue bg-blue-50/60 scale-[1.01] border-dashed'
                : 'border-dashed border-slate-200 bg-slate-50/40 hover:bg-slate-50 hover:border-slate-300'
            }`}
          >
            {/* Slot Tag */}
            <div className="absolute top-2.5 left-2.5 z-10">
              <span className="px-2.5 py-0.5 rounded-full bg-slate-200/90 text-slate-700 text-[11px] font-bold shadow-2xs">
                Before
              </span>
            </div>

            {beforeImage ? (
              <div className="w-full h-full flex flex-col items-center justify-center">
                <div className="relative w-full h-28 rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
                  <img
                    src={beforeImage.url}
                    alt="Before observation preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1.5 right-1.5 bg-emerald-500/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 shadow-xs backdrop-blur-xs">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Loaded</span>
                  </div>
                </div>

                <div className="w-full flex items-center justify-between mt-2.5 px-0.5">
                  <span
                    className="text-xs font-semibold text-brand-dark truncate max-w-[100px]"
                    title={beforeImage.name}
                  >
                    {beforeImage.name}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      type="button"
                      onClick={() => beforeInputRef.current?.click()}
                      className="text-[11px] font-semibold text-brand-blue hover:text-blue-700 bg-brand-blue-tint hover:bg-blue-100 px-2 py-0.5 rounded-md transition-colors flex items-center gap-1 cursor-pointer"
                    >
                      <RefreshCw className="w-2.5 h-2.5" />
                      <span>Replace</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRemove(true)}
                      className="text-[11px] font-semibold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 p-1 rounded-md transition-colors cursor-pointer"
                      title="Remove image"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-2">
                <div className="w-10 h-10 rounded-full bg-brand-blue-tint text-brand-blue flex items-center justify-center mb-2 shadow-2xs">
                  <UploadCloud className="w-5 h-5" />
                </div>
                <p className="text-xs font-medium text-brand-dark mb-0.5">
                  Drag & drop first image
                </p>
                <span className="text-[11px] text-brand-muted mb-2">or</span>
                <button
                  type="button"
                  onClick={() => beforeInputRef.current?.click()}
                  className="px-3.5 py-1.5 bg-brand-blue hover:bg-blue-600 text-white text-xs font-bold rounded-xl shadow-xs shadow-brand-blue/20 transition-colors cursor-pointer"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>

          {/* Warning notice for unsupported format */}
          {beforeWarning && (
            <div className="mt-1.5 p-2 rounded-lg bg-red-50 border border-red-100 text-[11px] text-red-700 flex items-start gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-red-500 shrink-0 mt-0.5" />
              <span>{beforeWarning}</span>
            </div>
          )}

          {/* Date / Year input */}
          <div className="mt-2">
            <input
              type="text"
              maxLength={12}
              value={beforeDate}
              onChange={(e) => setBeforeDate(e.target.value)}
              placeholder="Date or year (optional)"
              className="w-full px-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-brand-dark placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-brand-blue focus:border-brand-blue"
            />
          </div>
        </div>

        {/* SLOT 2: AFTER */}
        <div className="flex flex-col">
          <div
            onDrop={(e) => {
              e.preventDefault();
              setAfterDragOver(false);
              if (e.dataTransfer.files?.[0])
                processFile(e.dataTransfer.files[0], false);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setAfterDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setAfterDragOver(false);
            }}
            className={`relative rounded-2xl border-2 transition-all duration-200 p-4 flex flex-col items-center justify-center min-h-[160px] text-center ${
              afterImage
                ? 'border-slate-200 bg-slate-50/70'
                : afterDragOver
                ? 'border-brand-blue bg-blue-50/60 scale-[1.01] border-dashed'
                : 'border-dashed border-slate-200 bg-slate-50/40 hover:bg-slate-50 hover:border-slate-300'
            }`}
          >
            {/* Slot Tag */}
            <div className="absolute top-2.5 left-2.5 z-10">
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold shadow-2xs">
                After
              </span>
            </div>

            {afterImage ? (
              <div className="w-full h-full flex flex-col items-center justify-center">
                <div className="relative w-full h-28 rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
                  <img
                    src={afterImage.url}
                    alt="After observation preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1.5 right-1.5 bg-emerald-500/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 shadow-xs backdrop-blur-xs">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Loaded</span>
                  </div>
                </div>

                <div className="w-full flex items-center justify-between mt-2.5 px-0.5">
                  <span
                    className="text-xs font-semibold text-brand-dark truncate max-w-[100px]"
                    title={afterImage.name}
                  >
                    {afterImage.name}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      type="button"
                      onClick={() => afterInputRef.current?.click()}
                      className="text-[11px] font-semibold text-brand-blue hover:text-blue-700 bg-brand-blue-tint hover:bg-blue-100 px-2 py-0.5 rounded-md transition-colors flex items-center gap-1 cursor-pointer"
                    >
                      <RefreshCw className="w-2.5 h-2.5" />
                      <span>Replace</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRemove(false)}
                      className="text-[11px] font-semibold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 p-1 rounded-md transition-colors cursor-pointer"
                      title="Remove image"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-2">
                <div className="w-10 h-10 rounded-full bg-brand-blue-tint text-brand-blue flex items-center justify-center mb-2 shadow-2xs">
                  <UploadCloud className="w-5 h-5" />
                </div>
                <p className="text-xs font-medium text-brand-dark mb-0.5">
                  Drag & drop second image
                </p>
                <span className="text-[11px] text-brand-muted mb-2">or</span>
                <button
                  type="button"
                  onClick={() => afterInputRef.current?.click()}
                  className="px-3.5 py-1.5 bg-brand-blue hover:bg-blue-600 text-white text-xs font-bold rounded-xl shadow-xs shadow-brand-blue/20 transition-colors cursor-pointer"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>

          {/* Warning notice for unsupported format */}
          {afterWarning && (
            <div className="mt-1.5 p-2 rounded-lg bg-red-50 border border-red-100 text-[11px] text-red-700 flex items-start gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-red-500 shrink-0 mt-0.5" />
              <span>{afterWarning}</span>
            </div>
          )}

          {/* Date / Year input */}
          <div className="mt-2">
            <input
              type="text"
              maxLength={12}
              value={afterDate}
              onChange={(e) => setAfterDate(e.target.value)}
              placeholder="Date or year (optional)"
              className="w-full px-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-brand-dark placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-brand-blue focus:border-brand-blue"
            />
          </div>
        </div>
      </div>

      {/* Reversed Order Notice & Swap Action */}
      {isReversed && (
        <div className="mt-3.5 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 flex items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 min-w-0">
            <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
            <span className="font-medium truncate">
              These look like they are in reverse order.
            </span>
          </div>
          <button
            type="button"
            onClick={onSwapSlots}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs shadow-xs transition-colors shrink-0 cursor-pointer"
          >
            <ArrowLeftRight className="w-3.5 h-3.5" />
            <span>Swap</span>
          </button>
        </div>
      )}

      {/* Aspect Ratio Mismatch Warning */}
      {aspectRatioMismatch && !isReversed && (
        <div className="mt-3.5 p-2.5 rounded-xl bg-amber-50/90 border border-amber-200 text-amber-800 flex items-center gap-2 text-xs">
          <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
          <span>
            The images have different shapes; the comparison may be misaligned.
          </span>
        </div>
      )}

      {/* Helper text under slots */}
      <div className="mt-3.5 text-center">
        <p className="text-[11px] text-brand-muted">
          Supports: <span className="font-semibold text-slate-600">.png, .jpg, .webp</span>
        </p>
      </div>
    </div>
  );
}
