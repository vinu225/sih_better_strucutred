import React, { useRef, useState } from 'react';
import { UploadCloud, FileCode, CheckCircle2, AlertCircle, RefreshCw, Layers } from 'lucide-react';
import { useChat } from '../context/ChatContext';

export default function UploadPanel() {
  const { activeTile, uploadState, uploadError, handleFileUpload } = useChat();
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft mb-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-lg bg-brand-blue-tint text-brand-blue">
          <UploadCloud className="w-4 h-4" />
        </div>
        <div>
          <h2 className="font-display font-normal text-[22px] sm:text-[24px] tracking-[-0.01em] leading-[1.15] text-brand-dark">Upload Satellite Image</h2>
          <p className="text-[12px] text-brand-muted">Drag and drop an image here or browse files</p>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".jpg,.jpeg,.png,.tif,.tiff,.npy,.npz,.bmp,.webp"
        className="hidden"
      />

      {uploadState === 'ready' && activeTile ? (
        <div className="space-y-3.5 pt-1">
          {/* Active Image Preview Card */}
          <div className="relative rounded-xl overflow-hidden border border-brand-border bg-slate-50 flex flex-col items-center justify-center min-h-[140px] p-2">
            {activeTile.isNpy || !activeTile.previewUrl ? (
              <div className="flex flex-col items-center justify-center p-4 text-center">
                <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mb-2">
                  <FileCode className="w-6 h-6" />
                </div>
                <span className="text-xs font-semibold text-brand-dark truncate max-w-[220px]">
                  {activeTile.filename}
                </span>
                {activeTile.shape && (
                  <span className="text-[11px] text-brand-muted mt-0.5">
                    Tensor Shape: [{activeTile.shape.join(', ')}]
                  </span>
                )}
              </div>
            ) : (
              <img
                src={activeTile.previewUrl || `/api/v1/tiles/${activeTile.tileId}/composite?mode=rgb`}
                alt="Uploaded satellite patch"
                className="w-full h-36 object-cover rounded-lg"
                onError={(e) => {
                  e.currentTarget.src = `/api/v1/tiles/${activeTile.tileId}/composite?mode=rgb`;
                }}
              />
            )}

            <div className="absolute top-2 right-2 bg-emerald-500/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 shadow-sm backdrop-blur-xs">
              <CheckCircle2 className="w-3 h-3" />
              <span>Loaded</span>
            </div>
          </div>

          {/* Detected Modality Badge */}
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 border border-slate-100">
            <div className="flex items-center gap-2 min-w-0">
              <Layers className="w-4 h-4 text-brand-blue shrink-0" />
              <div className="truncate">
                <p className="text-[11px] text-brand-muted font-medium uppercase tracking-wider">Detected Modality</p>
                <p className="text-xs font-bold text-brand-dark truncate">{activeTile.displayLabel}</p>
              </div>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-xs font-semibold text-brand-blue hover:text-blue-700 bg-brand-blue-tint hover:bg-blue-100 px-2.5 py-1 rounded-lg transition-colors shrink-0 flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Replace</span>
            </button>
          </div>
        </div>
      ) : (
        /* Dropzone Box */
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center transition-all duration-200 ${isDragOver
              ? 'border-brand-blue bg-blue-50/50 scale-[1.01]'
              : 'border-slate-200 bg-slate-50/40 hover:bg-slate-50 hover:border-slate-300'
            }`}
        >
          {uploadState === 'uploading' ? (
            <div className="flex flex-col items-center py-4 space-y-2">
              <div className="w-8 h-8 border-3 border-brand-blue border-t-transparent rounded-full animate-spin" />
              <p className="text-xs font-semibold text-brand-dark">Standardizing & Uploading Tile...</p>
              <p className="text-[11px] text-brand-muted">Detecting multi-channel sensor spec</p>
            </div>
          ) : (
            <>
              <div className="w-12 h-12 rounded-full bg-brand-blue-tint text-brand-blue flex items-center justify-center mb-3 shadow-sm">
                <UploadCloud className="w-6 h-6" />
              </div>
              <p className="text-xs font-medium text-brand-dark mb-1">
                Drag & drop an image here <span className="text-brand-muted">or</span>
              </p>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="my-2 px-4 py-2 bg-brand-blue hover:bg-blue-600 text-white text-xs font-bold rounded-xl shadow-sm shadow-brand-blue/20 transition-colors focus:ring-2 focus:ring-brand-blue/30 outline-hidden"
              >
                Browse Files
              </button>
              <p className="text-[11px] text-brand-muted mt-2">
                Supports: <span className="font-semibold text-slate-600">.tif, .tiff, .png, .jpg, .npy</span>
                <br />
                (Optical, SAR or Multispectral)
              </p>
            </>
          )}
        </div>
      )}

      {/* Upload Error Banner */}
      {uploadState === 'error' && uploadError && (
        <div className="mt-3 p-3 rounded-xl bg-red-50 border border-red-100 flex items-start gap-2 text-xs text-red-700">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-500" />
          <div className="flex-1">
            <p className="font-semibold">Upload failed</p>
            <p className="text-[11px] text-red-600 mt-0.5">{uploadError}</p>
          </div>
        </div>
      )}
    </div>
  );
}
