import React, { useRef } from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';
import { useChat } from '../context/ChatContext';

export default function ChatInput() {
  const {
    activeTile,
    uploadState,
    isAnalyzing,
    queryInput,
    setQueryInput,
    handleSendMessage,
    handleFileUpload,
  } = useChat();

  const fileInputRef = useRef(null);
  const isImageReady = uploadState === 'ready' && !!activeTile;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isImageReady || isAnalyzing || !queryInput.trim()) return;
    handleSendMessage();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="pt-3 border-t border-brand-border bg-white rounded-b-2xl">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".jpg,.jpeg,.png,.tif,.tiff,.npy,.npz,.bmp,.webp"
        className="hidden"
      />

      <form onSubmit={handleSubmit} className="flex items-center gap-2 relative">
        {/* Paperclip upload trigger */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Upload or replace image"
          className="p-2.5 text-slate-400 hover:text-brand-blue hover:bg-brand-blue-tint/50 rounded-xl transition-colors shrink-0"
        >
          <Paperclip className="w-4 h-4" />
        </button>

        <div className="relative flex-1">
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!isImageReady || isAnalyzing}
            placeholder={
              !isImageReady
                ? 'Upload an image first, then type your question...'
                : isAnalyzing
                ? 'Analyzing the image...'
                : 'Type your question here...'
            }
            className={`w-full py-2.5 px-3.5 text-xs sm:text-sm rounded-xl border transition-all outline-hidden ${
              !isImageReady
                ? 'bg-slate-50 border-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-slate-50/70 focus:bg-white border-slate-200 focus:border-brand-blue focus:ring-2 focus:ring-brand-blue/15 text-brand-dark'
            }`}
          />
        </div>

        {/* Send Button */}
        <button
          type="submit"
          disabled={!isImageReady || isAnalyzing || !queryInput.trim()}
          className={`p-2.5 rounded-xl flex items-center justify-center transition-all shrink-0 shadow-sm ${
            !isImageReady || isAnalyzing || !queryInput.trim()
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
              : 'bg-brand-blue hover:bg-blue-600 text-white shadow-brand-blue/20 hover:scale-[1.03]'
          }`}
        >
          {isAnalyzing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </form>

      <p className="text-[11px] text-brand-muted text-center mt-2 font-medium">
        Ask anything about the image — land cover, vegetation, water bodies, built-up areas, and more.
      </p>
    </div>
  );
}
