import React from 'react';
import { User, Satellite, Check, AlertCircle } from 'lucide-react';
import StructuredResponse from './StructuredResponse';
import LandCoverCard from './LandCoverCard';
import { useChat } from '../context/ChatContext';

const FOLLOW_UPS = [
  'Show NDVI for this image?',
  'Are there any water bodies?',
  'Where are the urban structures?',
];

export default function MessageBubble({ message, isLast = false }) {
  const { setQueryInput } = useChat();
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end items-end gap-2.5 mb-5">
        <div className="flex flex-col items-end max-w-[80%] sm:max-w-[70%]">
          <div className="bg-brand-blue text-white rounded-2xl rounded-br-xs px-4 py-3 text-xs sm:text-sm font-medium shadow-sm leading-relaxed">
            {message.text}
          </div>
          <div className="flex items-center gap-1 text-[11px] text-brand-muted mt-1 px-1">
            <span>{message.timestamp}</span>
            <Check className="w-3 h-3 text-brand-blue" />
          </div>
        </div>
        <div className="w-8 h-8 rounded-full bg-slate-200 border border-slate-300 flex items-center justify-center text-slate-600 shrink-0 mb-5">
          <User className="w-4 h-4" />
        </div>
      </div>
    );
  }

  // Assistant Message
  const predictions = message.toolArtifacts?.classification?.top_predictions || [];
  const isNew = Boolean(message.isNew);

  return (
    <div
      className={`flex items-start gap-3 mb-6 max-w-full ${
        isNew ? 'animate-reveal-0' : ''
      }`}
    >
      {/* Bot Avatar */}
      <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-500 text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <Satellite className="w-4 h-4" />
      </div>

      <div className="flex-1 space-y-3 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-brand-dark">SatQuery AI</span>
          <span className="text-[11px] text-brand-muted">{message.timestamp}</span>
        </div>

        {/* Bubble Container */}
        <div
          className={`rounded-2xl rounded-tl-xs p-4 border text-xs sm:text-sm leading-relaxed shadow-xs ${
            message.isError
              ? 'bg-red-50/70 border-red-200 text-red-800'
              : 'bg-white border-brand-border text-slate-800'
          }`}
        >
          {message.isError ? (
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
              <p>{message.text}</p>
            </div>
          ) : (
            <>
              {/* Structured Response: Short answer + Collapsed 'How this was computed' panel with tool states */}
              <StructuredResponse
                text={message.text}
                plan={message.plan}
                toolArtifacts={message.toolArtifacts}
                visualEvidence={message.visualEvidence}
                executionTrace={message.executionTrace}
                detectedModality={message.detectedModality || message.modality}
                isAnimated={isNew}
              />

              {/* Visual Evidence (Overlay / Heatmap / Grounding) */}
              {(message.visualEvidence?.image_base64 || message.toolArtifacts?.visual_evidence?.image_base64) && (
                <div
                  className={`mt-3.5 pt-3 border-t border-slate-100 ${
                    isNew ? 'animate-reveal-3' : ''
                  }`}
                >
                  <div className="rounded-xl overflow-hidden border border-brand-border bg-slate-900 relative group">
                    <img
                      src={message.visualEvidence?.image_base64 || message.toolArtifacts?.visual_evidence?.image_base64}
                      alt="Visual Evidence Overlay"
                      className="w-full max-h-80 object-contain bg-slate-950 mx-auto"
                    />
                    <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-2.5 text-white flex items-center justify-between text-xs">
                      <span className="font-semibold flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        AI Detection & Localization Overlay
                      </span>
                      {message.visualEvidence?.count !== undefined && (
                        <span className="bg-white/20 px-2 py-0.5 rounded text-[11px] font-mono">
                          {message.visualEvidence.count} localized
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Analyzed Thumbnail & Land Cover Breakdown (if no visual evidence overlay) */}
              {!message.visualEvidence?.image_base64 && !message.toolArtifacts?.visual_evidence?.image_base64 && (message.imagePreviewUrl || predictions.length > 0) && (
                <div
                  className={`mt-3.5 pt-3 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start ${
                    isNew ? 'animate-reveal-3' : ''
                  }`}
                >
                  {message.imagePreviewUrl && (
                    <div className="rounded-xl overflow-hidden border border-brand-border bg-slate-50 relative group">
                      <img
                        src={message.imagePreviewUrl}
                        alt="Analyzed Satellite Region"
                        className="w-full h-40 object-cover"
                      />
                      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/60 to-transparent p-2 text-white text-[10px] font-semibold">
                        Analyzed Satellite Patch
                      </div>
                    </div>
                  )}

                  {predictions.length > 0 && (
                    <div className="mt-0">
                      <LandCoverCard predictions={predictions} />
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Follow-up Suggestion Chips (Only on latest message) */}
        {isLast && !message.isError && (
          <div
            className={`flex flex-wrap items-center gap-1.5 pt-1 ${
              isNew ? 'animate-reveal-4' : ''
            }`}
          >
            {FOLLOW_UPS.map((q, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQueryInput(q)}
                className="px-3 py-1 rounded-full bg-slate-100/80 hover:bg-brand-blue-tint text-slate-600 hover:text-brand-blue text-[11px] font-medium border border-slate-200 transition-colors shadow-2xs cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
