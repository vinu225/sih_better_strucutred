import React from 'react';
import { Satellite, Loader2, Check } from 'lucide-react';

/**
 * Shared AnalysisProgress component for displaying progress indicators with animated step chips.
 */
export default function AnalysisProgress({
  title = 'Analyzing the image...',
  steps = [
    'Image received',
    'Modality detected',
    'Selecting tools',
    'Running analysis',
    'Composing answer',
  ],
  activeStepIndex = 0,
  reducedMotion = false,
}) {
  return (
    <div className="flex items-start gap-3 mb-6">
      <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-500 text-white flex items-center justify-center shrink-0 shadow-sm animate-bounce">
        <Satellite className="w-4 h-4" />
      </div>
      <div className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-xs p-4 flex flex-col gap-2.5 text-xs text-brand-dark max-w-md shadow-2xs">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-brand-blue" />
          <span className="font-semibold text-brand-dark">{title}</span>
        </div>

        {/* Step progression chips */}
        {!reducedMotion && steps && steps.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            {steps.map((step, idx) => {
              const isDone = idx < activeStepIndex;
              const isCurrent = idx === activeStepIndex;
              return (
                <span
                  key={idx}
                  className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium border transition-all duration-200 ${
                    isDone
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : isCurrent
                      ? 'bg-brand-blue-tint text-brand-blue border-brand-blue/30 font-semibold shadow-2xs scale-[1.02]'
                      : 'bg-slate-100 text-slate-400 border-slate-200 opacity-60'
                  }`}
                >
                  {isDone && <Check className="w-3 h-3 text-emerald-600" />}
                  {isCurrent && (
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-blue animate-pulse" />
                  )}
                  <span>{step}</span>
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
