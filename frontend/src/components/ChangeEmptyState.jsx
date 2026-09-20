import React from 'react';
import { Search, Trees, BarChart3, Satellite } from 'lucide-react';

const CHIPS = [
  { label: 'What has changed?', icon: Search, query: 'What has changed between these two images?' },
  { label: 'Show deforestation', icon: Trees, query: 'Show the areas of deforestation.' },
  { label: 'Compare urban growth', icon: BarChart3, query: 'Show urban expansion in this region.' },
];

export default function ChangeEmptyState({ onSelectQuery }) {
  return (
    <div className="h-full flex flex-col items-center justify-center p-8 text-center my-auto">
      {/* Centered Graphic */}
      <div className="relative mb-5">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-brand-blue/10 via-cyan-50 to-blue-50 border border-brand-border flex items-center justify-center text-brand-blue shadow-soft">
          <Satellite className="w-10 h-10 text-brand-blue animate-pulse" />
        </div>
        <span className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-brand-green text-white flex items-center justify-center text-[10px] font-bold shadow-sm">
          AI
        </span>
      </div>

      <h3 className="font-display font-normal text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark mb-1.5">
        Start a Conversation
      </h3>
      <p className="text-xs text-brand-muted max-w-md leading-relaxed mb-6 font-medium">
        Upload two satellite images (before and after) and ask a question to detect and analyze changes.
      </p>

      {/* Suggestion Chips */}
      <div className="flex flex-wrap items-center justify-center gap-2 max-w-md">
        {CHIPS.map((chip, idx) => {
          const Icon = chip.icon;
          return (
            <button
              key={idx}
              type="button"
              onClick={() => onSelectQuery(chip.query)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 hover:bg-brand-blue-tint text-slate-700 hover:text-brand-blue border border-brand-border text-xs font-semibold transition-all hover:scale-[1.02] shadow-2xs cursor-pointer"
            >
              <Icon className="w-3.5 h-3.5 text-brand-blue" />
              <span>{chip.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
