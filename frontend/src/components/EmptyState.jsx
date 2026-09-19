import React from 'react';
import { Satellite, Leaf, Droplets, LineChart } from 'lucide-react';
import { useChat } from '../context/ChatContext';

const CHIPS = [
  { label: 'Ask about land cover', icon: Leaf, query: 'What are the main land cover types in this image?' },
  { label: 'Find water bodies', icon: Droplets, query: 'Show me water bodies in this area.' },
  { label: 'Analyze vegetation', icon: LineChart, query: 'What is the vegetation coverage here?' },
];

export default function EmptyState() {
  const { setQueryInput } = useChat();

  return (
    <div className="h-full flex flex-col items-center justify-center p-8 text-center my-auto">
      {/* Centered Graphic */}
      <div className="relative mb-5">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-brand-blue/10 via-cyan-50 to-emerald-50 border border-brand-border flex items-center justify-center text-brand-blue shadow-soft">
          <Satellite className="w-10 h-10 text-brand-blue animate-pulse" />
        </div>
        <span className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-brand-green text-white flex items-center justify-center text-[10px] font-bold shadow-sm">
          AI
        </span>
      </div>

      <h3 className="font-display font-normal text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark mb-1.5">Start a Conversation</h3>
      <p className="text-xs text-brand-muted max-w-sm leading-relaxed mb-6 font-medium">
        Upload a satellite image and ask a question to explore real-time geospatial insights about our planet.
      </p>

      {/* Suggestion Chips */}
      <div className="flex flex-wrap items-center justify-center gap-2 max-w-md">
        {CHIPS.map((chip, idx) => {
          const Icon = chip.icon;
          return (
            <button
              key={idx}
              type="button"
              onClick={() => setQueryInput(chip.query)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 hover:bg-brand-blue-tint text-slate-700 hover:text-brand-blue border border-brand-border text-xs font-semibold transition-all hover:scale-[1.02] shadow-2xs"
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
