import React from 'react';
import { Lightbulb, ChevronRight } from 'lucide-react';
import { useChat } from '../context/ChatContext';

const EXAMPLES = [
  'What is visible in this image?',
  'Where are the buildings?',
  'What is the vegetation coverage here?',
  'Show me water bodies in this area.',
];

export default function ExampleQueries() {
  const { setQueryInput } = useChat();

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft">
      <div className="flex items-center justify-between mb-3.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-50 text-amber-600">
            <Lightbulb className="w-4 h-4" />
          </div>
          <h2 className="font-display font-normal text-[22px] sm:text-[24px] tracking-[-0.01em] leading-[1.15] text-brand-dark">Example Queries</h2>
        </div>
      </div>

      <div className="space-y-2">
        {EXAMPLES.map((query, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setQueryInput(query)}
            className="w-full text-left p-2.5 rounded-xl border border-slate-100 hover:border-brand-blue/30 hover:bg-brand-blue-tint/40 text-xs font-medium text-slate-700 hover:text-brand-blue transition-all flex items-center justify-between group"
          >
            <span className="truncate pr-2">{query}</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-brand-blue group-hover:translate-x-0.5 transition-all shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}
