import React, { useState } from 'react';
import { Lightbulb, ChevronRight, ArrowUpRight } from 'lucide-react';

const CHANGE_QUERIES = [
  'What has changed between these two images?',
  'Show the areas of deforestation.',
  'Compare the water body changes.',
  'Highlight new buildings or construction.',
  'What changed in vegetation over time?',
  'Show urban expansion in this region.',
];

const ALL_CHANGE_QUERIES = [
  'What has changed between these two images?',
  'Show the areas of deforestation.',
  'Compare the water body changes.',
  'Highlight new buildings or construction.',
  'What changed in vegetation over time?',
  'Show urban expansion in this region.',
  'Highlight newly paved roads or infrastructure.',
  'How much agricultural land was converted?',
  'Detect changes in water body surface area.',
];

export default function ChangeExampleQueries({ onSelectQuery }) {
  const [showAll, setShowAll] = useState(false);
  const queriesToDisplay = showAll ? ALL_CHANGE_QUERIES : CHANGE_QUERIES;

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft">
      <div className="flex items-center justify-between mb-3.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-50 text-amber-600">
            <Lightbulb className="w-4 h-4" />
          </div>
          <h2 className="font-display font-normal text-[22px] sm:text-[24px] tracking-[-0.01em] leading-[1.15] text-brand-dark">
            Example Queries
          </h2>
        </div>
        <button
          type="button"
          onClick={() => setShowAll((prev) => !prev)}
          className="text-xs font-semibold text-brand-blue hover:text-blue-700 flex items-center gap-1 group cursor-pointer transition-colors"
        >
          <span>{showAll ? 'Show less' : 'See all'}</span>
          <ArrowUpRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </button>
      </div>

      <div className="space-y-2">
        {queriesToDisplay.map((query, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectQuery(query)}
            className="w-full text-left p-2.5 rounded-xl border border-slate-100 hover:border-brand-blue/30 hover:bg-brand-blue-tint/40 text-xs font-medium text-slate-700 hover:text-brand-blue transition-all flex items-center justify-between group cursor-pointer"
          >
            <span className="truncate pr-2">{query}</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-brand-blue group-hover:translate-x-0.5 transition-all shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}
