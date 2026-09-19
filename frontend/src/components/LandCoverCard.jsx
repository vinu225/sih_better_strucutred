import React from 'react';

const CLASS_COLORS = {
  vegetation: 'bg-emerald-500',
  forest: 'bg-emerald-600',
  tree: 'bg-emerald-600',
  'agricultural land': 'bg-amber-400',
  agriculture: 'bg-amber-400',
  crop: 'bg-amber-400',
  arable: 'bg-amber-400',
  'urban / built-up': 'bg-orange-500',
  urban: 'bg-orange-500',
  builtup: 'bg-orange-500',
  building: 'bg-orange-500',
  'water bodies': 'bg-blue-500',
  water: 'bg-blue-500',
  wetland: 'bg-cyan-500',
  others: 'bg-slate-400',
  other: 'bg-slate-400',
};

function getDotColor(className) {
  const lower = className.toLowerCase();
  for (const [key, color] of Object.entries(CLASS_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return 'bg-slate-400';
}

export default function LandCoverCard({ predictions = [] }) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-brand-border p-3.5 mt-3 shadow-xs">
      <h4 className="text-xs font-bold text-brand-dark mb-2.5 flex items-center justify-between">
        <span>Estimated Land Cover</span>
        <span className="text-[10px] text-brand-muted font-normal">BigEarthNet-19</span>
      </h4>

      <div className="space-y-2">
        {predictions.map((item, idx) => {
          const name = item.class_name || item.name || 'Unknown';
          const conf = item.confidence !== undefined ? item.confidence : item.percent / 100 || 0;
          const percentStr = (conf <= 1.0 ? conf * 100 : conf).toFixed(1) + '%';
          const dotColor = getDotColor(name);

          return (
            <div key={idx} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${dotColor} shrink-0`} />
                <span className="font-medium text-slate-700 truncate max-w-[150px]">{name}</span>
              </div>
              <span className="font-semibold text-brand-dark tabular-nums">{percentStr}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
