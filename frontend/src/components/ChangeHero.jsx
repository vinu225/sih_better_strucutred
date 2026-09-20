import React from 'react';
import { ArrowLeftRight, Layers } from 'lucide-react';

export default function ChangeHero({ hasMessages = false }) {
  if (hasMessages) {
    return (
      <div className="relative overflow-hidden rounded-xl py-2 px-3 mb-4 border border-brand-border/60 bg-[#f4f8fc]">
        {/* Background image & gradient overlay for compact state */}
        <div
          className="absolute inset-0 bg-cover bg-no-repeat pointer-events-none opacity-50"
          style={{
            backgroundImage: `url('/assets/hero-bg.jpg')`,
            backgroundPosition: 'right center',
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-r from-[#f4f8fc] via-[#f4f8fc]/85 to-transparent pointer-events-none" />

        <div className="relative z-10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-brand-dark">SatQuery AI</span>
            <span className="text-xs text-brand-muted">•</span>
            <span className="text-xs text-brand-muted font-medium">Temporal Change Detection</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-brand-blue bg-blue-50/90 backdrop-blur-xs px-2.5 py-1 rounded-full font-medium border border-blue-100 shadow-2xs">
            <ArrowLeftRight className="w-3 h-3" />
            <span>Comparison Active</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-2xl bg-[#f4f8fc] bg-gradient-to-r from-blue-50/70 via-white to-emerald-50/40 border border-brand-border p-6 lg:p-7 mb-6 shadow-soft">
      {/* Background Image: right center, cover, fading into #f4f8fc toward the left */}
      <div
        className="absolute inset-0 bg-cover bg-no-repeat pointer-events-none opacity-90"
        style={{
          backgroundImage: `url('/assets/hero-bg.jpg')`,
          backgroundPosition: 'right center',
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-r from-[#f4f8fc] from-30% via-[#f4f8fc]/85 to-transparent pointer-events-none" />

      {/* Background Graphic Decor */}
      <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-gradient-to-l from-brand-blue/5 via-transparent to-transparent pointer-events-none rounded-r-2xl" />
      <div className="absolute -right-8 -top-8 w-44 h-44 rounded-full bg-gradient-to-br from-cyan-400/20 to-blue-400/20 blur-2xl pointer-events-none" />

      <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-2 max-w-2xl">
          <div className="flex items-center gap-2">
            <div className="inline-flex flex-col items-start">
              <span className="text-[11px] font-extrabold tracking-wider uppercase text-brand-blue bg-brand-blue-tint px-2.5 py-0.5 rounded-md border border-brand-blue/20">
                CHANGE DETECTION
              </span>
              <span className="w-8 h-0.5 bg-brand-blue rounded-full mt-1" />
            </div>
          </div>
          <h1 className="font-display font-normal text-4xl sm:text-5xl lg:text-[54px] tracking-[-0.01em] leading-[1.08] text-brand-dark">
            See Changes. <span className="text-brand-blue">Understand the Story.</span>
          </h1>
          <p className="text-sm sm:text-base text-brand-muted font-medium leading-relaxed">
            Compare satellite images from different time periods and uncover what's changed on Earth.
          </p>
        </div>

        {/* Right Badge / Tagline (Hidden on mobile) */}
        <div className="hidden sm:flex flex-col items-end shrink-0 gap-2">
          <div className="inline-flex flex-col items-end">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/90 border border-brand-border shadow-sm text-xs font-semibold text-brand-dark">
              <span className="w-2 h-2 rounded-full bg-brand-blue animate-ping" />
              <span>A clearer Earth a brighter tomorrow.</span>
            </div>
            <span className="w-12 h-0.5 bg-brand-blue/60 rounded-full mt-1.5 mr-2" />
          </div>
          <p className="italic text-xs text-brand-muted font-serif">
            "Temporal Geospatial Intelligence"
          </p>
        </div>
      </div>
    </div>
  );
}
