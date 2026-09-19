import React from 'react';
import { ArrowRight, Cpu, Database, HelpCircle, Layers, Server, Globe, Satellite } from 'lucide-react';

export default function AboutPage() {
  const pipelineSteps = [
    'Satellite Tile',
    'Modality Handler',
    'MobileViT Encoder',
    'Vision Projector',
    'Qwen2.5-0.5B',
    'Answer',
  ];

  return (
    <div className="max-w-4xl space-y-6 pb-10">
      {/* Header Card */}
      <div className="bg-white rounded-2xl border border-brand-border p-7 shadow-soft space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-blue-tint text-brand-blue text-xs font-bold border border-brand-blue/20">
          <Satellite className="w-3.5 h-3.5" />
          <span>SIH 2026 Prototype (SIH26167)</span>
        </div>
        <h1 className="font-display font-normal text-3xl sm:text-[40px] tracking-[-0.01em] leading-[1.1] text-brand-dark">
          About SatQuery AI
        </h1>
        <p className="text-sm sm:text-base text-slate-600 leading-relaxed font-normal">
          SatQuery AI is an Earth Observation (EO) multimodal vision-language framework built for the Smart India Hackathon 2026. It enables natural-language question answering over satellite imagery by combining multispectral optical and radar (SAR) data into a unified multimodal intelligence agent.
        </p>
      </div>

      {/* How it Works Pipeline */}
      <div className="bg-white rounded-2xl border border-brand-border p-7 shadow-soft space-y-4">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <Cpu className="w-4 h-4" />
          </div>
          <h2 className="font-display font-normal text-[24px] sm:text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark">How It Works</h2>
        </div>

        {/* Horizontal Flow Diagram */}
        <div className="overflow-x-auto py-2">
          <div className="flex items-center gap-2 min-w-[650px]">
            {pipelineSteps.map((step, idx) => (
              <React.Fragment key={idx}>
                <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-3 text-center shadow-2xs">
                  <span className="text-xs font-bold text-slate-800">{step}</span>
                </div>
                {idx < pipelineSteps.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-brand-blue shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
          When a query is submitted, the SatQuery autonomous agent inspects the user question and selects the appropriate specialized tool—such as spectral index calculations (NDVI/NDWI/NDBI), 19-class land-cover classification, SAR radar backscatter analysis, or the multimodal vision-language model (VLM)—to synthesize a grounded geospatial briefing.
        </p>
      </div>

      {/* Input Data & What you can ask Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Input Data */}
        <div className="bg-white rounded-2xl border border-brand-border p-6 shadow-soft space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-600">
              <Database className="w-4 h-4" />
            </div>
            <h2 className="font-display font-normal text-[24px] sm:text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark">Input Data</h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Standard 12-channel BigEarthNet-v2.0 format containing <strong>10 Sentinel-2 optical bands</strong> (B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A) plus <strong>2 Sentinel-1 SAR radar channels</strong> (VH, VV backscatter). Standard RGB optical photographs and single-channel SAR files are also supported.
          </p>
        </div>

        {/* What You Can Ask */}
        <div className="bg-white rounded-2xl border border-brand-border p-6 shadow-soft space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-amber-50 text-amber-600">
              <HelpCircle className="w-4 h-4" />
            </div>
            <h2 className="font-display font-normal text-[24px] sm:text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark">What You Can Ask</h2>
          </div>
          <ul className="text-xs sm:text-sm text-slate-600 space-y-1.5 list-disc list-inside">
            <li>Land cover classification (19 BigEarthNet categories)</li>
            <li>Vegetation vigor, canopy health, and NDVI indices</li>
            <li>Water body presence and hydrological NDWI coverage</li>
            <li>Built-up / urban structures and NDBI index</li>
            <li>SAR surface roughness and radar backscatter analysis</li>
          </ul>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="bg-white rounded-2xl border border-brand-border p-6 shadow-soft space-y-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-brand-blue-tint text-brand-blue">
            <Server className="w-4 h-4" />
          </div>
          <h2 className="font-display font-normal text-[24px] sm:text-[26px] tracking-[-0.01em] leading-[1.15] text-brand-dark">Technology Stack</h2>
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          {[
            'PyTorch',
            'Hugging Face Transformers',
            'MobileViT Backbone',
            'Qwen2.5-0.5B-Instruct',
            'FastAPI',
            'Vite',
            'React 18',
            'Tailwind CSS',
          ].map((tech, idx) => (
            <span
              key={idx}
              className="px-3 py-1.5 rounded-xl bg-slate-50 border border-brand-border text-xs font-semibold text-slate-700 shadow-2xs"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
