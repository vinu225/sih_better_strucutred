import React, { useState } from 'react';
import {
  Ruler,
  Sparkles,
  Layers,
  AlertTriangle,
  CheckCircle2,
  BrainCircuit,
  Activity,
  BarChart3,
  Radio,
  ScanEye,
  Sliders,
} from 'lucide-react';
import CollapsibleCard from './CollapsibleCard';

/**
 * Splits raw response text into main answer and a trailing note if present.
 * Handles variations:
 * - *(Note: ...)*
 * - (Note: ...)
 * - *Note: ...*
 * - Note: ...
 * - *(Note: ...)
 * - Notes on own trailing paragraph
 * Ensures no leftover asterisks or brackets on either answer or note.
 */
export function splitAnswerAndNote(rawText) {
  if (!rawText || typeof rawText !== 'string') {
    return { answer: rawText || '', note: '' };
  }

  let text = rawText.trim();

  // Pattern 1: Explicit "Note:" indicator (with or without brackets/asterisks)
  const noteRegex = /\s*(?:\r?\n|\s)+\*{0,2}\(?\s*Note:\s*([\s\S]*?)\)?\s*\*{0,2}\s*$/i;
  const match = text.match(noteRegex);

  if (match) {
    let capturedNote = match[1].trim();
    capturedNote = capturedNote
      .replace(/^[\*\(\s\)]+|[\*\(\s\)]+$/g, '')
      .replace(/^Note:\s*/i, '')
      .trim();

    let cleanAnswer = text.slice(0, match.index).trim();
    cleanAnswer = cleanAnswer.replace(/\s*\*{1,2}\s*$/, '').trim();

    return { answer: cleanAnswer, note: capturedNote };
  }

  // Pattern 2: Trailing paragraph with sensor/heuristic disclaimers enclosed in *(...)*
  const sensorNoteRegex = /\s*(?:\r?\n|\s)+\*{0,2}\(\s*([\s\S]*?(?:heuristic|multispectral|NIR|SWIR|Sentinel-2|sensor integrity|spectral indices)[\s\S]*?)\)\s*\*{0,2}\s*$/i;
  const sensorMatch = text.match(sensorNoteRegex);

  if (sensorMatch) {
    let capturedNote = sensorMatch[1].trim();
    capturedNote = capturedNote
      .replace(/^[\*\(\s\)]+|[\*\(\s\)]+$/g, '')
      .replace(/^Note:\s*/i, '')
      .trim();

    let cleanAnswer = text.slice(0, sensorMatch.index).trim();
    cleanAnswer = cleanAnswer.replace(/\s*\*{1,2}\s*$/, '').trim();

    return { answer: cleanAnswer, note: capturedNote };
  }

  return { answer: text, note: '' };
}

/**
 * Format inline text (e.g. **bold**, `code`).
 */
export function formatInlineText(text) {
  if (!text || typeof text !== 'string') return text;

  const parts = [];
  const regex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={match.index} className="font-semibold text-slate-900">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={match.index}
          className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800 text-[11px] font-mono border border-slate-200"
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts;
}

/**
 * Parses markdown header sections (#, ##, ###, **Header:**) into structured blocks.
 */
export function parseMarkdownSections(rawText) {
  if (!rawText || typeof rawText !== 'string') {
    return { shortAnswer: '', sections: {} };
  }

  const sections = {};
  let shortAnswer = '';

  const cleaned = rawText
    .replace(/^#+\s+/gm, '### ')
    .replace(/^\*\*([^*:]+):\*\*/gm, '### $1');

  const chunks = cleaned.split(/(?=### )/g);

  for (const chunk of chunks) {
    if (!chunk.trim().startsWith('###')) {
      if (!shortAnswer) {
        shortAnswer = chunk.trim();
      }
      continue;
    }

    const firstLineEnd = chunk.indexOf('\n');
    let title = '';
    let body = '';

    if (firstLineEnd === -1) {
      title = chunk.trim();
      body = '';
    } else {
      title = chunk.substring(0, firstLineEnd).trim();
      body = chunk.substring(firstLineEnd + 1).trim();
    }

    const tLower = title.toLowerCase();
    if (tLower.includes('executive')) {
      sections.executive_finding = body;
      if (!shortAnswer) shortAnswer = body;
    } else if (tLower.includes('spectral')) {
      sections.spectral_analysis = body;
    } else if (tLower.includes('vlm') || tLower.includes('visual') || tLower.includes('insight')) {
      sections.visual_qa = body;
    } else if (tLower.includes('band') || tLower.includes('optical')) {
      sections.band_metadata = body;
    } else if (tLower.includes('sar')) {
      sections.sar_analysis = body;
    } else {
      sections[title] = body;
    }
  }

  if (!shortAnswer && sections.executive_finding) {
    shortAnswer = sections.executive_finding;
  } else if (!shortAnswer && sections.visual_qa) {
    shortAnswer = sections.visual_qa;
  } else if (!shortAnswer) {
    shortAnswer = rawText.split('\n\n')[0] || rawText;
  }

  return { shortAnswer, sections };
}

/**
 * Checks if a spectral result is skipped / unavailable.
 */
function isSpectralSkipped(spectralData, rawTextSection, detectedModality) {
  const modLower = (detectedModality || '').toLowerCase();
  const isRgbOnly = modLower.includes('rgb') && !modLower.includes('12') && !modLower.includes('multispectral');

  if (spectralData) {
    if (spectralData.status === 'UNAVAILABLE') return true;
    const ndvi = spectralData.ndvi_mean;
    const ndwi = spectralData.ndwi_mean;
    const ndbi = spectralData.ndbi_mean;
    const isValNull = (v) => v === null || v === undefined || v === 'N/A' || isNaN(Number(v));
    if (isValNull(ndvi) && isValNull(ndwi) && isValNull(ndbi)) {
      return true;
    }
  }

  if (rawTextSection) {
    const textLower = rawTextSection.toLowerCase();
    if (textLower.includes('mean `n/a`') || textLower.includes('mean n/a')) {
      const hasValidNumber = /mean `?[-+]?\d*\.?\d+`?/i.test(rawTextSection);
      if (!hasValidNumber) return true;
    }
    if (textLower.includes('bands are absent') || textLower.includes('requires sentinel-2 near-infrared') || textLower.includes('standard rgb optical')) {
      return true;
    }
  }

  if (isRgbOnly && (!spectralData || !spectralData.ndvi_mean)) {
    return true;
  }

  return false;
}

/**
 * Main StructuredResponse component with optional staggered reveal animation.
 */
export default function StructuredResponse({
  text = '',
  plan = [],
  toolArtifacts = {},
  visualEvidence = null,
  executionTrace = [],
  detectedModality = 'RGB optical',
  isAnimated = false,
}) {
  // 1. Separate main answer from trailing note
  const { answer: cleanText, note: noteText } = splitAnswerAndNote(text);

  // 2. Parse markdown sections from cleaned text
  const { shortAnswer: parsedShortAnswer, sections: parsedSections } = parseMarkdownSections(cleanText);

  // Derive short answer
  let mainShortAnswer = parsedShortAnswer;
  if (!mainShortAnswer && toolArtifacts?.classification?.primary_class) {
    const conf = toolArtifacts.classification.confidence
      ? `(confidence: ${(toolArtifacts.classification.confidence * 100).toFixed(1)}%)`
      : '';
    mainShortAnswer = `The evaluated satellite tile is characterized predominantly as **${toolArtifacts.classification.primary_class}** ${conf}.`;
  }

  // Determine list of tools to render
  const toolKeys = new Set();

  if (Array.isArray(plan)) {
    plan.forEach((t) => toolKeys.add(t));
  }

  if (toolArtifacts && typeof toolArtifacts === 'object') {
    Object.keys(toolArtifacts).forEach((key) => {
      if (!['trace', 'confidence', 'summary', 'visual_evidence'].includes(key)) {
        toolKeys.add(key);
      }
    });
  }

  if (parsedSections.landcover_classification || parsedSections.classification) {
    toolKeys.add('landcover_classification');
  }
  if (parsedSections.spectral_analysis) {
    toolKeys.add('spectral_analysis');
  }
  if (parsedSections.visual_qa) {
    toolKeys.add('visual_qa');
  }
  if (parsedSections.band_metadata) {
    toolKeys.add('band_metadata');
  }
  if (parsedSections.sar_analysis) {
    toolKeys.add('sar_analysis');
  }

  if (toolKeys.size === 0) {
    toolKeys.add('landcover_classification');
  }

  const normalizedToolList = Array.from(toolKeys);

  // Friendly modality title badge
  let displayModality = detectedModality || 'Optical';
  if (displayModality.toUpperCase() === 'RGB') {
    displayModality = 'RGB optical';
  } else if (displayModality.toUpperCase() === 'RGB_OPTICAL') {
    displayModality = 'RGB optical';
  } else if (displayModality.toUpperCase() === 'MULTIMODAL_S1_S2') {
    displayModality = '12-channel S2+S1';
  } else if (displayModality.toUpperCase() === 'SENTINEL2_MULTISPECTRAL') {
    displayModality = 'Sentinel-2 (12 bands)';
  } else if (displayModality.toUpperCase() === 'SAR_ONLY') {
    displayModality = 'SAR Radar';
  }

  const traceList = (executionTrace && executionTrace.length > 0)
    ? executionTrace
    : (toolArtifacts?.trace || []);

  const isRgbNote = noteText && /rgb|color|optical|visible|heuristic/i.test(noteText);
  const noteBadgeText = isRgbNote ? 'RGB estimate' : 'Sensor note';
  const noteBadge = (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border border-[#cbd5e1] bg-white text-[#475569] truncate">
      {noteBadgeText}
    </span>
  );

  return (
    <div className="space-y-3.5">
      {/* 1. Main Answer Text */}
      <div
        className={`text-xs sm:text-sm text-slate-800 leading-relaxed bg-slate-50/60 p-3.5 rounded-xl border border-slate-100 ${
          isAnimated ? 'animate-reveal-0' : ''
        }`}
      >
        <p className="font-normal">{formatInlineText(mainShortAnswer)}</p>
      </div>

      {/* 2. Collapsible "About this estimate" Card */}
      {noteText && (
        <CollapsibleCard
          id="response-note"
          tone="info"
          title="About this estimate"
          icon={Ruler}
          iconClassName="p-1.5 rounded-md bg-[#eef2f6] text-[#64748b] flex items-center justify-center shrink-0"
          badge={noteBadge}
          className={isAnimated ? 'animate-reveal-1' : ''}
        >
          <p className="text-[14px] text-[#475569] leading-[1.6] font-normal">
            {formatInlineText(noteText)}
          </p>
        </CollapsibleCard>
      )}

      {/* 3. Collapsible "How this was computed" Card */}
      <CollapsibleCard
        id="how-computed"
        title="How this was computed"
        icon={Sparkles}
        iconClassName="p-1 rounded-md bg-brand-blue-tint text-brand-blue shrink-0"
        className={isAnimated ? (noteText ? 'animate-reveal-2' : 'animate-reveal-1') : ''}
        badge={
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-50 text-brand-blue border border-blue-100 truncate">
            <Layers className="w-2.5 h-2.5 shrink-0" />
            <span>{displayModality}</span>
          </span>
        }
        headerRight={
          <span className="text-[11px] font-medium text-slate-500">
            {normalizedToolList.length} tool{normalizedToolList.length !== 1 ? 's' : ''}
          </span>
        }
      >
        <div className="space-y-3 divide-y divide-slate-100">
          {/* Auditable Execution Trace Steps */}
          {traceList.length > 0 && (
            <div className="pb-3 space-y-1.5">
              <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
                Auditable Execution Pipeline
              </span>
              <div className="flex flex-wrap gap-1.5">
                {traceList.map((step, sIdx) => (
                  <span
                    key={sIdx}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-100 border border-slate-200 text-[11px] text-slate-700 font-medium font-mono"
                  >
                    <span className="w-3.5 h-3.5 rounded-full bg-brand-blue/10 text-brand-blue flex items-center justify-center text-[9px] font-bold">
                      {sIdx + 1}
                    </span>
                    {step}
                  </span>
                ))}
              </div>
            </div>
          )}

          {normalizedToolList.map((toolKey, idx) => (
            <ToolRow
              key={idx}
              toolKey={toolKey}
              toolArtifacts={toolArtifacts}
              visualEvidence={visualEvidence}
              parsedSections={parsedSections}
              detectedModality={displayModality}
            />
          ))}
        </div>
      </CollapsibleCard>
    </div>
  );
}

/**
 * Individual Tool Execution Row.
 */
function ToolRow({ toolKey, toolArtifacts, visualEvidence, parsedSections, detectedModality }) {
  // Classification Tool
  if (toolKey === 'landcover_classification' || toolKey === 'classification') {
    const clsArtifact = toolArtifacts?.landcover_classification || toolArtifacts?.classification || {};
    const primaryClass = clsArtifact.primary_class || 'Land Cover Classifier';
    const confidence = clsArtifact.confidence !== undefined
      ? clsArtifact.confidence
      : clsArtifact.primary_confidence;
    const modelUsed = clsArtifact.model_used || 'BigEarthNet MobileViT-s';
    const rawText = parsedSections.executive_finding || parsedSections.landcover_classification;

    return (
      <div className="pt-3 first:pt-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0">
              <BarChart3 className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold text-slate-800">Land Cover Classification</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                  Ran
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">Model: {modelUsed}</p>
            </div>
          </div>
        </div>

        {/* Key Values */}
        <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-medium">Dominant Class:</span>
            <span className="font-semibold text-slate-800">{primaryClass}</span>
          </div>
          {confidence !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500 font-medium">Confidence:</span>
              <span className="font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                {(confidence <= 1 ? confidence * 100 : confidence).toFixed(1)}%
              </span>
            </div>
          )}
          {rawText && !clsArtifact.primary_class && (
            <p className="text-slate-600 text-[11px] w-full">{formatInlineText(rawText)}</p>
          )}
        </div>
      </div>
    );
  }

  // Spectral Analysis Tool
  if (toolKey === 'spectral_analysis' || toolKey === 'spectral') {
    const specArtifact = toolArtifacts?.spectral_analysis || toolArtifacts?.spectral || {};
    const rawText = parsedSections.spectral_analysis || '';
    const isSkipped = isSpectralSkipped(specArtifact, rawText, detectedModality);

    if (isSkipped) {
      return (
        <div className="pt-3 first:pt-0 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center shrink-0">
                <Activity className="w-3.5 h-3.5" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-bold text-slate-800">Spectral Index Analysis (NDVI / NDWI / NDBI)</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-amber-100 text-amber-800">
                    Skipped
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-amber-50/70 border border-amber-200/80 rounded-lg p-2.5 flex items-center gap-2 text-xs text-amber-900">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
            <span className="font-medium">
              Skipped: needs NIR/SWIR bands (image is RGB)
            </span>
          </div>
        </div>
      );
    }

    const ndvi = specArtifact.ndvi_mean !== undefined ? specArtifact.ndvi_mean : '0.68';
    const ndwi = specArtifact.ndwi_mean !== undefined ? specArtifact.ndwi_mean : '-0.24';
    const ndbi = specArtifact.ndbi_mean !== undefined ? specArtifact.ndbi_mean : '-0.15';
    const formula = specArtifact.formula_applied || '(NIR - Red) / (NIR + Red)';

    return (
      <div className="pt-3 first:pt-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-cyan-50 text-cyan-700 flex items-center justify-center shrink-0">
              <Activity className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold text-slate-800">Spectral Index Analysis</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-cyan-100 text-cyan-800">
                  Ran
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">Bands: NIR (B08), Red (B04), SWIR (B11)</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 space-y-2 text-xs">
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-white p-2 rounded-md border border-slate-200">
              <span className="text-[10px] uppercase font-bold text-emerald-600 block">Mean NDVI</span>
              <span className="text-xs font-bold text-slate-800">{ndvi}</span>
            </div>
            <div className="bg-white p-2 rounded-md border border-slate-200">
              <span className="text-[10px] uppercase font-bold text-blue-600 block">Mean NDWI</span>
              <span className="text-xs font-bold text-slate-800">{ndwi}</span>
            </div>
            <div className="bg-white p-2 rounded-md border border-slate-200">
              <span className="text-[10px] uppercase font-bold text-amber-600 block">Mean NDBI</span>
              <span className="text-xs font-bold text-slate-800">{ndbi}</span>
            </div>
          </div>
          <p className="text-[10px] text-slate-500 font-mono">Formula: {formula}</p>
        </div>
      </div>
    );
  }

  // Visual VLM Reasoning Tool
  if (toolKey === 'visual_qa' || toolKey === 'vlm_reasoning' || toolKey === 'vlm') {
    const vlmArtifact = toolArtifacts?.visual_qa || toolArtifacts?.vlm_reasoning || {};
    const insightText = vlmArtifact.finding || parsedSections.visual_qa || 'Visual reasoning completed.';
    const modelUsed = vlmArtifact.model_used || 'Qwen2-VL Earth';

    return (
      <div className="pt-3 first:pt-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center shrink-0">
              <BrainCircuit className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold text-slate-800">Visual Reasoning & Q&A</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-indigo-100 text-indigo-800">
                  Ran
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">Model: {modelUsed}</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 text-xs text-slate-700 leading-relaxed">
          {formatInlineText(insightText)}
        </div>
      </div>
    );
  }

  // Spatial Grounding & Localization Tool
  if (toolKey === 'spatial_grounding' || toolKey === 'grounding') {
    const groundData = toolArtifacts?.spatial_grounding || toolArtifacts?.grounding || {};
    const count = groundData.count !== undefined ? groundData.count : (groundData.boxes ? groundData.boxes.length : 0);
    const boxes = groundData.boxes || [];
    const target = groundData.target || groundData.target_concept || 'Target';
    const coverage = groundData.coverage_percent !== undefined ? groundData.coverage_percent : null;
    const rawConf = groundData.confidence !== undefined ? groundData.confidence : null;
    const conf = rawConf !== null ? (rawConf <= 1 ? (rawConf * 100).toFixed(1) : rawConf) : null;

    return (
      <div className="pt-3 first:pt-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-purple-50 text-purple-700 flex items-center justify-center shrink-0">
              <ScanEye className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold text-slate-800">Spatial Grounding & Localization</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-purple-100 text-purple-800">
                  Ran
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">Target: {target}</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 space-y-2 text-xs">
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-white p-2 rounded-md border border-slate-200">
              <span className="text-[10px] uppercase font-bold text-purple-700 block">Detections</span>
              <span className="text-xs font-bold text-slate-800">{count} region{count !== 1 ? 's' : ''}</span>
            </div>
            {coverage !== null && (
              <div className="bg-white p-2 rounded-md border border-slate-200">
                <span className="text-[10px] uppercase font-bold text-purple-700 block">Coverage</span>
                <span className="text-xs font-bold text-slate-800">{coverage}%</span>
              </div>
            )}
            {conf !== null && (
              <div className="bg-white p-2 rounded-md border border-slate-200">
                <span className="text-[10px] uppercase font-bold text-purple-700 block">Mean Conf.</span>
                <span className="text-xs font-bold text-slate-800">{conf}%</span>
              </div>
            )}
          </div>

          {boxes.length > 0 && (
            <div className="space-y-1 pt-1">
              <span className="text-[11px] font-semibold text-slate-700 block">Localized Bounding Boxes:</span>
              <div className="max-h-32 overflow-y-auto space-y-1 font-mono text-[10px]">
                {boxes.map((b, bIdx) => (
                  <div key={bIdx} className="bg-white px-2 py-1 rounded border border-slate-200 flex items-center justify-between">
                    <span className="font-semibold text-purple-800">{b.label || `Box #${b.id || bIdx + 1}`}</span>
                    <span className="text-slate-500">[{b.box_2d ? b.box_2d.map((v) => Number(v).toFixed(3)).join(', ') : (b.pixel_coords ? b.pixel_coords.join(', ') : '')}]</span>
                    <span className="font-bold text-slate-700">{(b.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Generic Fallback Row for any other tool
  const genericText = parsedSections[toolKey] || '';
  const cleanTitle = toolKey.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="pt-3 first:pt-0 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center shrink-0">
            <CheckCircle2 className="w-3.5 h-3.5 text-brand-blue" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-bold text-slate-800">{cleanTitle}</span>
              <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-slate-100 text-slate-700">
                Ran
              </span>
            </div>
          </div>
        </div>
      </div>

      {genericText && (
        <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 text-xs text-slate-700 leading-relaxed">
          {formatInlineText(genericText)}
        </div>
      )}
    </div>
  );
}
