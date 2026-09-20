import React, { useState, useEffect, useRef } from 'react';
import {
  Satellite,
  User,
  Check,
  CheckCheck,
  Send,
  Paperclip,
  Loader2,
  Wrench,
  BarChart2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import ChangeEmptyState from './ChangeEmptyState';
import ImageCompareSlider from './ImageCompareSlider';
import { buildDemoAnswer } from '../demo/changeDemo';

const PIPELINE_STEPS = [
  'Images loaded',
  'Images aligned',
  'Differences computed',
  'Summary ready',
];

export default function ChangeChatWindow({
  beforeImage,
  afterImage,
  beforeDate,
  afterDate,
  sceneResult,
  isMatched,
  queryInput,
  setQueryInput,
  messages,
  setMessages,
}) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [expandedAnalysisId, setExpandedAnalysisId] = useState(null);
  const scrollRef = useRef(null);
  const bottomAnchorRef = useRef(null);
  const inputRef = useRef(null);

  const canSend = Boolean(beforeImage && afterImage);

  // Auto-scroll to newest message smoothly
  useEffect(() => {
    if (bottomAnchorRef.current) {
      bottomAnchorRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [messages, isAnalyzing, activeStepIndex, expandedAnalysisId]);

  const handleSend = (textToSend = null) => {
    const text = (textToSend || queryInput || '').trim();
    if (!text || !canSend || isAnalyzing) return;

    const userTimestamp = new Date().toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      text,
      timestamp: userTimestamp,
    };

    const isFirstInComparison =
      messages.length === 0 ||
      messages[messages.length - 1]?.type === 'comparison_divider' ||
      !messages.some((m) => m.role === 'assistant' && m.hasSlider);

    setMessages((prev) => [...prev, userMessage]);
    setQueryInput('');
    setIsAnalyzing(true);
    setActiveStepIndex(0);

    // Respect prefers-reduced-motion
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const stepInterval = prefersReducedMotion ? 50 : 400;
    const totalAnalysisTime = prefersReducedMotion ? 200 : 1800;

    const timer = setInterval(() => {
      setActiveStepIndex((prev) => {
        if (prev < PIPELINE_STEPS.length - 1) return prev + 1;
        return prev;
      });
    }, stepInterval);

    setTimeout(() => {
      clearInterval(timer);
      setIsAnalyzing(false);

      const botTimestamp = new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      });

      const isFollowUp = !isFirstInComparison;
      const responseData = buildDemoAnswer({
        query: text,
        sceneResult,
        isMatched,
        isFollowUp,
      });

      const botMessage = {
        id: `bot-${Date.now()}`,
        role: 'assistant',
        text: responseData.text,
        timestamp: botTimestamp,
        hasSlider: isFirstInComparison,
        beforeUrl: beforeImage?.url,
        afterUrl: afterImage?.url,
        diffUrl: responseData.diffUrl,
        initialView: responseData.shouldShowDiffMap ? 'diff' : 'split',
        beforeDate: beforeDate?.trim() || null,
        afterDate: afterDate?.trim() || null,
        location: responseData.location,
        illustrative: responseData.illustrative,
        toolsUsed: responseData.toolsUsed,
        stats: responseData.stats,
        followUps: responseData.followUps,
        isMatched,
      };

      setMessages((prev) => [...prev, botMessage]);
    }, totalAnalysisTime);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const placeholderText = !canSend
    ? 'Upload both images to start'
    : messages.length === 0
    ? 'Ask a question about the changes between these images...'
    : 'Ask a follow-up question...';

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft flex flex-col h-[700px] min-h-[500px]">
      {/* Scrollable Message Area (only scrollable zone) */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto pr-1.5 pb-6 space-y-4"
      >
        {messages.length === 0 ? (
          <ChangeEmptyState onSelectQuery={(q) => setQueryInput(q)} />
        ) : (
          messages.map((msg, idx) => {
            if (msg.type === 'comparison_divider') {
              return (
                <div
                  key={msg.id || idx}
                  className="flex items-center gap-3 my-4 py-1 justify-center"
                >
                  <span className="h-[1px] bg-slate-200 flex-1" />
                  <span className="text-[11px] font-semibold text-slate-500 bg-slate-100 px-3 py-1 rounded-full border border-slate-200">
                    {msg.text}
                  </span>
                  <span className="h-[1px] bg-slate-200 flex-1" />
                </div>
              );
            }

            const isUser = msg.role === 'user';
            const isLast = idx === messages.length - 1;

            if (isUser) {
              return (
                <div
                  key={msg.id || idx}
                  className="flex justify-end items-end gap-2.5 mb-4"
                >
                  <div className="flex flex-col items-end max-w-[85%] sm:max-w-[75%]">
                    <div className="bg-[#eef5fc] text-brand-dark border border-brand-blue/20 rounded-2xl rounded-br-xs px-4 py-3 text-xs sm:text-sm font-medium shadow-2xs leading-relaxed">
                      {msg.text}
                    </div>
                    <div className="flex items-center gap-1 text-[11px] text-brand-muted mt-1 px-1">
                      <span>{msg.timestamp}</span>
                      <CheckCheck className="w-3.5 h-3.5 text-brand-blue" />
                    </div>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-brand-blue text-white flex items-center justify-center shrink-0 mb-4 shadow-sm">
                    <User className="w-4 h-4" />
                  </div>
                </div>
              );
            }

            // Assistant message
            return (
              <div
                key={msg.id || idx}
                className="flex items-start gap-3 mb-6 max-w-full"
              >
                {/* Bot Avatar */}
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-500 text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
                  <Satellite className="w-4 h-4" />
                </div>

                <div className="flex-1 space-y-3 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-brand-dark">
                      SatQuery AI
                    </span>
                    <span className="text-[11px] text-brand-muted">
                      {msg.timestamp}
                    </span>
                  </div>

                  {/* Bubble Container */}
                  <div className="rounded-2xl rounded-tl-xs p-4 border border-brand-border bg-white text-slate-800 text-xs sm:text-sm leading-relaxed shadow-2xs space-y-3.5">
                    {/* 1. Answer text */}
                    <p className="text-slate-800 leading-relaxed font-normal">
                      {msg.text}
                    </p>

                    {/* 2. Image Compare Slider (First message only) */}
                    {msg.hasSlider && msg.beforeUrl && msg.afterUrl && (
                      <div className="pt-1">
                        <ImageCompareSlider
                          beforeUrl={msg.beforeUrl}
                          afterUrl={msg.afterUrl}
                          diffUrl={msg.diffUrl}
                          initialView={msg.initialView || 'split'}
                          beforeDate={msg.beforeDate}
                          afterDate={msg.afterDate}
                          location={msg.location}
                          illustrative={msg.illustrative}
                        />
                      </div>
                    )}

                    {/* 3. Tools used & Detailed Analysis Bar */}
                    {(msg.toolsUsed?.length > 0 || msg.stats) && (
                      <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-xs">
                        {/* Tools Used Chips */}
                        {msg.toolsUsed && msg.toolsUsed.length > 0 && (
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[11px] text-brand-muted font-medium flex items-center gap-1">
                              <Wrench className="w-3 h-3 text-slate-400" />
                              Tools used:
                            </span>
                            {msg.toolsUsed.map((tool, tIdx) => (
                              <span
                                key={tIdx}
                                className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[11px] font-medium border border-slate-200/80"
                              >
                                {tool}
                              </span>
                            ))}
                          </div>
                        )}

                        {/* Detailed Analysis Collapsible Trigger (If stats present) */}
                        {msg.stats && (
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedAnalysisId(
                                expandedAnalysisId === msg.id ? null : msg.id
                              )
                            }
                            className="inline-flex items-center gap-1.5 text-xs font-semibold text-brand-blue hover:text-blue-700 bg-brand-blue-tint hover:bg-blue-100 px-2.5 py-1 rounded-lg transition-colors cursor-pointer"
                          >
                            <BarChart2 className="w-3.5 h-3.5" />
                            <span>Show Detailed Analysis</span>
                            {expandedAnalysisId === msg.id ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            )}
                          </button>
                        )}
                      </div>
                    )}

                    {/* Detailed Analysis Expanded Content (Numbers & Category Bars) */}
                    {expandedAnalysisId === msg.id && msg.stats && (
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-3 mt-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-slate-800 flex items-center gap-1.5">
                            <BarChart2 className="w-4 h-4 text-brand-blue" />
                            <span>Quantitative Change Metrics</span>
                          </h4>
                          {msg.illustrative && (
                            <span className="text-[10px] bg-slate-200 text-slate-700 px-2 py-0.5 rounded-full font-medium">
                              Computed from sample pixels
                            </span>
                          )}
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-2xs">
                            <p className="text-[10px] text-slate-500 font-medium uppercase">
                              Total Changed
                            </p>
                            <p className="text-sm font-bold text-brand-dark mt-0.5">
                              {msg.stats.changed_percent}%
                            </p>
                          </div>
                          <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-2xs">
                            <p className="text-[10px] text-rose-600 font-medium uppercase">
                              Vegetation Loss
                            </p>
                            <p className="text-sm font-bold text-rose-700 mt-0.5">
                              -{msg.stats.decrease_percent}%
                            </p>
                          </div>
                          <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-2xs">
                            <p className="text-[10px] text-emerald-600 font-medium uppercase">
                              Built-up Increase
                            </p>
                            <p className="text-sm font-bold text-emerald-700 mt-0.5">
                              +{msg.stats.increase_percent}%
                            </p>
                          </div>
                        </div>

                        {/* Category Breakdown Progress Bars */}
                        {msg.stats.categories && msg.stats.categories.length > 0 && (
                          <div className="space-y-2 pt-1">
                            <p className="text-[11px] font-semibold text-slate-700">
                              Class Breakdown:
                            </p>
                            {msg.stats.categories.map((cat, cIdx) => (
                              <div key={cIdx} className="space-y-1">
                                <div className="flex justify-between text-[11px] text-slate-600">
                                  <span>{cat.label}</span>
                                  <span className="font-semibold text-slate-800">
                                    {cat.percent}%
                                  </span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${
                                      cat.label.toLowerCase().includes('vegetation')
                                        ? 'bg-rose-500'
                                        : cat.label.toLowerCase().includes('built')
                                        ? 'bg-brand-blue'
                                        : cat.label.toLowerCase().includes('water')
                                        ? 'bg-cyan-500'
                                        : 'bg-amber-500'
                                    }`}
                                    style={{
                                      width: `${Math.min(100, Math.max(0, cat.percent * 4))}%`,
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Follow-up Suggestion Chips (Only on latest message when matched) */}
                  {isLast && msg.followUps && msg.followUps.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 pt-1">
                      {msg.followUps.map((q, qIdx) => (
                        <button
                          key={qIdx}
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
          })
        )}

        {/* Loading Indicator with Animated Step Chips */}
        {isAnalyzing && (
          <div className="flex items-start gap-3 mb-6">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-500 text-white flex items-center justify-center shrink-0 shadow-sm animate-bounce">
              <Satellite className="w-4 h-4" />
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-xs p-4 flex flex-col gap-2.5 text-xs text-brand-dark max-w-md shadow-2xs">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-brand-blue" />
                <span className="font-semibold text-brand-dark">
                  Analyzing the images...
                </span>
              </div>

              {/* Step progression chips */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                {PIPELINE_STEPS.map((step, idx) => {
                  const isDone = idx < activeStepIndex;
                  const isCurrent = idx === activeStepIndex;
                  return (
                    <span
                      key={idx}
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium border transition-all ${
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
            </div>
          </div>
        )}

        {/* Bottom Anchor for reliable scroll into view */}
        <div ref={bottomAnchorRef} className="h-1" />
      </div>

      {/* Pinned Bottom Input Outside Message Scroll Container */}
      <div className="mt-3 pt-3 border-t border-slate-100 space-y-2 shrink-0 bg-white">
        <div
          className={`flex items-center gap-2 p-2 rounded-xl border bg-slate-50 transition-all ${
            canSend
              ? 'border-slate-200 focus-within:border-brand-blue focus-within:ring-2 focus-within:ring-brand-blue/20 bg-white'
              : 'border-slate-200 bg-slate-50/60 opacity-80'
          }`}
        >
          {/* Attachment decor icon */}
          <div className="text-slate-400 pl-1">
            <Paperclip className="w-4 h-4" />
          </div>

          {/* Text Input */}
          <input
            ref={inputRef}
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!canSend || isAnalyzing}
            placeholder={placeholderText}
            className="flex-1 bg-transparent text-xs sm:text-sm text-brand-dark placeholder-slate-400 focus:outline-none disabled:cursor-not-allowed"
          />

          {/* Send Button */}
          <button
            type="button"
            onClick={() => handleSend()}
            disabled={!canSend || !queryInput.trim() || isAnalyzing}
            className={`p-2 rounded-xl text-white transition-all flex items-center justify-center shrink-0 ${
              canSend && queryInput.trim() && !isAnalyzing
                ? 'bg-brand-blue hover:bg-blue-600 shadow-sm shadow-brand-blue/30 cursor-pointer active:scale-95'
                : 'bg-slate-300 text-slate-100 cursor-not-allowed'
            }`}
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Helper Note Under Input */}
        <p className="text-[11px] text-center text-brand-muted">
          Ask about changes, land cover, vegetation, water bodies, built-up
          areas, and more.
        </p>
      </div>
    </div>
  );
}
