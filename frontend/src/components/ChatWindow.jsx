import React, { useEffect, useRef } from 'react';
import { useChat, CHAT_PIPELINE_STEPS } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import EmptyState from './EmptyState';
import ChatInput from './ChatInput';
import AnalysisProgress from './AnalysisProgress';

export default function ChatWindow() {
  const { messages, isAnalyzing, analysisStepIndex } = useChat();
  const scrollRef = useRef(null);
  const bottomAnchorRef = useRef(null);

  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    if (bottomAnchorRef.current) {
      bottomAnchorRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAnalyzing, analysisStepIndex]);

  return (
    <div className="bg-white rounded-2xl border border-brand-border p-5 shadow-soft flex flex-col h-[640px]">
      {/* Scrollable Message Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pr-1.5 space-y-2">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble
              key={msg.id || idx}
              message={msg}
              isLast={idx === messages.length - 1}
            />
          ))
        )}

        {/* Loading Indicator with Animated Step Chips */}
        {isAnalyzing && (
          <AnalysisProgress
            title="Analyzing the image..."
            steps={CHAT_PIPELINE_STEPS}
            activeStepIndex={analysisStepIndex}
            reducedMotion={prefersReducedMotion}
          />
        )}

        <div ref={bottomAnchorRef} className="h-1" />
      </div>

      {/* Pinned Bottom Input */}
      <ChatInput />
    </div>
  );
}
