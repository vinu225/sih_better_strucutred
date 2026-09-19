import React, { useEffect, useRef } from 'react';
import { useChat } from '../context/ChatContext';
import MessageBubble from './MessageBubble';
import EmptyState from './EmptyState';
import ChatInput from './ChatInput';
import { Satellite, Loader2 } from 'lucide-react';

export default function ChatWindow() {
  const { messages, isAnalyzing } = useChat();
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAnalyzing]);

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

        {/* Loading Indicator when analyzing */}
        {isAnalyzing && (
          <div className="flex items-start gap-3 mb-6">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-500 text-white flex items-center justify-center shrink-0 shadow-sm animate-bounce">
              <Satellite className="w-4 h-4" />
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-2xl rounded-tl-xs p-3.5 flex items-center gap-2.5 text-xs text-brand-dark">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-brand-blue" />
              <span className="font-medium">Analyzing the satellite image with SatQuery AI...</span>
            </div>
          </div>
        )}
      </div>

      {/* Pinned Bottom Input */}
      <ChatInput />
    </div>
  );
}
