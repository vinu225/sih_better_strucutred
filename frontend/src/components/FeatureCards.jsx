import React from 'react';
import { MessageSquareText, Layers, ScanSearch, LineChart } from 'lucide-react';
import { useChat } from '../context/ChatContext';

const FEATURES = [
  {
    title: 'Understand',
    desc: 'natural language queries',
    icon: MessageSquareText,
    iconColor: 'text-emerald-600',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-100',
  },
  {
    title: 'Analyze',
    desc: 'optical & SAR imagery',
    icon: Layers,
    iconColor: 'text-indigo-600',
    bgColor: 'bg-indigo-50',
    borderColor: 'border-indigo-100',
  },
  {
    title: 'Detect',
    desc: 'changes over time',
    icon: ScanSearch,
    iconColor: 'text-amber-600',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-100',
  },
  {
    title: 'Get Insights',
    desc: 'with AI-powered analysis',
    icon: LineChart,
    iconColor: 'text-brand-blue',
    bgColor: 'bg-brand-blue-tint',
    borderColor: 'border-blue-100',
  },
];

export default function FeatureCards() {
  const { messages } = useChat();
  if (messages.length > 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5 mb-6">
      {FEATURES.map((item, idx) => {
        const Icon = item.icon;
        return (
          <div
            key={idx}
            className="bg-white rounded-2xl border border-brand-border p-4 shadow-soft hover:shadow-soft-hover transition-all duration-200 flex items-center gap-3.5"
          >
            <div className={`w-10 h-10 rounded-xl ${item.bgColor} ${item.borderColor} border flex items-center justify-center shrink-0`}>
              <Icon className={`w-5 h-5 ${item.iconColor}`} />
            </div>
            <div className="min-w-0">
              <h3 className="text-sm font-bold text-brand-dark leading-tight">{item.title}</h3>
              <p className="text-xs text-brand-muted truncate font-medium mt-0.5">{item.desc}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
