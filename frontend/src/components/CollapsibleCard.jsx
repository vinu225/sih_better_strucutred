import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

/**
 * Reusable collapsible card component for structured panels like "About this estimate" and "How this was computed".
 */
export default function CollapsibleCard({
  id,
  isOpen,
  onToggle,
  defaultOpen = false,
  tone = 'default', // 'default' | 'info' | 'slate'
  icon: Icon,
  iconClassName,
  title,
  badge = null,
  headerRight = null,
  children,
  className = '',
}) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen);
  const open = isOpen !== undefined ? isOpen : internalOpen;

  const toggle = () => {
    if (onToggle) {
      onToggle(!open);
    } else {
      setInternalOpen(!open);
    }
  };

  const contentId = id || `collapsible-${Math.random().toString(36).substring(2, 9)}`;
  const isSlateTone = tone === 'info' || tone === 'slate';

  const defaultIconClass = isSlateTone
    ? 'p-1.5 rounded-md bg-[#eef2f6] text-[#64748b] flex items-center justify-center shrink-0'
    : 'p-1 rounded-md bg-brand-blue-tint text-brand-blue shrink-0';

  const cardBorderClass = isSlateTone
    ? 'border border-[#e2e8f0] rounded-xl overflow-hidden bg-[#f8fafc] shadow-2xs'
    : 'border border-brand-border/80 rounded-xl overflow-hidden bg-white shadow-2xs';

  const headerClass = isSlateTone
    ? 'w-full px-3.5 py-2.5 bg-[#f8fafc] hover:bg-[#f1f5f9] focus:outline-none focus:ring-2 focus:ring-brand-blue/30 transition-colors flex items-center justify-between text-left group cursor-pointer'
    : 'w-full px-3.5 py-2.5 bg-slate-50/80 hover:bg-slate-100/80 focus:outline-none focus:ring-2 focus:ring-brand-blue/30 transition-colors flex items-center justify-between text-left group cursor-pointer';

  const titleClass = isSlateTone
    ? 'text-xs font-semibold text-[#334155]'
    : 'text-xs font-semibold text-brand-dark';

  const bodyClass = isSlateTone
    ? 'p-3.5 bg-[#f8fafc] border-t border-[#e2e8f0]'
    : 'p-3.5 bg-white border-t border-slate-100';

  return (
    <div className={`${cardBorderClass} ${className}`}>
      {/* Accordion Header */}
      <button
        type="button"
        id={`${contentId}-header`}
        aria-expanded={open}
        aria-controls={contentId}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
          }
        }}
        className={headerClass}
      >
        <div className="flex items-center gap-2 min-w-0">
          {Icon && (
            <div className={iconClassName || defaultIconClass}>
              <Icon className="w-[18px] h-[18px]" size={18} />
            </div>
          )}
          <span className={titleClass}>{title}</span>
          {badge}
        </div>

        <div className="flex items-center gap-1.5 text-[#94a3b8] group-hover:text-slate-600 transition-colors shrink-0">
          {headerRight}
          {open ? (
            <ChevronDown className="w-4 h-4 text-[#94a3b8] transition-transform duration-200" />
          ) : (
            <ChevronRight className="w-4 h-4 text-[#94a3b8] transition-transform duration-200" />
          )}
        </div>
      </button>

      {/* Accordion Body */}
      {open && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={`${contentId}-header`}
          className={bodyClass}
        >
          {children}
        </div>
      )}
    </div>
  );
}
