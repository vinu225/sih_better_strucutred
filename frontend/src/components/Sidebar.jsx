import React from 'react';
import { NavLink } from 'react-router-dom';
import { MessageSquare, Info, Satellite } from 'lucide-react';

export default function Sidebar() {
  return (
    <aside className="relative overflow-hidden w-full lg:w-[320px] lg:min-h-screen bg-white border-b lg:border-b-0 lg:border-r border-brand-border flex flex-col p-6 z-20 shrink-0">
      {/* Background Image on Desktop (Hidden under 900px / mobile) - Full strength, center bottom */}
      <div
        className="hidden lg:block absolute inset-0 bg-no-repeat pointer-events-none"
        style={{
          backgroundImage: `url('/assets/sidebar-bg.png'), url('/assets/sidebar-bg.jpg')`,
          backgroundPosition: 'center bottom',
          backgroundSize: 'cover',
        }}
      />
      {/* Topmost subtle gradient overlay only (30% downward is completely clear) */}
      <div
        className="hidden lg:block absolute inset-0 pointer-events-none"
        style={{
          background:
            'linear-gradient(to bottom, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.5) 8%, rgba(255,255,255,0.15) 18%, rgba(255,255,255,0) 28%)',
        }}
      />

      <div className="relative z-10 space-y-6">
        {/* Brand Header with translucent background and blur */}
        <div
          className="flex items-center gap-3.5 p-3.5 rounded-2xl border border-white/60 shadow-2xs"
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.55)',
            backdropFilter: 'blur(6px)',
            WebkitBackdropFilter: 'blur(6px)',
          }}
        >
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-400 flex items-center justify-center text-white shadow-md shadow-brand-blue/20 shrink-0">
            <Satellite className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-display font-normal text-[28px] text-brand-dark tracking-[-0.01em] leading-[1.1]">
                SatQuery AI
              </span>
            </div>
            <p className="text-[14px] text-brand-muted font-medium leading-normal">
              Ask the Earth. Get Answers.
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex lg:flex-col gap-2.5">
          <NavLink
            to="/"
            end
            style={({ isActive }) => ({
              backgroundColor: isActive ? undefined : 'rgba(255, 255, 255, 0.55)',
              backdropFilter: 'blur(6px)',
              WebkitBackdropFilter: 'blur(6px)',
            })}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-[14px] rounded-xl font-semibold text-[16px] transition-all duration-150 border ${
                isActive
                  ? 'bg-brand-blue-tint text-brand-blue shadow-sm border-brand-blue/20'
                  : 'text-brand-dark hover:bg-white/80 border-white/50 shadow-2xs'
              }`
            }
          >
            <MessageSquare className="w-5 h-5 shrink-0" />
            <span>Chat</span>
          </NavLink>

          <NavLink
            to="/about"
            style={({ isActive }) => ({
              backgroundColor: isActive ? undefined : 'rgba(255, 255, 255, 0.55)',
              backdropFilter: 'blur(6px)',
              WebkitBackdropFilter: 'blur(6px)',
            })}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-[14px] rounded-xl font-semibold text-[16px] transition-all duration-150 border ${
                isActive
                  ? 'bg-brand-blue-tint text-brand-blue shadow-sm border-brand-blue/20'
                  : 'text-brand-dark hover:bg-white/80 border-white/50 shadow-2xs'
              }`
            }
          >
            <Info className="w-5 h-5 shrink-0" />
            <span>About</span>
          </NavLink>
        </nav>
      </div>
    </aside>
  );
}

