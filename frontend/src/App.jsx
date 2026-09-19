import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import AboutPage from './pages/AboutPage';
import { ChatProvider } from './context/ChatContext';
import { Sparkles } from 'lucide-react';

export default function App() {
  const isMockMode = import.meta.env.VITE_USE_MOCK === 'true';

  return (
    <ChatProvider>
      <div className="min-h-screen bg-canvas flex flex-col lg:flex-row antialiased">
        {/* Left Fixed/Sticky Navigation Sidebar */}
        <Sidebar />

        {/* Main Application Content Area */}
        <main className="flex-1 min-w-0 p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full">
          {/* Demo Mode Badge if mock active */}
          {isMockMode && (
            <div className="mb-4 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-100 border border-amber-300 text-amber-800 text-xs font-bold shadow-xs">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Demo Mock Mode Active (VITE_USE_MOCK=true)</span>
            </div>
          )}

          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </ChatProvider>
  );
}
