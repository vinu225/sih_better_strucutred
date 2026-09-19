import React from 'react';
import Hero from '../components/Hero';
import FeatureCards from '../components/FeatureCards';
import UploadPanel from '../components/UploadPanel';
import ExampleQueries from '../components/ExampleQueries';
import ChatWindow from '../components/ChatWindow';

export default function ChatPage() {
  return (
    <div className="space-y-4">
      {/* Hero Header Banner */}
      <Hero />

      {/* 4 Feature Cards */}
      <FeatureCards />

      {/* Main 2-Column Responsive Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: Upload & Examples (~340px) */}
        <div className="lg:col-span-4 xl:col-span-4 space-y-4">
          <UploadPanel />
          <ExampleQueries />
        </div>

        {/* Right Column: Chat Window */}
        <div className="lg:col-span-8 xl:col-span-8">
          <ChatWindow />
        </div>
      </div>
    </div>
  );
}
