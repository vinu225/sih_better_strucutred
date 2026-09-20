import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { uploadTile, sendChat } from '../api/client';

const ChatContext = createContext(null);

export const CHAT_PIPELINE_STEPS = [
  'Image received',
  'Modality detected',
  'Selecting tools',
  'Running analysis',
  'Composing answer',
];

export function ChatProvider({ children }) {
  const [activeTile, setActiveTile] = useState(null);
  const [uploadState, setUploadState] = useState('idle'); // idle, uploading, ready, error
  const [uploadError, setUploadError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStepIndex, setAnalysisStepIndex] = useState(0);
  const [queryInput, setQueryInput] = useState('');

  // Active request controller ref to cancel ongoing timers on unmount/replacement
  const activeTimersRef = useRef({ interval: null, timeout: null });

  const clearActiveTimers = useCallback(() => {
    if (activeTimersRef.current.interval) {
      clearInterval(activeTimersRef.current.interval);
      activeTimersRef.current.interval = null;
    }
    if (activeTimersRef.current.timeout) {
      clearTimeout(activeTimersRef.current.timeout);
      activeTimersRef.current.timeout = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearActiveTimers();
    };
  }, [clearActiveTimers]);

  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;

    clearActiveTimers();
    setIsAnalyzing(false);
    setUploadState('uploading');
    setUploadError(null);

    const isNpy = file.name.endsWith('.npy') || file.name.endsWith('.npz');
    const previewUrl = isNpy ? null : URL.createObjectURL(file);

    try {
      const data = await uploadTile(file);
      setActiveTile({
        tileId: data.tile_id,
        filename: file.name,
        previewUrl: previewUrl,
        isNpy: isNpy,
        modality: data.modality,
        shape: data.shape,
        displayLabel: data.display_label || data.modality,
        detectionBasis: data.detection_basis,
      });
      setUploadState('ready');
    } catch (err) {
      console.error('File upload failed:', err);
      setUploadState('error');
      setUploadError(err.message || 'Upload failed. Check that the backend is running.');
    }
  }, [clearActiveTimers]);

  const handleSendMessage = useCallback(async (userText) => {
    const textToSend = userText !== undefined ? userText : queryInput;
    if (!textToSend || !textToSend.trim()) return;
    if (!activeTile?.tileId) return;

    const query = textToSend.trim();
    setQueryInput('');
    clearActiveTimers();

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: query,
      timestamp: timeStr,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsAnalyzing(true);
    setAnalysisStepIndex(0);

    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const startTime = Date.now();

    // Step ticker: advances every 450ms up to step 4 ("Composing answer")
    if (!prefersReducedMotion) {
      activeTimersRef.current.interval = setInterval(() => {
        setAnalysisStepIndex((prev) => {
          if (prev < CHAT_PIPELINE_STEPS.length - 1) {
            return prev + 1;
          }
          return prev;
        });
      }, 450);
    }

    try {
      const responseData = await sendChat({
        query: query,
        tileId: activeTile.tileId,
      });

      const finishAndRender = () => {
        clearActiveTimers();
        setIsAnalyzing(false);

        const assistantMsg = {
          id: `bot-${Date.now()}`,
          role: 'assistant',
          text: responseData.response || responseData.answer || 'Analysis complete.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          imagePreviewUrl: activeTile.previewUrl || `/api/v1/tiles/${activeTile.tileId}/composite?mode=rgb`,
          plan: responseData.plan || [responseData.selected_task || 'analysis'],
          toolArtifacts: responseData.tool_artifacts || {},
          visualEvidence: responseData.visual_evidence || responseData.tool_artifacts?.visual_evidence || null,
          executionTrace: responseData.execution_trace || responseData.tool_artifacts?.trace || [],
          selectedTask: responseData.selected_task || null,
          selectedModelOrTool: responseData.selected_model_or_tool || null,
          confidence: responseData.confidence !== undefined ? responseData.confidence : null,
          detectedModality: responseData.detected_modality || activeTile.displayLabel || activeTile.modality || 'RGB optical',
          isNew: !prefersReducedMotion,
        };

        setMessages((prev) => [...prev, assistantMsg]);
      };

      if (prefersReducedMotion) {
        finishAndRender();
      } else {
        const elapsed = Date.now() - startTime;
        const minDisplayTime = 1800; // minimum ~1.8s
        const remainingTime = Math.max(0, minDisplayTime - elapsed);

        activeTimersRef.current.timeout = setTimeout(() => {
          finishAndRender();
        }, remainingTime);
      }
    } catch (err) {
      console.error('Chat request failed:', err);
      clearActiveTimers();
      setIsAnalyzing(false);

      const errorMsg = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        text: `Error: ${err.message || 'Could not reach the backend.'} Make sure the SatQuery AI backend is running on http://localhost:8000.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true,
        isNew: false,
      };
      setMessages((prev) => [...prev, errorMsg]);
    }
  }, [queryInput, activeTile, clearActiveTimers]);

  const handleClearChat = useCallback(() => {
    clearActiveTimers();
    setIsAnalyzing(false);
    setMessages([]);
  }, [clearActiveTimers]);

  return (
    <ChatContext.Provider
      value={{
        activeTile,
        uploadState,
        uploadError,
        messages,
        isAnalyzing,
        analysisStepIndex,
        queryInput,
        setQueryInput,
        handleFileUpload,
        handleSendMessage,
        handleClearChat,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
