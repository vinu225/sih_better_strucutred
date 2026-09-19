import React, { createContext, useContext, useState, useCallback } from 'react';
import { uploadTile, sendChat } from '../api/client';

const ChatContext = createContext(null);

export function ChatProvider({ children }) {
  const [activeTile, setActiveTile] = useState(null);
  const [uploadState, setUploadState] = useState('idle'); // idle, uploading, ready, error
  const [uploadError, setUploadError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [queryInput, setQueryInput] = useState('');

  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;

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
  }, []);

  const handleSendMessage = useCallback(async (userText) => {
    const textToSend = userText !== undefined ? userText : queryInput;
    if (!textToSend || !textToSend.trim()) return;
    if (!activeTile?.tileId) return;

    const query = textToSend.trim();
    setQueryInput('');

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message
    const userMsg = {
      id: Date.now(),
      role: 'user',
      text: query,
      timestamp: timeStr,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsAnalyzing(true);

    try {
      const responseData = await sendChat({
        query: query,
        tileId: activeTile.tileId,
      });

      const assistantMsg = {
        id: Date.now() + 1,
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
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Chat request failed:', err);
      const errorMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        text: `Error: ${err.message || 'Could not reach the backend.'} Make sure the SatQuery AI backend is running on http://localhost:8000.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsAnalyzing(false);
    }
  }, [queryInput, activeTile]);

  const handleClearChat = useCallback(() => {
    setMessages([]);
  }, []);

  return (
    <ChatContext.Provider
      value={{
        activeTile,
        uploadState,
        uploadError,
        messages,
        isAnalyzing,
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
