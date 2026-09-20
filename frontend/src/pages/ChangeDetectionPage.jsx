import React, { useState, useEffect, useRef } from 'react';
import ChangeHero from '../components/ChangeHero';
import FeatureCards from '../components/FeatureCards';
import ChangeUploadPanel from '../components/ChangeUploadPanel';
import ChangeExampleQueries from '../components/ChangeExampleQueries';
import ChangeChatWindow from '../components/ChangeChatWindow';
import {
  loadReferenceFingerprints,
  classifyImageFingerprint,
  fetchSceneResult,
} from '../demo/changeDemo';

export default function ChangeDetectionPage() {
  const [refData, setRefData] = useState(null);
  const [sceneResult, setSceneResult] = useState(null);

  // Upload slots start completely EMPTY
  const [beforeImage, setBeforeImage] = useState(null);
  const [afterImage, setAfterImage] = useState(null);
  const [beforeDate, setBeforeDate] = useState('');
  const [afterDate, setAfterDate] = useState('');

  // Classification states
  const [beforeClass, setBeforeClass] = useState('unknown');
  const [afterClass, setAfterClass] = useState('unknown');

  const [queryInput, setQueryInput] = useState('');
  const [messages, setMessages] = useState([]);

  // Load reference fingerprints and scene-01 result.json on mount
  useEffect(() => {
    let isMounted = true;
    async function init() {
      const [refs, scene] = await Promise.all([
        loadReferenceFingerprints(),
        fetchSceneResult('scene-01'),
      ]);
      if (isMounted) {
        setRefData(refs);
        setSceneResult(scene);
      }
    }
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  // Classify Before slot when image or refData updates
  useEffect(() => {
    if (!beforeImage?.fingerprint || !refData) {
      setBeforeClass('unknown');
      return;
    }
    const cls = classifyImageFingerprint(beforeImage.fingerprint, refData);
    setBeforeClass(cls);
  }, [beforeImage?.fingerprint, refData]);

  // Classify After slot when image or refData updates
  useEffect(() => {
    if (!afterImage?.fingerprint || !refData) {
      setAfterClass('unknown');
      return;
    }
    const cls = classifyImageFingerprint(afterImage.fingerprint, refData);
    setAfterClass(cls);
  }, [afterImage?.fingerprint, refData]);

  const isMatched = beforeClass === 'before-ref' && afterClass === 'after-ref';
  const isReversed = beforeClass === 'after-ref' && afterClass === 'before-ref';

  const handleSwapSlots = () => {
    const tempImg = beforeImage;
    const tempDate = beforeDate;
    setBeforeImage(afterImage);
    setBeforeDate(afterDate);
    setAfterImage(tempImg);
    setAfterDate(tempDate);
  };

  // Detect image replacement during active conversation
  const prevImagesRef = useRef({ beforeUrl: null, afterUrl: null });
  useEffect(() => {
    const prev = prevImagesRef.current;
    if (messages.length > 0) {
      if (
        (prev.beforeUrl && beforeImage?.url && prev.beforeUrl !== beforeImage.url) ||
        (prev.afterUrl && afterImage?.url && prev.afterUrl !== afterImage.url)
      ) {
        const bName = beforeImage?.name || 'Image 1';
        const aName = afterImage?.name || 'Image 2';
        setMessages((curr) => [
          ...curr,
          {
            id: `divider-${Date.now()}`,
            type: 'comparison_divider',
            text: `New comparison: ${bName} vs ${aName}`,
          },
        ]);
      }
    }
    prevImagesRef.current = {
      beforeUrl: beforeImage?.url || null,
      afterUrl: afterImage?.url || null,
    };
  }, [beforeImage?.url, afterImage?.url, messages.length]);

  const hasMessages = messages.length > 0;

  return (
    <div className="space-y-4">
      {/* Hero Header Banner */}
      <ChangeHero hasMessages={hasMessages} />

      {/* 4 Feature Cards (Hidden after first message) */}
      {!hasMessages && <FeatureCards />}

      {/* Main 2-Column Responsive Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: Upload & Examples (~340px) */}
        <div className="lg:col-span-4 xl:col-span-4 space-y-4">
          <ChangeUploadPanel
            beforeImage={beforeImage}
            afterImage={afterImage}
            setBeforeImage={setBeforeImage}
            setAfterImage={setAfterImage}
            beforeDate={beforeDate}
            setBeforeDate={setBeforeDate}
            afterDate={afterDate}
            setAfterDate={setAfterDate}
            isReversed={isReversed}
            onSwapSlots={handleSwapSlots}
          />
          <ChangeExampleQueries onSelectQuery={(q) => setQueryInput(q)} />
        </div>

        {/* Right Column: Chat Window */}
        <div className="lg:col-span-8 xl:col-span-8">
          <ChangeChatWindow
            beforeImage={beforeImage}
            afterImage={afterImage}
            beforeDate={beforeDate}
            afterDate={afterDate}
            sceneResult={sceneResult}
            isMatched={isMatched}
            queryInput={queryInput}
            setQueryInput={setQueryInput}
            messages={messages}
            setMessages={setMessages}
          />
        </div>
      </div>
    </div>
  );
}
