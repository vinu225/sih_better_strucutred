/**
 * changeDemo.js - Client-side image processing, fingerprinting, scene loader, and answer generator.
 */

const FINGERPRINT_SIZE = 64; // 64x64 pixels for robust comparison
const MAX_UPLOAD_DIM = 1600; // Downscale uploads to max 1600px on long edge for fast rendering & low memory

/**
 * Per-channel normalization for an RGB Float32Array (subtract mean, divide by std per channel).
 */
export function normalizeFingerprint(raw) {
  const norm = new Float32Array(raw.length);
  for (let c = 0; c < 3; c++) {
    let sum = 0;
    let count = 0;
    for (let i = c; i < raw.length; i += 3) {
      sum += raw[i];
      count++;
    }
    const mean = sum / count;
    let variance = 0;
    for (let i = c; i < raw.length; i += 3) {
      const diff = raw[i] - mean;
      variance += diff * diff;
    }
    const std = Math.sqrt(variance / count) || 1e-6;
    for (let i = c; i < raw.length; i += 3) {
      norm[i] = (raw[i] - mean) / std;
    }
  }
  return norm;
}

/**
 * Compute Pearson correlation between two normalized Float32Array fingerprints.
 * Value is in range [-1.0, 1.0].
 */
export function computePearsonCorrelation(fp1, fp2) {
  if (!fp1 || !fp2 || fp1.length !== fp2.length) return 0;
  let dot = 0;
  for (let i = 0; i < fp1.length; i++) {
    dot += fp1[i] * fp2[i];
  }
  return dot / fp1.length;
}

/**
 * Compute 64x64 normalized RGB fingerprint directly from a raw File or Blob using createImageBitmap.
 */
export async function computeFingerprintFromFile(fileOrBlob) {
  try {
    let bitmap;
    if (typeof createImageBitmap === 'function') {
      bitmap = await createImageBitmap(fileOrBlob);
    } else {
      bitmap = await new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(fileOrBlob);
        img.onload = () => {
          URL.revokeObjectURL(url);
          resolve(img);
        };
        img.onerror = (e) => {
          URL.revokeObjectURL(url);
          reject(e);
        };
        img.src = url;
      });
    }

    const canvas = document.createElement('canvas');
    canvas.width = FINGERPRINT_SIZE;
    canvas.height = FINGERPRINT_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, FINGERPRINT_SIZE, FINGERPRINT_SIZE);

    if (bitmap.close) bitmap.close();

    const imgData = ctx.getImageData(0, 0, FINGERPRINT_SIZE, FINGERPRINT_SIZE);
    const data = imgData.data;
    const raw = new Float32Array(FINGERPRINT_SIZE * FINGERPRINT_SIZE * 3);
    let j = 0;
    for (let i = 0; i < data.length; i += 4) {
      raw[j++] = data[i];     // R
      raw[j++] = data[i + 1]; // G
      raw[j++] = data[i + 2]; // B
    }

    return normalizeFingerprint(raw);
  } catch (err) {
    console.warn('Fingerprint computation error:', err);
    return null;
  }
}

/**
 * Convert a File object into a persistent downscaled data URL (JPEG 0.92, max 1600px)
 * and compute its 64x64 normalized fingerprint from the original file.
 */
export async function processUploadedFile(file) {
  const fingerprintPromise = computeFingerprintFromFile(file);

  const dataUrlPromise = new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const srcDataUrl = reader.result;
      const img = new Image();
      img.onload = () => {
        try {
          const origW = img.naturalWidth || img.width;
          const origH = img.naturalHeight || img.height;
          const aspectRatio = origW / origH;

          // Downscale persistent Data URL (max 1600px)
          const scale = Math.min(1.0, MAX_UPLOAD_DIM / Math.max(origW, origH));
          const targetW = Math.max(1, Math.round(origW * scale));
          const targetH = Math.max(1, Math.round(origH * scale));

          const mainCanvas = document.createElement('canvas');
          mainCanvas.width = targetW;
          mainCanvas.height = targetH;
          const mainCtx = mainCanvas.getContext('2d');
          mainCtx.drawImage(img, 0, 0, targetW, targetH);

          const persistentDataUrl =
            file.type === 'image/png' && scale === 1.0
              ? srcDataUrl
              : mainCanvas.toDataURL('image/jpeg', 0.92);

          resolve({
            dataUrl: persistentDataUrl,
            aspectRatio,
            width: targetW,
            height: targetH,
          });
        } catch (err) {
          resolve({
            dataUrl: srcDataUrl,
            aspectRatio: 1,
            width: 800,
            height: 600,
          });
        }
      };
      img.onerror = () => reject(new Error('Failed to load image.'));
      img.src = srcDataUrl;
    };
    reader.onerror = () => reject(new Error('FileReader error.'));
    reader.readAsDataURL(file);
  });

  const [fp, imgMeta] = await Promise.all([fingerprintPromise, dataUrlPromise]);

  return {
    dataUrl: imgMeta.dataUrl,
    aspectRatio: imgMeta.aspectRatio,
    width: imgMeta.width,
    height: imgMeta.height,
    fingerprint: fp,
  };
}

/**
 * Load reference images from public/demo/change/scene-01/ on mount.
 */
export async function loadReferenceFingerprints() {
  const sceneId = 'scene-01';
  const beforeUrl = `/demo/change/${sceneId}/before.jpeg`;
  const afterUrl = `/demo/change/${sceneId}/after.jpeg`;

  try {
    const [resBefore, resAfter] = await Promise.all([
      fetch(beforeUrl, { cache: 'no-cache' }),
      fetch(afterUrl, { cache: 'no-cache' }),
    ]);

    const typeBefore = resBefore.headers.get('content-type') || '';
    const typeAfter = resAfter.headers.get('content-type') || '';

    if (!resBefore.ok || !resAfter.ok) {
      console.warn('Reference images failed to load:', resBefore.status, resAfter.status);
      return null;
    }

    if (!typeBefore.startsWith('image/') && !typeBefore.includes('octet-stream')) {
      console.warn('Reference before image has unexpected content-type:', typeBefore);
    }

    const [blobBefore, blobAfter] = await Promise.all([
      resBefore.blob(),
      resAfter.blob(),
    ]);

    const [fpBefore, fpAfter] = await Promise.all([
      computeFingerprintFromFile(blobBefore),
      computeFingerprintFromFile(blobAfter),
    ]);

    if (!fpBefore || !fpAfter) {
      console.warn('Could not extract reference fingerprints.');
      return null;
    }

    const crossCorr = computePearsonCorrelation(fpBefore, fpAfter);
    if (import.meta.env.DEV) {
      console.log(`[ChangeDemo] Loaded reference fingerprints (cross-correlation: ${crossCorr.toFixed(4)})`);
    }

    return {
      sceneId,
      fpBefore,
      fpAfter,
      crossCorr,
      beforeUrl,
      afterUrl,
      diffUrl: `/demo/change/${sceneId}/diff.png`,
    };
  } catch (err) {
    console.warn('Error loading reference fingerprints:', err);
    return null;
  }
}

/**
 * Classify an uploaded image against the reference fingerprints.
 * Returns: 'before-ref' | 'after-ref' | 'unknown'
 */
export function classifyImageFingerprint(uploadedFp, refData) {
  if (!uploadedFp || !refData?.fpBefore || !refData?.fpAfter) return 'unknown';

  const rBefore = computePearsonCorrelation(uploadedFp, refData.fpBefore);
  const rAfter = computePearsonCorrelation(uploadedFp, refData.fpAfter);

  if (import.meta.env.DEV) {
    console.log(`[ChangeDemo] Slot classification -> r(before): ${rBefore.toFixed(4)}, r(after): ${rAfter.toFixed(4)}`);
  }

  // Classify as before-ref or after-ref when correlation is at least 0.85 and higher than the other reference
  if (rBefore >= 0.85 && rBefore > rAfter) {
    return 'before-ref';
  }
  if (rAfter >= 0.85 && rAfter > rBefore) {
    return 'after-ref';
  }
  return 'unknown';
}

/**
 * Fetch scene-01/result.json data.
 */
export async function fetchSceneResult(sceneId = 'scene-01') {
  try {
    const res = await fetch(`/demo/change/${sceneId}/result.json`, { cache: 'no-cache' });
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn(`Could not load scene ${sceneId} result.json:`, err);
    return null;
  }
}

/**
 * Keyword-based intent classification.
 */
export function matchIntent(query) {
  const q = (query || '').toLowerCase();

  if (/deforest/i.test(q)) return 'deforestation';
  if (/vegetation|forest|green|tree|canopy|crop|plant/i.test(q)) return 'vegetation';
  if (/water|river|lake|flood|stream|shoreline|sea|reservoir/i.test(q)) return 'water';
  if (/building|urban|construction|city|built|infrastructure|road/i.test(q)) return 'urban';
  if (/how much|percent|area|extent|rate|amount|stat/i.test(q)) return 'howMuch';
  if (/difference|heatmap|highlight|mask|overlay/i.test(q)) return 'showDifference';
  if (/what|change|compare|overview|describe|summary/i.test(q)) return 'overview';

  return 'fallback';
}

/**
 * Build demo answer according to prompt requirements:
 * - If unmatched / user uploaded images:
 *     "Demo mode: no analysis is run on uploaded images. Use the slider to compare the two dates."
 *     No stats, no diff toggle, no tools chips.
 * - If sample scene matched:
 *     Uses result.json numbers and computed spatial positions.
 */
export function buildDemoAnswer({ query, sceneResult, isMatched = false, isFollowUp = false }) {
  if (!isMatched || !sceneResult) {
    return {
      text: 'Demo mode: no analysis is run on uploaded images. Use the slider to compare the two dates.',
      stats: null,
      toolsUsed: [],
      diffUrl: null,
      shouldShowDiffMap: false,
      illustrative: false,
      location: null,
      followUps: [],
    };
  }

  const intent = matchIntent(query);
  const answers = sceneResult.answers || {};

  let answerText = answers[intent];
  if (!answerText) {
    answerText = isFollowUp ? (answers.fallback || answers.overview) : answers.overview;
  }

  // If follow up and intent is specific, provide focused reference
  if (isFollowUp && intent !== 'showDifference') {
    if (intent === 'vegetation' || intent === 'deforestation') {
      answerText = `Looking at the comparison above, vegetation canopy modifications are most noticeable across the upper-right sectors. Dragging the slider reveals where green cover has altered between the two observations. You can also ask about built-up areas or water bodies.`;
    } else if (intent === 'urban') {
      answerText = `Referring to the comparison above, built-up structures and developed parcels show subtle progression in the central area. Moving the divider highlights the localized progression of infrastructure.`;
    } else if (intent === 'water') {
      answerText = `As shown in the comparison above, surface water margins and moisture zones along the river and upper shoreline show localized variations. Inquire about vegetation loss or overall area for further metrics.`;
    }
  }

  const shouldShowDiffMap = intent === 'showDifference';

  return {
    text: answerText,
    stats: sceneResult.stats || null,
    toolsUsed: sceneResult.tools_used || ['Change detection'],
    diffUrl: `/demo/change/scene-01/diff.png`,
    shouldShowDiffMap,
    illustrative: Boolean(sceneResult.illustrative),
    location: sceneResult.location || null,
    followUps: sceneResult.follow_ups || [
      'What changed in vegetation?',
      'Show urban expansion',
      'How much total area changed?',
    ],
  };
}
