import { MOCK_UPLOAD_RESPONSE, MOCK_CHAT_RESPONSE } from './mock';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
const BASE_URL = import.meta.env.VITE_API_URL || '';

export async function checkHealth() {
  if (USE_MOCK) {
    return { status: 'ok', version: '1.0.0 (Mock Mode)', device: 'cpu' };
  }
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return await res.json();
  } catch (err) {
    console.warn('Backend /health unreachable:', err.message);
    throw err;
  }
}

export async function uploadTile(file, modalityHint = null) {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 800));
    return { ...MOCK_UPLOAD_RESPONSE, tile_id: file.name.replace(/\.[^/.]+$/, "") };
  }

  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`${BASE_URL}/api/v1/upload-tile`, window.location.origin);
  if (modalityHint) {
    url.searchParams.append('modality_hint', modalityHint);
  }

  const res = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = `Upload failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch (_) {}
    throw new Error(errorDetail);
  }

  return await res.json();
}

export async function sendChat({ query, tileId, modalityHint = null }) {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    return { ...MOCK_CHAT_RESPONSE, query, tile_id: tileId };
  }

  const res = await fetch(`${BASE_URL}/api/v1/agent/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      tile_id: tileId,
      query: query,
      modality_hint: modalityHint || undefined,
    }),
  });

  if (!res.ok) {
    let errorDetail = `Backend returned HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch (_) {}
    throw new Error(errorDetail);
  }

  return await res.json();
}
