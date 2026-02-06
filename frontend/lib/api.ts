export type AIResponse = {
  response?: string;
  triage_level?: string;
  findings?: string[] | string;
  confidence?: number;
  recommendations?: string[] | string;
  transcript?: string | null;
  raw_results?: {
    audio?: Record<string, number>;
  };
};

/** WebSocket chat: response payload from backend */
export type ChatFinding = {
  id: string;
  label: string;
  confidence: number;
  confidence_label: string;
};
export type ChatDifferential = {
  id: string;
  condition: string;
  probability: number;
  probability_label: string;
};
export type ChatAction = {
  id: string;
  text: string;
  priority: string;
  priority_label: string;
  priority_color: string;
};
export type ChatResponseData = {
  message_id: string;
  timestamp: string;
  response_text: string;
  triage_level: string;
  triage_color: string;
  confidence: number;
  findings: ChatFinding[];
  differential_diagnosis: ChatDifferential[];
  recommended_actions: ChatAction[];
};

const DEFAULT_API_BASE = "http://localhost:8000";

function getApiBase() {
  return process.env.NEXT_PUBLIC_API_BASE || DEFAULT_API_BASE;
}

/** WebSocket URL for chat (ws scheme, same host as API). */
export function getChatWsUrl(): string {
  const base = getApiBase();
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/chat/ws`;
}

/** HTTP fallback for chat when WebSocket unavailable. */
export async function chatHttpFallback(payload: {
  text?: string;
  image?: string;
  audio?: string;
  patient_id?: string;
  consultation_id?: string;
}): Promise<ChatResponseData> {
  const body = new FormData();
  if (payload.text) body.append("text", payload.text);
  if (payload.patient_id) body.append("patient_id", payload.patient_id);
  if (payload.consultation_id) body.append("consultation_id", payload.consultation_id);
  if (payload.image) {
    const blob = base64DataUrlToBlob(payload.image);
    if (blob) body.append("image", blob, "image.jpg");
  }
  if (payload.audio) {
    const blob = base64DataUrlToBlob(payload.audio);
    if (blob) body.append("audio", blob, "audio.wav");
  }
  const res = await request("/chat/", { method: "POST", body });
  return res as ChatResponseData;
}

function base64DataUrlToBlob(dataUrl: string): Blob | null {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) return null;
  const base64 = match[2];
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: match[1] });
}

async function request(path: string, init?: RequestInit) {
  const url = `${getApiBase()}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    let detail = "Request failed.";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function healthCheck() {
  return request("/health/");
}

export async function sendChat(options: {
  text?: string;
  image?: File;
  audio?: File;
  mode?: "chat" | "voice";
}): Promise<AIResponse> {
  const body = new FormData();
  if (options.text) body.append("text", options.text);
  if (options.image) body.append("image", options.image);
  if (options.audio) body.append("audio", options.audio);
  body.append("mode", options.mode || "chat");
  return request("/chat/", { method: "POST", body });
}

export async function analyzeImage(options: {
  image: File;
  prompt?: string;
}): Promise<AIResponse> {
  const body = new FormData();
  body.append("image", options.image);
  if (options.prompt) body.append("prompt", options.prompt);
  return request("/vision/analyze", { method: "POST", body });
}

export async function analyzeAudio(options: {
  audio: File;
  prompt?: string;
}): Promise<AIResponse> {
  const body = new FormData();
  body.append("audio", options.audio);
  if (options.prompt) body.append("prompt", options.prompt);
  return request("/audio/analyze", { method: "POST", body });
}

export async function sendWebrtcOffer(payload: {
  room_id: string;
  peer_id: string;
  sdp: Record<string, unknown>;
}) {
  return request("/chat/webrtc/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function sendWebrtcAnswer(payload: {
  room_id: string;
  peer_id: string;
  sdp: Record<string, unknown>;
}) {
  return request("/chat/webrtc/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function sendWebrtcIce(payload: {
  room_id: string;
  peer_id: string;
  candidate: Record<string, unknown>;
}) {
  return request("/chat/webrtc/ice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
