import { create } from "zustand";
import type { ChatResponseData, ChatFinding, ChatDifferential, ChatAction } from "@/lib/api";

export type ChatMessageRole = "user" | "assistant" | "system";

export type ChatMessage = {
  id: string;
  role: ChatMessageRole;
  content: string;
  timestamp: string;
  imageDataUrl?: string;
  status?: "sending" | "sent" | "error";
  error?: string;
  responseData?: ChatResponseData;
};

export type AddMessageInput = Omit<ChatMessage, "id" | "timestamp">;

export type QueuedMessage = {
  id: string;
  text?: string;
  image?: string;
  audio?: string;
  patient_id?: string;
  consultation_id?: string;
};

type ChatState = {
  messages: ChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
  statusMessage: string | null;
  currentPatientId: string | null;
  currentConsultationId: string | null;
  findings: ChatFinding[];
  differential: ChatDifferential[];
  actions: ChatAction[];
  triageLevel: string;
  triageColor: string;
  confidence: number;
  queue: QueuedMessage[];
  addMessage: (msg: AddMessageInput) => ChatMessage;
  updateMessage: (id: string, update: Partial<ChatMessage>) => void;
  setConnection: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setStatusMessage: (msg: string | null) => void;
  setCurrentPatient: (id: string | null) => void;
  setCurrentConsultation: (id: string | null) => void;
  setAnalysis: (data: Partial<ChatResponseData> | null) => void;
  enqueue: (msg: QueuedMessage) => void;
  dequeue: (id: string) => void;
  clearQueue: () => void;
  reset: () => void;
};

const defaultAnalysis = {
  findings: [] as ChatFinding[],
  differential: [] as ChatDifferential[],
  actions: [] as ChatAction[],
  triageLevel: "LOW",
  triageColor: "green",
  confidence: 0,
};

function genId() {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isConnected: false,
  isLoading: false,
  statusMessage: null,
  currentPatientId: null,
  currentConsultationId: null,
  ...defaultAnalysis,
  queue: [],

  addMessage: (msg) => {
    const full: ChatMessage = {
      ...msg,
      id: genId(),
      timestamp: new Date().toISOString(),
    };
    set((s) => ({ messages: [...s.messages, full] }));
    return full;
  },

  updateMessage: (id, update) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...update } : m)),
    }));
  },

  setConnection: (isConnected) => set({ isConnected }),
  setLoading: (isLoading) => set({ isLoading }),
  setStatusMessage: (statusMessage) => set({ statusMessage }),
  setCurrentPatient: (currentPatientId) => set({ currentPatientId }),
  setCurrentConsultation: (currentConsultationId) => set({ currentConsultationId }),

  setAnalysis: (data) => {
    if (!data) {
      set(defaultAnalysis);
      return;
    }
    set({
      findings: data.findings ?? get().findings,
      differential: data.differential_diagnosis ?? get().differential,
      actions: data.recommended_actions ?? get().actions,
      triageLevel: data.triage_level ?? get().triageLevel,
      triageColor: data.triage_color ?? get().triageColor,
      confidence: data.confidence ?? get().confidence,
    });
  },

  enqueue: (msg) => set((s) => ({ queue: [...s.queue, msg] })),
  dequeue: (id) => set((s) => ({ queue: s.queue.filter((m) => m.id !== id) })),
  clearQueue: () => set({ queue: [] }),

  reset: () =>
    set({
      messages: [],
      isLoading: false,
      statusMessage: null,
      ...defaultAnalysis,
      queue: [],
    }),
}));
