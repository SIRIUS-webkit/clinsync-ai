import { create } from "zustand";

export type ConsultationMode = "video" | "chat" | "voice";

type ConsultationState = {
  currentMode: ConsultationMode;
  patientId: string | null;
  findings: string[];
  isAICapturing: boolean;
  setMode: (mode: ConsultationMode) => void;
  setPatientId: (id: string | null) => void;
  setFindings: (findings: string[]) => void;
  addFinding: (finding: string) => void;
  setAICapturing: (capturing: boolean) => void;
  reset: () => void;
};

const initialState = {
  currentMode: "video" as ConsultationMode,
  patientId: null,
  findings: [],
  isAICapturing: false,
};

export const useConsultationStore = create<ConsultationState>((set) => ({
  ...initialState,
  setMode: (currentMode) => set({ currentMode }),
  setPatientId: (patientId) => set({ patientId }),
  setFindings: (findings) => set({ findings }),
  addFinding: (finding) => set((s) => ({ findings: [...s.findings, finding] })),
  setAICapturing: (isAICapturing) => set({ isAICapturing }),
  reset: () => set(initialState),
}));
