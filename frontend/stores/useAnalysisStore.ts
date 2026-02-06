import { create } from "zustand";

import type { AIResponse } from "@/lib/api";

type AnalysisState = {
  findings: string[];
  recommendations: string[];
  triageLevel: string;
  confidence: number;
  response: string;
  lastUpdated: string | null;
  setFromResponse: (data: AIResponse) => void;
  clear: () => void;
};

const initialState = {
  findings: [],
  recommendations: [],
  triageLevel: "LOW",
  confidence: 0,
  response: "",
  lastUpdated: null,
};

export const useAnalysisStore = create<AnalysisState>((set) => ({
  ...initialState,
  setFromResponse: (data) => {
    const findings = Array.isArray(data.findings)
      ? data.findings
      : data.findings
      ? [data.findings]
      : [];
    const recommendations = Array.isArray(data.recommendations)
      ? data.recommendations
      : data.recommendations
      ? [data.recommendations]
      : [];
    set({
      findings: findings.map((f: any) => typeof f === 'string' ? f : f?.label || "Unknown finding"),
      recommendations,
      triageLevel: data.triage_level || "LOW",
      confidence: typeof data.confidence === "number" ? data.confidence : 0,
      response: data.response || (data as any).response_text || "",
      lastUpdated: new Date().toLocaleTimeString(),
    });
  },
  clear: () => set({ ...initialState }),
}));
