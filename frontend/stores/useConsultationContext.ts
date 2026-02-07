import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface PatientIntake {
  patientId: string;
  fullName: string;
  age: string;
  gender: string;
  chiefComplaint: string;
  symptoms: string[];
  symptomDuration: string;
  painLevel: number;
  allergies: string;
  currentMedications: string;
  medicalHistory: string;
  attachedImages: {
    id: string;
    name: string;
    type: string; // "skin" | "radiology" | "wound" | "other"
    dataUrl: string;
  }[];
  consultationType: "general" | "skin" | "respiratory" | "cardiology" | "other";
  createdAt: string;
}

interface ConsultationContextState {
  intake: PatientIntake | null;
  sessionActive: boolean;
  conversationHistory: { role: "user" | "ai"; message: string; timestamp: string }[];
  
  // Actions
  setIntake: (intake: PatientIntake) => void;
  clearIntake: () => void;
  startSession: () => void;
  endSession: () => void;
  addMessage: (role: "user" | "ai", message: string) => void;
  clearHistory: () => void;
  getSystemPromptContext: () => string;
}

export const useConsultationContext = create<ConsultationContextState>()(
  persist(
    (set, get) => ({
      intake: null,
      sessionActive: false,
      conversationHistory: [],

      setIntake: (intake) => set({ intake }),
      
      clearIntake: () => set({ intake: null }),
      
      startSession: () => set({ sessionActive: true }),
      
      endSession: () => set({ sessionActive: false, conversationHistory: [] }),
      
      addMessage: (role, message) =>
        set((state) => ({
          conversationHistory: [
            ...state.conversationHistory,
            { role, message, timestamp: new Date().toISOString() },
          ],
        })),
      
      clearHistory: () => set({ conversationHistory: [] }),
      
      getSystemPromptContext: () => {
        const { intake } = get();
        if (!intake) return "";
        
        const lines = [
          "=== PATIENT CONTEXT ===",
          `Patient ID: ${intake.patientId}`,
          `Name: ${intake.fullName}`,
          `Age: ${intake.age} years`,
          `Gender: ${intake.gender}`,
          "",
          `Chief Complaint: ${intake.chiefComplaint}`,
          `Symptoms: ${intake.symptoms.join(", ") || "Not specified"}`,
          `Duration: ${intake.symptomDuration}`,
          `Pain Level: ${intake.painLevel}/10`,
          "",
          `Allergies: ${intake.allergies || "None reported"}`,
          `Current Medications: ${intake.currentMedications || "None"}`,
          `Medical History: ${intake.medicalHistory || "Not provided"}`,
          "",
          `Consultation Type: ${intake.consultationType}`,
          `Images Attached: ${intake.attachedImages.length} (${intake.attachedImages.map(i => i.type).join(", ") || "none"})`,
          "=== END CONTEXT ===",
        ];
        
        return lines.join("\n");
      },
    }),
    {
      name: "consultation-context",
      partialize: (state) => ({
        intake: state.intake,
        conversationHistory: state.conversationHistory,
      }),
    }
  )
);

// Generate unique patient ID
export function generatePatientId(): string {
  const date = new Date();
  const dateStr = date.toISOString().slice(0, 10).replace(/-/g, "");
  const random = Math.random().toString(36).slice(2, 6).toUpperCase();
  return `PT-${dateStr}-${random}`;
}
