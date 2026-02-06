import { create } from "zustand";

export type Patient = {
  id: string;
  name: string;
  age: number;
  symptoms?: string;
  history?: string;
  lastConsult?: string;
};

type PatientState = {
  patientList: Patient[];
  selectedPatient: Patient | null;
  setPatientList: (list: Patient[]) => void;
  setSelectedPatient: (patient: Patient | null) => void;
  addPatient: (patient: Patient) => void;
};

export const usePatientStore = create<PatientState>((set) => ({
  patientList: [
    { id: "1", name: "Jane Doe", age: 54, symptoms: "Chest tightness, fatigue", lastConsult: "2025-02-04" },
    { id: "2", name: "John Smith", age: 62, symptoms: "Shortness of breath", lastConsult: "2025-02-03" },
  ],
  selectedPatient: null,
  setPatientList: (patientList) => set({ patientList }),
  setSelectedPatient: (selectedPatient) => set({ selectedPatient }),
  addPatient: (patient) => set((s) => ({ patientList: [patient, ...s.patientList] })),
}));
