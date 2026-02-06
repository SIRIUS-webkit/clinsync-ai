import { create } from "zustand";

type ConnectionState = {
  online: boolean;
  edgeMode: boolean;
  queueCount: number;
  lastSync: string | null;
  apiOnline: boolean | null;
  setOnline: (online: boolean) => void;
  setEdgeMode: (edgeMode: boolean) => void;
  setQueueCount: (count: number) => void;
  setLastSync: (timestamp: string | null) => void;
  setApiOnline: (apiOnline: boolean) => void;
};

export const useConnectionStore = create<ConnectionState>((set) => ({
  online: true,
  edgeMode: false,
  queueCount: 0,
  lastSync: null,
  apiOnline: null,
  setOnline: (online) => set({ online, edgeMode: !online }),
  setEdgeMode: (edgeMode) => set({ edgeMode }),
  setQueueCount: (queueCount) => set({ queueCount }),
  setLastSync: (lastSync) => set({ lastSync }),
  setApiOnline: (apiOnline) => set({ apiOnline }),
}));
