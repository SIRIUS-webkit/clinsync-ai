import { create } from "zustand";

type UIState = {
  sidebarExpanded: boolean;
  aiPanelCollapsed: boolean;
  setSidebarExpanded: (expanded: boolean) => void;
  setAIPanelCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  toggleAIPanel: () => void;
};

export const useUIStore = create<UIState>((set) => ({
  sidebarExpanded: true,
  aiPanelCollapsed: false,
  setSidebarExpanded: (sidebarExpanded) => set({ sidebarExpanded }),
  setAIPanelCollapsed: (aiPanelCollapsed) => set({ aiPanelCollapsed }),
  toggleSidebar: () => set((s) => ({ sidebarExpanded: !s.sidebarExpanded })),
  toggleAIPanel: () => set((s) => ({ aiPanelCollapsed: !s.aiPanelCollapsed })),
}));
