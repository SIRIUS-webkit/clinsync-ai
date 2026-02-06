"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import toast from "react-hot-toast";

import { AnalysisPanel } from "@/components/ai/AnalysisPanel";
import { AIPanel } from "@/components/chat/AIPanel";
import { Button } from "@/components/ui/button";
import { useChatStore } from "@/stores/chatStore";
import { useChatSocket } from "@/hooks/useChatSocket";
import { useUIStore } from "@/stores/useUIStore";
import { cn } from "@/lib/utils";

export type ConsultationPanelMode = "video" | "chat" | "voice";

type ConsultationAIPanelProps = {
  mode: ConsultationPanelMode;
  className?: string;
};

const PANEL_WIDTH = 320;

export function ConsultationAIPanel({ mode, className }: ConsultationAIPanelProps) {
  const aiPanelCollapsed = useUIStore((s) => s.aiPanelCollapsed);
  const toggleAIPanel = useUIStore((s) => s.toggleAIPanel);
  const chatFindings = useChatStore((s) => s.findings);
  const chatDifferential = useChatStore((s) => s.differential);
  const chatActions = useChatStore((s) => s.actions);
  const triageLevel = useChatStore((s) => s.triageLevel);
  const triageColor = useChatStore((s) => s.triageColor);
  const confidence = useChatStore((s) => s.confidence);
  const { isConnected } = useChatSocket();

  const handleExportFHIR = React.useCallback(() => {
    const payload = {
      resourceType: "DiagnosticReport",
      status: "final",
      code: { text: "ClinSync AI Analysis" },
      effectiveDateTime: new Date().toISOString(),
      conclusionCode: chatFindings.map((f) => ({ display: f.label })),
      extension: [
        { url: "triage_level", valueString: triageLevel },
        { url: "confidence", valueDecimal: confidence },
      ],
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clinsync-analysis.fhir.json";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("FHIR export downloaded.");
  }, [chatFindings, triageLevel, confidence]);

  const handleExportSOAP = React.useCallback(async () => {
    try {
      // Get the last response text from messages
      const messages = useChatStore.getState().messages;
      const lastAssistantMessage = messages
        .filter((m) => m.role === "assistant")
        .pop();
      const responseText = lastAssistantMessage?.content || "";

      // Build findings for API
      const findingsForApi = chatFindings.map((f) => f.label);
      const recommendationsForApi = chatActions.map((a) => a.text);

      // Create form data for the SOAP endpoint
      const formData = new FormData();
      formData.append("patient_id", "anonymous");
      formData.append("consultation_type", "chat");
      formData.append("response_text", responseText);
      formData.append("findings", JSON.stringify(findingsForApi));
      formData.append("recommendations", JSON.stringify(recommendationsForApi));
      formData.append("triage_level", triageLevel);
      formData.append("transcript", ""); // Could store user messages
      formData.append("format", "text");

      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_URL}/chat/soap/download`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to generate SOAP note");
      }

      // Get the blob and trigger download
      const blob = await response.blob();
      const noteId = response.headers.get("X-Note-ID") || "soap-note";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${noteId}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success("SOAP note downloaded successfully!");
    } catch (error) {
      console.error("SOAP export failed:", error);
      toast.error("Failed to download SOAP note");
    }
  }, [chatFindings, chatActions, triageLevel]);

  return (
    <div
      className={cn(
        "relative flex shrink-0 flex-col border-l border-border bg-card transition-[width] duration-200",
        aiPanelCollapsed ? "w-12" : "w-[var(--ai-panel-width)]",
        className
      )}
      style={aiPanelCollapsed ? undefined : { ["--ai-panel-width" as string]: `${PANEL_WIDTH}px` }}
    >
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleAIPanel}
        className="absolute left-0 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-full border border-border bg-card shadow-md hover:bg-accent"
        aria-label={aiPanelCollapsed ? "Expand AI panel" : "Collapse AI panel"}
      >
        {aiPanelCollapsed ? (
          <ChevronRight className="h-4 w-4" aria-hidden />
        ) : (
          <ChevronLeft className="h-4 w-4" aria-hidden />
        )}
      </Button>
      {!aiPanelCollapsed && (
        <div className="h-full overflow-y-auto pl-2">
          {mode === "chat" ? (
            <AIPanel
              findings={chatFindings}
              differential={chatDifferential}
              actions={chatActions}
              triageLevel={triageLevel}
              triageColor={triageColor}
              confidence={confidence}
              isConnected={isConnected}
              onExportFHIR={handleExportFHIR}
              onExportSOAP={handleExportSOAP}
            />
          ) : (
            <AnalysisPanel />
          )}
        </div>
      )}
    </div>
  );
}
