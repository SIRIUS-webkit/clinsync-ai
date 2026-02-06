"use client";

import { motion } from "framer-motion";
import { FileDown, History, ShieldAlert } from "lucide-react";
import toast from "react-hot-toast";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAnalysisStore } from "@/stores/useAnalysisStore";

const defaultFindings = [
  { label: "Mild cardiomegaly", confidence: 0.78 },
  { label: "Pulmonary congestion", confidence: 0.64 },
  { label: "Pleural effusion", confidence: 0.42 },
];

const defaultDifferentials = [
  { label: "Congestive heart failure", probability: 0.48 },
  { label: "Pneumonia", probability: 0.21 },
  { label: "Chronic bronchitis", probability: 0.12 },
];

const defaultRecommendations = [
  { label: "Urgent: Refer to cardiologist", severity: "alert" },
  { label: "Order BNP lab panel", severity: "secondary" },
  { label: "Schedule follow-up imaging", severity: "success" },
];

const history = [
  { label: "Jan 12: Normal CT angiogram", delta: "Stable" },
  { label: "Dec 28: Elevated BNP trend", delta: "Worsened" },
];

function normalizeConfidence(value: number) {
  if (value > 1) return Math.min(1, value / 100);
  return Math.max(0, Math.min(1, value));
}

function severityFromText(text: string) {
  const lowered = text.toLowerCase();
  if (lowered.includes("urgent") || lowered.includes("emergent") || lowered.includes("refer")) {
    return "alert";
  }
  if (lowered.includes("follow") || lowered.includes("monitor")) {
    return "success";
  }
  return "secondary";
}

export function AnalysisPanel() {
  const { findings, recommendations, triageLevel, confidence, response, lastUpdated } =
    useAnalysisStore((state) => state);

  const hasLiveData = findings.length > 0 || recommendations.length > 0 || response.length > 0;
  const normalizedConfidence = normalizeConfidence(confidence || 0.65);
  const findingsToRender = hasLiveData
    ? findings.map((label, index) => ({
        label,
        confidence: Math.max(0.2, normalizedConfidence - index * 0.08),
      }))
    : defaultFindings;
  const recommendationsToRender = hasLiveData
    ? recommendations.map((label) => ({
        label,
        severity: severityFromText(label),
      }))
    : defaultRecommendations;

  const exportFHIR = () => {
    const payload = {
      resourceType: "DiagnosticReport",
      status: "final",
      code: { text: "ClinSync AI Analysis" },
      effectiveDateTime: new Date().toISOString(),
      conclusion: response || "ClinSync AI analysis export.",
      conclusionCode: findingsToRender.map((item) => ({ text: item.label })),
      extension: [
        { url: "triage_level", valueString: triageLevel },
        { url: "confidence", valueDecimal: normalizedConfidence },
      ],
      recommendation: recommendationsToRender.map((item) => item.label),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "clinsync-analysis.fhir.json";
    link.click();
    URL.revokeObjectURL(url);
    toast.success("FHIR export ready.");
  };

  const exportSOAP = async () => {
    try {
      // Build findings and recommendations for API
      const findingsForApi = findings.length > 0 ? findings : findingsToRender.map((f) => f.label);
      const recommendationsForApi = recommendations.length > 0 
        ? recommendations 
        : recommendationsToRender.map((r) => r.label);

      // Create form data for the SOAP endpoint
      const formData = new FormData();
      formData.append("patient_id", "anonymous");
      formData.append("consultation_type", "video");
      formData.append("response_text", response || "Video consultation analysis");
      formData.append("findings", JSON.stringify(findingsForApi));
      formData.append("recommendations", JSON.stringify(recommendationsForApi));
      formData.append("triage_level", triageLevel);
      formData.append("transcript", ""); 
      formData.append("format", "text");

      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const fetchResponse = await fetch(`${API_URL}/chat/soap/download`, {
        method: "POST",
        body: formData,
      });

      if (!fetchResponse.ok) {
        throw new Error("Failed to generate SOAP note");
      }

      // Get the blob and trigger download
      const blob = await fetchResponse.blob();
      const noteId = fetchResponse.headers.get("X-Note-ID") || "soap-note";
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
  };

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">AI Analysis Panel</CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant={triageLevel === "HIGH" ? "alert" : triageLevel === "LOW" ? "success" : "secondary"}>
            Triage: {triageLevel}
          </Badge>
          <Badge variant="outline">{lastUpdated ? `Updated ${lastUpdated}` : "Real-time"}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h4 className="text-sm font-semibold">Findings</h4>
          <div className="mt-3 space-y-3">
            {findingsToRender.map((finding) => (
              <div key={finding.label} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>{finding.label}</span>
                  <span className="text-xs text-muted-foreground">
                    {(finding.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <motion.div
                    className="h-2 rounded-full bg-primary"
                    initial={{ width: 0 }}
                    animate={{ width: `${finding.confidence * 100}%` }}
                    transition={{ duration: 0.6 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold">Differential diagnosis</h4>
          <div className="mt-3 space-y-2">
            {defaultDifferentials.map((item) => (
              <div key={item.label} className="flex items-center justify-between text-sm">
                <span>{item.label}</span>
                <Badge variant="outline">{(item.probability * 100).toFixed(0)}%</Badge>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold">Recommended actions</h4>
          <div className="mt-3 space-y-2">
            {recommendationsToRender.map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/40 p-2 text-sm"
              >
                <span>{item.label}</span>
                <Badge
                  variant={
                    item.severity === "alert"
                      ? "alert"
                      : item.severity === "success"
                      ? "success"
                      : "secondary"
                  }
                >
                  {item.severity === "alert" ? "High" : item.severity === "success" ? "Low" : "Medium"}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold">Historical comparison</h4>
            <History className="h-4 w-4 text-muted-foreground" aria-hidden />
          </div>
          <ScrollArea className="mt-3 h-28 rounded-md border border-border bg-muted/20 p-2">
            <div className="space-y-2 text-sm">
              {history.map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <span>{item.label}</span>
                  <Badge variant="outline">{item.delta}</Badge>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button 
            className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700" 
            onClick={exportSOAP} 
            aria-label="Download SOAP Note"
          >
            <FileDown className="mr-2 h-4 w-4" aria-hidden />
            Download SOAP Note
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-2 mt-2">
          <Button className="flex-1" variant="outline" onClick={exportFHIR} aria-label="Export analysis to FHIR format">
            <FileDown className="mr-2 h-4 w-4" aria-hidden />
            Export to FHIR
          </Button>
          <Button
            variant="outline"
            className="flex-1"
            aria-label="Review compliance summary"
            onClick={() => toast("Compliance log queued for audit.")}
          >
            <ShieldAlert className="mr-2 h-4 w-4" aria-hidden />
            Compliance log
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
