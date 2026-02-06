"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { ChevronDown, ChevronUp, FileDown, History } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ChatFinding, ChatDifferential, ChatAction } from "@/lib/api";
import { cn } from "@/lib/utils";

type AIPanelProps = {
  findings: ChatFinding[];
  differential: ChatDifferential[];
  actions: ChatAction[];
  triageLevel: string;
  triageColor: string;
  confidence: number;
  isConnected: boolean;
  onExportFHIR?: () => void;
  onExportSOAP?: () => void;
  transcript?: string;
};

const triageColors: Record<string, string> = {
  green: "bg-green-600 text-white",
  yellow: "bg-yellow-600 text-white",
  orange: "bg-orange-600 text-white",
  red: "bg-red-600 text-white",
  gray: "bg-muted text-muted-foreground",
};

const priorityColors: Record<string, string> = {
  red: "bg-red-600/90 text-white",
  yellow: "bg-yellow-600/90 text-white",
  green: "bg-green-600/90 text-white",
};

export function AIPanel({
  findings,
  differential,
  actions,
  triageLevel,
  triageColor,
  confidence,
  isConnected,
  onExportFHIR,
  onExportSOAP,
  transcript,
}: AIPanelProps) {
  const [findingsOpen, setFindingsOpen] = React.useState(true);
  const [differentialOpen, setDifferentialOpen] = React.useState(true);
  const [actionsOpen, setActionsOpen] = React.useState(true);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="shrink-0 space-y-2 border-b border-border pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">AI Analysis</CardTitle>
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              isConnected ? "bg-green-500" : "bg-muted-foreground"
            )}
            title={isConnected ? "Connected" : "Disconnected"}
          />
        </div>
        <div className="flex items-center gap-2">
          <Badge
            className={cn(
              "font-medium",
              triageColors[triageColor] ?? triageColors.gray
            )}
          >
            Triage: {triageLevel}
          </Badge>
          <span className="text-xs text-muted-foreground">
            Confidence {Math.round(confidence * 100)}%
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 pt-4">
        <div>
          <button
            type="button"
            className="flex w-full items-center justify-between text-sm font-semibold"
            onClick={() => setFindingsOpen(!findingsOpen)}
          >
            Findings
            {findingsOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          {findingsOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-2 space-y-2"
            >
              {findings.length === 0 ? (
                <p className="text-xs text-muted-foreground">No findings yet.</p>
              ) : (
                findings.map((f) => (
                  <div key={f.id} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>{f.label}</span>
                      <span className="text-muted-foreground">{f.confidence_label}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <motion.div
                        className="h-full rounded-full bg-primary"
                        initial={{ width: 0 }}
                        animate={{ width: `${f.confidence}%` }}
                        transition={{ duration: 0.4 }}
                      />
                    </div>
                  </div>
                ))
              )}
            </motion.div>
          )}
        </div>

        <div>
          <button
            type="button"
            className="flex w-full items-center justify-between text-sm font-semibold"
            onClick={() => setDifferentialOpen(!differentialOpen)}
          >
            Differential diagnosis
            {differentialOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          {differentialOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-2 space-y-2"
            >
              {differential.length === 0 ? (
                <p className="text-xs text-muted-foreground">None yet.</p>
              ) : (
                differential.map((dx) => (
                  <div key={dx.id} className="flex items-center justify-between rounded-md border border-border bg-muted/20 px-2 py-1.5 text-sm">
                    <span>{dx.condition}</span>
                    <Badge variant="outline">{dx.probability_label}</Badge>
                  </div>
                ))
              )}
            </motion.div>
          )}
        </div>

        <div>
          <button
            type="button"
            className="flex w-full items-center justify-between text-sm font-semibold"
            onClick={() => setActionsOpen(!actionsOpen)}
          >
            Recommended actions
            {actionsOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          {actionsOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-2 space-y-2"
            >
              {actions.length === 0 ? (
                <p className="text-xs text-muted-foreground">None yet.</p>
              ) : (
                actions.map((a) => (
                  <div
                    key={a.id}
                    className="flex items-center justify-between gap-2 rounded-md border border-border bg-muted/20 p-2 text-sm"
                  >
                    <span>{a.text}</span>
                    <Badge
                      className={cn(
                        "shrink-0",
                        priorityColors[a.priority_color] ?? "bg-muted text-muted-foreground"
                      )}
                    >
                      {a.priority_label}
                    </Badge>
                  </div>
                ))
              )}
            </motion.div>
          )}
        </div>

        <ScrollArea className="min-h-0 flex-1" />

        <div className="flex flex-col gap-2 border-t border-border pt-4">
          <Button
            variant="default"
            size="sm"
            className="w-full justify-start bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700"
            onClick={onExportSOAP}
          >
            <FileDown className="mr-2 h-4 w-4" />
            Download SOAP Note
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start"
            onClick={onExportFHIR}
          >
            <FileDown className="mr-2 h-4 w-4" />
            Export to FHIR
          </Button>
          <Button variant="ghost" size="sm" className="w-full justify-start text-muted-foreground">
            <History className="mr-2 h-4 w-4" />
            Historical comparison
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
