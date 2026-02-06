"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AnalysisPanel } from "@/components/ai/AnalysisPanel";

export default function InsightsPage() {
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_400px]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">AI Analytics</CardTitle>
          <p className="text-sm text-muted-foreground">
            Longitudinal analytics, population trends, and audit-ready exports.
          </p>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Analytics dashboard content will appear here.
        </CardContent>
      </Card>
      <AnalysisPanel />
    </div>
  );
}
