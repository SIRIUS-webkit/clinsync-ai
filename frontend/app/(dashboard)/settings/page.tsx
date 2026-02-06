"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-1">Configure clinician profiles and system preferences.</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Application</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Notification preferences, integrations, and theme options.
        </CardContent>
      </Card>
    </div>
  );
}
