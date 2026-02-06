"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Error({ reset }: { reset: () => void }) {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <Card className="max-w-md">
        <CardHeader>
          <CardTitle>Something went wrong</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            The ClinSync AI dashboard hit an unexpected error. Please try again.
          </p>
          <Button onClick={reset}>Retry</Button>
        </CardContent>
      </Card>
    </div>
  );
}
