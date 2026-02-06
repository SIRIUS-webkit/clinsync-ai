"use client";

import { motion } from "framer-motion";
import { CloudUpload } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useConnectionStore } from "@/stores/useConnectionStore";

export function OfflineSyncIndicator() {
  const { online, queueCount } = useConnectionStore((state) => state);

  return (
    <Card className="border-dashed">
      <CardContent className="flex items-center justify-between gap-4 p-4">
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ opacity: online ? 0.7 : 1 }}
            transition={{ duration: 0.4 }}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-muted"
          >
            <CloudUpload className="h-5 w-5 text-muted-foreground" aria-hidden />
          </motion.div>
          <div>
            <p className="text-sm font-medium">Sync when online</p>
            <p className="text-xs text-muted-foreground">
              {online ? "Queue will flush automatically." : "Offline storage is active."}
            </p>
          </div>
        </div>
        <Badge variant={queueCount > 0 ? "alert" : "secondary"}>{queueCount} pending</Badge>
      </CardContent>
    </Card>
  );
}
