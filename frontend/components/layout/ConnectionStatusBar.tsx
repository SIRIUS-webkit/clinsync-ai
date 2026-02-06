"use client";

import { motion } from "framer-motion";
import { Wifi, WifiOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { useConnectionStore } from "@/stores/useConnectionStore";

export function ConnectionStatusBar() {
  const { online, edgeMode, queueCount, lastSync, apiOnline } = useConnectionStore((state) => state);

  return (
    <div className="border-b border-border bg-card/80 px-4 py-2 text-sm backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <motion.span
            key={online ? "online" : "offline"}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2"
          >
            {online ? (
              <Wifi className="h-4 w-4 text-success" aria-hidden />
            ) : (
              <WifiOff className="h-4 w-4 text-alert" aria-hidden />
            )}
            <span className="font-medium">{online ? "Online" : "Offline"}</span>
          </motion.span>
          {edgeMode && (
            <Badge variant="outline" className="border-medical-blue text-medical-blue">
              Edge AI Mode - Local Processing
            </Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Badge variant={apiOnline ? "success" : apiOnline === false ? "alert" : "secondary"}>
            API: {apiOnline ? "Online" : apiOnline === false ? "Offline" : "Checking"}
          </Badge>
          <Badge variant={queueCount > 0 ? "alert" : "secondary"}>
            Sync Queue: {queueCount}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {lastSync ? `Last sync ${lastSync}` : "No sync yet"}
          </span>
        </div>
      </div>
    </div>
  );
}
