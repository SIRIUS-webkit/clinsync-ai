"use client";

import * as React from "react";

import { ThemeProvider } from "@/components/providers/theme-provider";
import { ToasterProvider } from "@/components/providers/toaster-provider";
import { registerServiceWorker } from "@/lib/service-worker";
import { useApiHealth } from "@/hooks/useApiHealth";
import { useOfflineQueue } from "@/hooks/useOfflineQueue";
import { useOnlineStatus } from "@/hooks/useOnlineStatus";
import { ChatSocketProvider } from "@/components/chat/ChatSocketProvider";

export function AppProviders({ children }: { children: React.ReactNode }) {
  useOnlineStatus();
  useApiHealth();
  useOfflineQueue();

  React.useEffect(() => {
    registerServiceWorker();
  }, []);

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <ChatSocketProvider>
        {children}
        <ToasterProvider />
      </ChatSocketProvider>
    </ThemeProvider>
  );
}
