"use client";

import * as React from "react";
import { useChatStore } from "@/stores/chatStore";
import { useChatSocket } from "@/hooks/useChatSocket";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";

export function ChatInterface() {
  const { send, isConnected, error, statusMessage } = useChatSocket();
  const addMessage = useChatStore((s) => s.addMessage);
  const isLoading = useChatStore((s) => s.isLoading);

  React.useEffect(() => {
    const messages = useChatStore.getState().messages;
    if (messages.length === 0) {
      addMessage({
        role: "assistant",
        content:
          "Hello — ready to assist. Send a message or upload an image to start.",
      });
      addMessage({
        role: "system",
        content: "Dr. Rivera joined the consultation.",
      });
    }
  }, []);

  const handleSend = React.useCallback(
    (payload: { text?: string; image?: string; audio?: string }) => {
      send({
        text: payload.text,
        image: payload.image,
        audio: payload.audio,
      });
    },
    [send],
  );

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      {/* Status banners — always visible at top */}
      {!isConnected && (
        <div className="shrink-0 border-b border-yellow-600/30 bg-yellow-600/10 px-4 py-1.5 text-center text-xs text-yellow-800 dark:text-yellow-200">
          Reconnecting…
        </div>
      )}
      {error && (
        <div className="shrink-0 border-b border-destructive/30 bg-destructive/10 px-4 py-1.5 text-center text-xs text-destructive">
          {error}
        </div>
      )}
      {statusMessage && isConnected && (
        <div className="shrink-0 border-b border-border bg-muted/30 px-4 py-1 text-center text-[11px] text-muted-foreground">
          {statusMessage}
        </div>
      )}

      {/* Messages — takes all remaining space, scrolls internally */}
      <MessageList />

      {/* Input — always pinned at bottom, never moves */}
      <ChatInput
        onSend={handleSend}
        disabled={isLoading}
        placeholder="Message ClinSync AI... (Enter to send, Shift+Enter for new line)"
      />
    </div>
  );
}
