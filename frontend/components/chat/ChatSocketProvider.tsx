"use client";

import * as React from "react";
import { chatHttpFallback, healthCheck, type ChatResponseData } from "@/lib/api";
import { useChatStore } from "@/stores/chatStore";

type ChatSocketContextValue = {
  send: (payload: {
    text?: string;
    image?: string;
    audio?: string;
    patient_id?: string;
    consultation_id?: string;
  }) => void;
  isConnected: boolean;
  lastMessage: ChatResponseData | null;
  error: string | null;
  statusMessage: string | null;
};

export const ChatSocketContext =
  React.createContext<ChatSocketContextValue | null>(null);

/* ------------------------------------------------------------------ */
/*  Provider – HTTP-based chat (no WebSocket)                          */
/* ------------------------------------------------------------------ */

const HEALTH_CHECK_INTERVAL_MS = 30_000;

export function ChatSocketProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isConnected, setIsConnected] = React.useState(false);
  const [lastMessage, setLastMessage] = React.useState<ChatResponseData | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [statusMessage, setStatusMsg] = React.useState<string | null>(null);
  const isSendingRef = React.useRef(false);

  /* ---- periodic health check to show connection status ---- */
  React.useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        await healthCheck();
        if (!cancelled) setIsConnected(true);
      } catch {
        if (!cancelled) setIsConnected(false);
      }
    };

    check();
    const id = setInterval(check, HEALTH_CHECK_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  /* ---- send a message via POST /chat/ ---- */
  const send = React.useCallback(
    (payload: {
      text?: string;
      image?: string;
      audio?: string;
      patient_id?: string;
      consultation_id?: string;
    }) => {
      if (!(payload.text || payload.image || payload.audio)) return;
      if (isSendingRef.current) return;
      isSendingRef.current = true;

      const store = useChatStore.getState();
      const patient_id =
        payload.patient_id ?? store.currentPatientId ?? undefined;
      const consultation_id =
        payload.consultation_id ?? store.currentConsultationId ?? undefined;

      // Add the user bubble immediately
      store.addMessage({
        role: "user",
        content:
          payload.text ||
          (payload.image ? "Sent an image" : "Sent voice message"),
        imageDataUrl: payload.image,
        status: "sending",
      });

      store.setLoading(true);
      setStatusMsg("Processing...");
      setError(null);

      chatHttpFallback({
        text: payload.text,
        image: payload.image,
        audio: payload.audio,
        patient_id,
        consultation_id,
      })
        .then((data) => {
          isSendingRef.current = false;
          setStatusMsg(null);
          setLastMessage(data);
          setIsConnected(true);

          const s = useChatStore.getState();
          s.setLoading(false);
          s.setAnalysis(data);

          // Mark pending user message as sent
          const pending = [...s.messages]
            .reverse()
            .find((m) => m.role === "user" && m.status === "sending");
          if (pending) s.updateMessage(pending.id, { status: "sent" });

          // Add exactly one assistant bubble
          s.addMessage({
            role: "assistant",
            content: data.response_text,
            status: "sent",
            responseData: data,
          });
        })
        .catch((err) => {
          isSendingRef.current = false;
          setStatusMsg(null);
          const errMsg = err?.message ?? "Request failed. Please try again.";
          setError(errMsg);

          const s = useChatStore.getState();
          s.setLoading(false);
          s.messages.forEach((m) => {
            if (m.status === "sending")
              s.updateMessage(m.id, { status: "error", error: errMsg });
          });
        });
    },
    [],
  );

  /* ---- context value ---- */
  const value = React.useMemo<ChatSocketContextValue>(
    () => ({ send, isConnected, lastMessage, error, statusMessage }),
    [send, isConnected, lastMessage, error, statusMessage],
  );

  return (
    <ChatSocketContext.Provider value={value}>
      {children}
    </ChatSocketContext.Provider>
  );
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

export function useChatSocketContext(): ChatSocketContextValue {
  const ctx = React.useContext(ChatSocketContext);
  if (!ctx)
    throw new Error("useChatSocket must be used within ChatSocketProvider");
  return ctx;
}

export type UseChatSocketReturn = ChatSocketContextValue;
