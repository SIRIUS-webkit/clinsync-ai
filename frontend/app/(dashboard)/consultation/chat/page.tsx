"use client";

import { ConsultationAIPanel } from "@/components/consultation/ConsultationAIPanel";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { ChatSocketProvider } from "@/components/chat/ChatSocketProvider";

export default function ConsultationChatPage() {
  return (
    <ChatSocketProvider>
      <div className="flex h-full min-h-0 flex-1 flex-col lg:flex-row">
        {/* Chat column — fills remaining space, height constrained */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <ChatInterface />
        </div>

        {/* AI analysis panel — side panel on large screens */}
        <ConsultationAIPanel mode="chat" className="hidden lg:flex" />
      </div>
    </ChatSocketProvider>
  );
}
