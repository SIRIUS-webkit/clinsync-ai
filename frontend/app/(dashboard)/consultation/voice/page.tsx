"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ConsultationAIPanel } from "@/components/consultation/ConsultationAIPanel";
import { VoiceInterface } from "@/components/voice/VoiceInterface";

export default function ConsultationVoicePage() {
  return (
    <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
      <div className="flex min-h-0 flex-1 basis-[60%] flex-col p-2 lg:p-4">
        <Card className="flex min-h-0 flex-1 flex-col">
          <CardContent className="flex min-h-0 flex-1 p-4">
            <div className="flex h-full min-h-[50vh] flex-col lg:min-h-0">
              <VoiceInterface />
            </div>
          </CardContent>
        </Card>
      </div>
      <ConsultationAIPanel mode="voice" className="hidden lg:block" />
    </div>
  );
}
