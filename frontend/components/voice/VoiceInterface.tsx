"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Mic, MicOff, Radio } from "lucide-react";
import toast from "react-hot-toast";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { analyzeAudio } from "@/lib/api";
import { convertToWavFile } from "@/lib/audio";
import { useAnalysisStore } from "@/stores/useAnalysisStore";

const defaultBiomarkerIndicators = [
  { label: "Cough detected", confidence: "High" },
  { label: "Wheezing", confidence: "85% confidence" },
  { label: "Shortness of breath", confidence: "Moderate" },
];

export function VoiceInterface() {
  const [listening, setListening] = React.useState(false);
  const [waveform, setWaveform] = React.useState<number[]>(Array.from({ length: 24 }, () => 18));
  const [transcript, setTranscript] = React.useState(
    "Transcription will appear here once voice capture begins."
  );
  const [biomarkers, setBiomarkers] = React.useState(defaultBiomarkerIndicators);
  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null);
  const audioChunksRef = React.useRef<Blob[]>([]);
  const mediaStreamRef = React.useRef<MediaStream | null>(null);
  const setFromResponse = useAnalysisStore((state) => state.setFromResponse);

  React.useEffect(() => {
    if (!listening) return;
    const interval = window.setInterval(() => {
      setWaveform((prev) => prev.map(() => Math.floor(12 + Math.random() * 52)));
    }, 120);
    return () => window.clearInterval(interval);
  }, [listening]);

  const toggleListening = async () => {
    if (listening) {
      mediaRecorderRef.current?.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      setListening(false);
      return;
    }

    try {
      if (typeof MediaRecorder === "undefined") {
        toast.error("Recording is not supported in this browser.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        setTranscript("Analyzing audio...");
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        let audioFile = new File([audioBlob], "voice-session.webm", { type: "audio/webm" });
        try {
          audioFile = await convertToWavFile(audioBlob, "voice-session.wav");
        } catch (error) {
          // Fallback to webm if wav conversion fails.
        }
        try {
          const response = await analyzeAudio({ audio: audioFile });
          setFromResponse(response);
          setTranscript(
            response.transcript || response.response || "No transcript detected from audio."
          );
          if (response.raw_results?.audio) {
            const entries = Object.entries(response.raw_results.audio)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([label, value]) => ({
                label: label.replace(/_/g, " "),
                confidence: `${Math.round(value * 100)}% confidence`,
              }));
            if (entries.length > 0) setBiomarkers(entries);
          }
          toast.success("Audio analysis complete.");
        } catch (error) {
          toast.error(error instanceof Error ? error.message : "Audio analysis failed.");
          setTranscript("Audio analysis failed. Please try again.");
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setListening(true);
      toast.success("Listening for patient voice...");
    } catch (error) {
      toast.error("Microphone permission denied.");
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Voice Only Mode</CardTitle>
        <Badge variant={listening ? "success" : "secondary"}>
          {listening ? "Listening" : "Idle"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="rounded-lg border border-border bg-muted/40 p-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Real-time waveform</span>
            <Radio className="h-4 w-4" aria-hidden />
          </div>
          <div className="mt-4 flex items-end justify-center gap-1">
            {waveform.map((height, index) => (
              <motion.span
                key={index}
                animate={{ height }}
                className="w-1 rounded-full bg-primary"
              />
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-semibold">Live transcription</h4>
          <div className="rounded-md border border-border bg-background p-3 text-sm text-muted-foreground">
            {listening ? "Listening..." : transcript}
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-semibold">Biomarker indicators</h4>
          <div className="grid gap-2 md:grid-cols-3">
            {biomarkers.map((item) => (
              <div key={item.label} className="rounded-md border border-border bg-muted/30 p-3">
                <p className="text-sm font-medium">{item.label}</p>
                <p className="text-xs text-muted-foreground">{item.confidence}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={toggleListening} aria-label="Toggle voice capture">
            {listening ? <MicOff className="mr-2 h-4 w-4" aria-hidden /> : <Mic className="mr-2 h-4 w-4" aria-hidden />}
            {listening ? "Stop Listening" : "Push-to-Talk"}
          </Button>
          <Button variant="outline" aria-label="Enable voice activated recording">
            Voice-activated mode
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
