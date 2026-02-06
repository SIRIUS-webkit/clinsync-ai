"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import SimplePeer, { Instance as SimplePeerInstance } from "simple-peer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { sendChat, sendWebrtcAnswer, sendWebrtcOffer } from "@/lib/api";
import { cn } from "@/lib/utils";
import { enqueueSync, saveConsultation } from "@/lib/offline-db";
import { useAnalysisStore } from "@/stores/useAnalysisStore";
import { useConnectionStore } from "@/stores/useConnectionStore";
import toast from "react-hot-toast";
import {
  Copy,
  Mic,
  MicOff,
  PhoneOff,
  Video,
  VideoOff,
  Wand2,
} from "lucide-react";

type Phase = "lobby" | "in-call" | "post";

const overlayFindings = [
  { label: "Stable HR", position: "top-6 left-6" },
  { label: "O2 96%", position: "top-6 right-6" },
  { label: "AI: Mild dyspnea", position: "bottom-8 left-10" },
];

export default function ConsultationRoomPage({ params }: { params: { roomId: string } }) {
  const [phase, setPhase] = React.useState<Phase>("lobby");
  const [localStream, setLocalStream] = React.useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = React.useState<MediaStream | null>(null);
  const [blurBackground, setBlurBackground] = React.useState(false);
  const [offerSignal, setOfferSignal] = React.useState("");
  const [remoteSignal, setRemoteSignal] = React.useState("");
  const [connectionStatus, setConnectionStatus] = React.useState("disconnected");
  const [muted, setMuted] = React.useState(false);
  const [cameraOff, setCameraOff] = React.useState(false);
  const peerRef = React.useRef<SimplePeerInstance | null>(null);
  const peerIdRef = React.useRef(`clinician-${Math.random().toString(36).slice(2, 8)}`);
  const localVideoRef = React.useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = React.useRef<HTMLVideoElement | null>(null);
  const online = useConnectionStore((state) => state.online);
  const setFromResponse = useAnalysisStore((state) => state.setFromResponse);

  React.useEffect(() => {
    const setupMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        setLocalStream(stream);
      } catch (error) {
        toast.error("Camera/microphone access denied.");
      }
    };

    setupMedia();
    return () => {
      peerRef.current?.destroy();
    };
  }, []);

  React.useEffect(() => {
    return () => {
      localStream?.getTracks().forEach((track) => track.stop());
    };
  }, [localStream]);

  React.useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  React.useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const startPeer = () => {
    if (!localStream) return;
    peerRef.current?.destroy();
    const peer = new SimplePeer({ initiator: true, trickle: false, stream: localStream });
    peer.on("signal", (data) => {
      setOfferSignal(JSON.stringify(data));
      sendWebrtcOffer({
        room_id: params.roomId,
        peer_id: peerIdRef.current,
        sdp: data as Record<string, unknown>,
      }).catch(() => null);
    });
    peer.on("connect", () => setConnectionStatus("connected"));
    peer.on("close", () => setConnectionStatus("closed"));
    peer.on("error", () => setConnectionStatus("error"));
    peer.on("stream", (stream) => setRemoteStream(stream));
    peerRef.current = peer;
    setConnectionStatus("connecting");
  };

  const applyRemoteSignal = () => {
    if (!peerRef.current || !remoteSignal) return;
    try {
      const parsed = JSON.parse(remoteSignal);
      peerRef.current.signal(parsed);
      sendWebrtcAnswer({
        room_id: params.roomId,
        peer_id: "remote-peer",
        sdp: parsed as Record<string, unknown>,
      }).catch(() => null);
      toast.success("Remote signal applied.");
    } catch (error) {
      toast.error("Invalid signal payload.");
    }
  };

  const toggleMute = () => {
    if (!localStream) return;
    localStream.getAudioTracks().forEach((track) => (track.enabled = muted));
    setMuted((prev) => !prev);
  };

  const toggleCamera = () => {
    if (!localStream) return;
    localStream.getVideoTracks().forEach((track) => (track.enabled = cameraOff));
    setCameraOff((prev) => !prev);
  };

  const endCall = async () => {
    peerRef.current?.destroy();
    setConnectionStatus("closed");
    setPhase("post");
    const record = {
      id: `consult-${Date.now()}`,
      patientName: "Jane Doe",
      summary: "AI summary generated for cardiology consult.",
      createdAt: new Date().toISOString(),
    };
    await saveConsultation(record);
    if (!online) {
      await enqueueSync({ id: `sync-${Date.now()}`, type: "consultation", payload: record });
    }
  };

  const exportCarePlan = () => {
    const content = "ClinSync AI Care Plan\n\n- Follow-up cardiology consult\n- Repeat imaging in 7 days";
    const blob = new Blob([content], { type: "application/pdf" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `care-plan-${params.roomId}.pdf`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success("Care plan exported.");
  };

  const generateSummary = async () => {
    try {
      const response = await sendChat({
        text: "Generate a clinical summary and next steps for this consultation.",
      });
      setFromResponse(response);
      toast.success("AI summary generated.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Summary generation failed.");
    }
  };

  return (
    <div className="space-y-6 p-4">
      {phase === "lobby" && (
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Pre-consultation Lobby</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="aspect-video overflow-hidden rounded-lg border border-border bg-muted/30">
                <video
                  ref={localVideoRef}
                  autoPlay
                  playsInline
                  muted
                  className={cn("h-full w-full object-cover", blurBackground ? "blur-sm" : "")}
                />
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Badge variant={localStream ? "success" : "secondary"}>
                  Camera {localStream ? "Ready" : "Pending"}
                </Badge>
                <Badge variant={localStream ? "success" : "secondary"}>
                  Microphone {localStream ? "Ready" : "Pending"}
                </Badge>
                <Button variant="outline" onClick={() => setBlurBackground((prev) => !prev)}>
                  Toggle background blur
                </Button>
              </div>
              <div className="flex flex-wrap gap-3">
                <Button onClick={() => { startPeer(); setPhase("in-call"); }}>
                  Join Consultation
                </Button>
                <Button variant="secondary" onClick={startPeer}>
                  Test Connection
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Signal Exchange</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <p className="text-muted-foreground">
                Paste the remote peer signal to complete the WebRTC handshake.
              </p>
              <Button variant="outline" onClick={() => { navigator.clipboard.writeText(offerSignal); toast.success("Offer copied."); }} aria-label="Copy offer signal" disabled={!offerSignal}>
                <Copy className="mr-2 h-4 w-4" aria-hidden /> Copy offer
              </Button>
              <textarea
                className="w-full rounded-md border border-input bg-background p-2 text-xs"
                rows={5}
                placeholder="Paste remote signal JSON"
                value={remoteSignal}
                onChange={(event) => setRemoteSignal(event.target.value)}
                aria-label="Remote signal"
              />
              <Button onClick={applyRemoteSignal} aria-label="Apply remote signal">
                Apply remote signal
              </Button>
              <Badge variant="secondary">Status: {connectionStatus}</Badge>
            </CardContent>
          </Card>
        </div>
      )}

      {phase === "in-call" && (
        <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
          <Card className="relative overflow-hidden">
            <CardContent className="p-0">
              <div className="relative aspect-video bg-black">
                <video ref={remoteVideoRef} autoPlay playsInline className="h-full w-full object-cover" />
                {!remoteStream && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-sm text-white/80">
                    Waiting for remote stream...
                  </div>
                )}
                {overlayFindings.map((finding) => (
                  <motion.div key={finding.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={cn("absolute", finding.position)}>
                    <Badge variant="secondary">{finding.label}</Badge>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="flex h-full flex-col">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Live Insights</CardTitle>
              <Badge variant="secondary">AI Assist</Badge>
            </CardHeader>
            <CardContent className="flex-1">
              <Tabs defaultValue="transcript" className="h-full">
                <TabsList className="w-full justify-start">
                  <TabsTrigger value="transcript">Transcript</TabsTrigger>
                  <TabsTrigger value="suggestions">AI Suggestions</TabsTrigger>
                  <TabsTrigger value="chat">Chat</TabsTrigger>
                </TabsList>
                <TabsContent value="transcript">
                  <ScrollArea className="h-72 rounded-md border border-border p-3 text-sm text-muted-foreground">
                    <p>Patient: Experiencing tightness in chest after climbing stairs.</p>
                    <p className="mt-2">Clinician: Any dizziness or nausea?</p>
                    <p className="mt-2">Patient: Mild dizziness, no nausea.</p>
                  </ScrollArea>
                </TabsContent>
                <TabsContent value="suggestions">
                  <div className="space-y-3 text-sm">
                    <Card className="p-3">
                      <p className="font-semibold">AI Suggestion</p>
                      <p className="text-muted-foreground">Ask about family cardiac history and recent medication adherence.</p>
                    </Card>
                    <Card className="p-3">
                      <p className="font-semibold">Protocol</p>
                      <p className="text-muted-foreground">Consider ordering ECG and troponin based on symptom onset.</p>
                    </Card>
                  </div>
                </TabsContent>
                <TabsContent value="chat">
                  <ChatInterface />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <div className="xl:col-span-2">
            <Card>
              <CardContent className="flex flex-wrap items-center justify-between gap-4 p-4">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">Vitals: HR 82</Badge>
                  <Badge variant="outline">BP 120/76</Badge>
                  <Badge variant="outline">SpO2 96%</Badge>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant={muted ? "destructive" : "outline"} size="icon" onClick={toggleMute} aria-label={muted ? "Unmute microphone" : "Mute microphone"}>
                    {muted ? <MicOff className="h-4 w-4" aria-hidden /> : <Mic className="h-4 w-4" aria-hidden />}
                  </Button>
                  <Button variant={cameraOff ? "destructive" : "outline"} size="icon" onClick={toggleCamera} aria-label={cameraOff ? "Enable camera" : "Disable camera"}>
                    {cameraOff ? <VideoOff className="h-4 w-4" aria-hidden /> : <Video className="h-4 w-4" aria-hidden />}
                  </Button>
                  <Button variant="secondary" aria-label="Generate AI summary" onClick={generateSummary}>
                    <Wand2 className="mr-2 h-4 w-4" aria-hidden /> Generate summary
                  </Button>
                  <Button variant="destructive" onClick={endCall} aria-label="End consultation">
                    <PhoneOff className="mr-2 h-4 w-4" aria-hidden /> End Call
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {phase === "post" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Post-consultation Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              AI summary: Patient reports exertional chest tightness with mild dizziness. Recommended ECG, troponin, and cardiology follow-up. No acute distress observed during call.
            </div>
            <div className="flex flex-wrap gap-3">
              <Button onClick={exportCarePlan} aria-label="Export care plan PDF">Export Care Plan (PDF)</Button>
              <Button variant="outline" asChild><Link href="/consultation/video">Start new consultation</Link></Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
