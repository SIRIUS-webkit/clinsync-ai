"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConsultationAIPanel } from "@/components/consultation/ConsultationAIPanel";
import { sendWebrtcAnswer, sendWebrtcOffer } from "@/lib/api";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import { Copy, Mic, MicOff, PhoneOff, Video, VideoOff, Activity, MessageSquare } from "lucide-react";
import { analyzeImage, sendChat } from "@/lib/api";
import { useAnalysisStore } from "@/stores/useAnalysisStore";

const overlayFindingsDefault = [
  { label: "AI Monitoring Active", position: "top-6 left-6", variant: "default" },
];

export default function ConsultationVideoPage() {
  const [localStream, setLocalStream] = React.useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = React.useState<MediaStream | null>(null);
  const [blurBackground, setBlurBackground] = React.useState(false);
  const [offerSignal, setOfferSignal] = React.useState("");
  const [remoteSignal, setRemoteSignal] = React.useState("");
  const [connectionStatus, setConnectionStatus] = React.useState("disconnected");
  const [muted, setMuted] = React.useState(false);
  const [cameraOff, setCameraOff] = React.useState(false);
  const peerRef = React.useRef<any>(null); // Use any for dynamic peer type
  const peerIdRef = React.useRef(`clinician-${Math.random().toString(36).slice(2, 8)}`);
  const localVideoRef = React.useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = React.useRef<HTMLVideoElement | null>(null);
  const analysisIntervalRef = React.useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null);
  const audioChunksRef = React.useRef<Blob[]>([]);
  
  const setAnalysisData = useAnalysisStore((s) => s.setFromResponse);
  const [aiAnalyzing, setAiAnalyzing] = React.useState(false);
  const [isTalking, setIsTalking] = React.useState(false);
  const [isProcessing, setIsProcessing] = React.useState(false);
  const [isSpeaking, setIsSpeaking] = React.useState(false);
  const [overlayItems, setOverlayItems] = React.useState(overlayFindingsDefault);

  React.useEffect(() => {
    const setupMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setLocalStream(stream);
      } catch (error) {
        toast.error("Camera/microphone access denied.");
      }
    };
    setupMedia();
    return () => {
      peerRef.current?.destroy();
      localStream?.getTracks().forEach((track) => track.stop());
      if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current);
    };
  }, []);

  // Periodic AI Analysis (Silent Vision Only)
  React.useEffect(() => {
    // Disabled purely for testing "Talk to AI" mode as requested
    return;

    /*
    if (!localStream || cameraOff || isTalking) return; 

    const captureAndAnalyze = async () => {
       // ... existing capture logic ...
    };

    const intervalId = setInterval(captureAndAnalyze, 10000);
    analysisIntervalRef.current = intervalId;
    
    return () => clearInterval(intervalId);
    */
  }, [localStream, cameraOff, setAnalysisData, isTalking]);

  const startTalking = async () => {
    if (!localStream) return;
    setIsTalking(true);
    audioChunksRef.current = [];
    
    try {
      const recorder = new MediaRecorder(localStream);
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      recorder.start();
      mediaRecorderRef.current = recorder;
      toast.success("Listening... Speak now.");
      
    } catch (err) {
      console.error("Failed to start recording:", err);
      setIsTalking(false);
    }
  };

  const stopTalking = async () => {
    if (!mediaRecorderRef.current || !isTalking) return;
    
    const recorder = mediaRecorderRef.current;
    
    recorder.onstop = async () => {
        setIsTalking(false);
        setIsProcessing(true); // Start processing animation
        
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const audioFile = new File([audioBlob], "voice_input.webm", { type: "audio/webm" });
        
        const processingStart = Date.now();
        
        // Capture specific frame for this interaction
        if (localVideoRef.current) {
            const canvas = document.createElement("canvas");
            canvas.width = localVideoRef.current.videoWidth;
            canvas.height = localVideoRef.current.videoHeight;
            canvas.getContext("2d")?.drawImage(localVideoRef.current, 0, 0);
            
            canvas.toBlob(async (imageBlob) => {
                if (!imageBlob) {
                    setIsProcessing(false);
                    return;
                }
                const imageFile = new File([imageBlob], "context_image.jpg", { type: "image/jpeg" });
                
                try {
                    // Send multimodal request in VOICE mode for conversational response
                    const response = await sendChat({
                        audio: audioFile,
                        image: imageFile,
                        mode: "voice"  // Conversational mode for real-time consultation
                    });
                    
                    const processingTime = Date.now() - processingStart;
                    console.log(`Processing took ${processingTime}ms`);
                    
                    setIsProcessing(false); // Stop processing animation
                    
                    // Cast to any to access custom fields if needed, simplified for safe access
                    const data = response as any;
                    setAnalysisData(data);
                    
                    // Get the response text - backend uses response_text field
                    const responseText = data.response_text || data.response;
                    
                    if (responseText) {
                        // Strip markdown for cleaner speech
                        const cleanText = responseText.replace(/\*\*/g, "").replace(/#/g, "").replace(/`/g, "");
                        
                        console.log("Speaking response:", cleanText.substring(0, 100) + "...");
                        
                        // Cancel any current speech
                        window.speechSynthesis.cancel();
                        
                        const utterance = new SpeechSynthesisUtterance(cleanText);
                        utterance.rate = 1.1; // Slightly faster for more natural conversation
                        utterance.pitch = 1.0;
                        
                        // Robust voice selection
                        const loadVoices = () => {
                            const voices = window.speechSynthesis.getVoices();
                            
                            // Preferred voices in order
                            const preferredVoice = voices.find(v => v.name.includes("Google US English")) || 
                                                 voices.find(v => v.lang === "en-US") ||
                                                 voices.find(v => v.lang.startsWith("en")) ||
                                                 voices[0]; // fallback to any available
                                                 
                            if (preferredVoice) {
                                utterance.voice = preferredVoice;
                            }
                            
                            utterance.onerror = (e) => {
                                console.error("Speech synthesis error:", e);
                                setIsSpeaking(false);
                            };
                            utterance.onstart = () => {
                                console.log("Speech started");
                                setIsSpeaking(true); // Start speaking animation
                            };
                            utterance.onend = () => {
                                console.log("Speech finished");
                                setIsSpeaking(false); // Stop speaking animation
                            };
                            
                            window.speechSynthesis.speak(utterance);
                        };

                        if (window.speechSynthesis.getVoices().length > 0) {
                             loadVoices();
                        } else {
                             window.speechSynthesis.onvoiceschanged = loadVoices;
                        }
                        
                        toast.success("AI Responded");
                    } else {
                        console.warn("No response text found in API response:", data);
                        toast.error("No response from AI");
                    }

                } catch (error) {
                    setIsProcessing(false);
                    setIsSpeaking(false);
                    toast.error("Failed to process request");
                    console.error(error);
                }
            }, "image/jpeg", 0.8);
        } else {
            setIsProcessing(false);
        }
    };
    
    recorder.stop();
  };

  React.useEffect(() => {
    if (localVideoRef.current && localStream) localVideoRef.current.srcObject = localStream;
  }, [localStream]);
  React.useEffect(() => {
    if (remoteVideoRef.current && remoteStream) remoteVideoRef.current.srcObject = remoteStream;
  }, [remoteStream]);

  const startPeer = async () => {
    if (!localStream) return;
    peerRef.current?.destroy();
    
    // Dynamically import simple-peer only on client
    const SimplePeer = (await import("simple-peer")).default;
    
    // @ts-ignore - SimplePeer types mismatch with dynamic import
    const peer = new SimplePeer({ initiator: true, trickle: false, stream: localStream });
    peer.on("signal", (data: any) => {
      setOfferSignal(JSON.stringify(data));
      sendWebrtcOffer({ room_id: "demo", peer_id: peerIdRef.current, sdp: data as Record<string, unknown> }).catch(() => null);
    });
    peer.on("connect", () => setConnectionStatus("connected"));
    peer.on("close", () => setConnectionStatus("closed"));
    peer.on("error", () => setConnectionStatus("error"));
    peer.on("stream", (stream: MediaStream) => setRemoteStream(stream));
    peerRef.current = peer;
    setConnectionStatus("connecting");
  };

  const applyRemoteSignal = () => {
    if (!peerRef.current || !remoteSignal) return;
    try {
      const parsed = JSON.parse(remoteSignal);
      peerRef.current.signal(parsed);
      sendWebrtcAnswer({ room_id: "demo", peer_id: "remote-peer", sdp: parsed as Record<string, unknown> }).catch(() => null);
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

  const endCall = () => {
    // Stop all media tracks
    if (localStream) {
      localStream.getTracks().forEach((track) => {
        track.stop();
      });
      setLocalStream(null);
    }
    
    // Stop any ongoing speech
    window.speechSynthesis.cancel();
    
    // Stop any recording
    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        // Ignore if already stopped
      }
      mediaRecorderRef.current = null;
    }
    
    // Clear analysis interval
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
    
    // Destroy peer connection
    if (peerRef.current) {
      peerRef.current.destroy();
      peerRef.current = null;
    }
    
    toast.success("Call ended");
  };

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div className="relative flex min-h-0 flex-1 basis-[70%] flex-col">
          <Card className="m-2 flex flex-1 flex-col overflow-hidden lg:m-4">
            <CardContent className="relative flex flex-1 flex-col p-0">
              <div className="relative flex-1 bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950 min-h-[400px]">
                {/* Main Area - AI Assistant Avatar (Full Size) */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  {/* Animated background circles - intensify during activity */}
                  <div className="absolute inset-0 overflow-hidden">
                    <div className={cn(
                      "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full blur-3xl transition-all duration-500",
                      isProcessing ? "bg-yellow-500/30 animate-pulse scale-110" :
                      isSpeaking ? "bg-green-500/20 animate-pulse" :
                      isTalking ? "bg-blue-500/20 animate-pulse" :
                      "bg-blue-500/10"
                    )} />
                    <div className={cn(
                      "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full blur-2xl transition-all duration-500",
                      isProcessing ? "bg-orange-500/30" :
                      isSpeaking ? "bg-emerald-500/30" :
                      "bg-purple-500/20"
                    )} />
                  </div>
                  
                  {/* AI Avatar */}
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ 
                      scale: isSpeaking ? [1, 1.05, 1] : 1, 
                      opacity: 1 
                    }}
                    transition={{ 
                      scale: { duration: 0.5, repeat: isSpeaking ? Infinity : 0 }
                    }}
                    className="relative z-10 flex flex-col items-center"
                  >
                    {/* Avatar circle with state-based styling */}
                    <div className={cn(
                      "h-32 w-32 rounded-full flex items-center justify-center shadow-2xl mb-6 transition-all duration-300",
                      isProcessing ? "bg-gradient-to-br from-yellow-500 to-orange-600 ring-4 ring-yellow-400/50" :
                      isSpeaking ? "bg-gradient-to-br from-green-500 to-emerald-600 ring-4 ring-green-400/50 animate-pulse" :
                      isTalking ? "bg-gradient-to-br from-blue-500 to-purple-600 ring-4 ring-blue-400/50 animate-pulse" :
                      "bg-gradient-to-br from-blue-500 to-purple-600"
                    )}>
                      {isProcessing ? (
                        <div className="flex space-x-1">
                          <div className="w-3 h-3 bg-white rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                          <div className="w-3 h-3 bg-white rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                          <div className="w-3 h-3 bg-white rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                      ) : isSpeaking ? (
                        <div className="flex items-end space-x-1 h-10">
                          <div className="w-2 bg-white rounded-full animate-pulse" style={{ height: "60%", animationDelay: "0ms" }} />
                          <div className="w-2 bg-white rounded-full animate-pulse" style={{ height: "100%", animationDelay: "100ms" }} />
                          <div className="w-2 bg-white rounded-full animate-pulse" style={{ height: "40%", animationDelay: "200ms" }} />
                          <div className="w-2 bg-white rounded-full animate-pulse" style={{ height: "80%", animationDelay: "300ms" }} />
                          <div className="w-2 bg-white rounded-full animate-pulse" style={{ height: "50%", animationDelay: "400ms" }} />
                        </div>
                      ) : (
                        <Activity className={cn("h-16 w-16 text-white", isTalking && "animate-bounce")} />
                      )}
                    </div>
                    
                    <h2 className="text-white text-2xl font-bold mb-2">ClinSync AI</h2>
                    <p className="text-blue-300 text-sm mb-1">Your Medical Assistant</p>
                    
                    {/* State badge */}
                    <Badge 
                      variant={isProcessing ? "default" : isSpeaking ? "default" : isTalking ? "default" : "secondary"} 
                      className={cn(
                        "mt-2 transition-all",
                        isProcessing && "bg-yellow-500 hover:bg-yellow-600",
                        isSpeaking && "bg-green-500 hover:bg-green-600"
                      )}
                    >
                      {isProcessing ? "🔄 Processing..." : 
                       isSpeaking ? "🔊 Speaking..." : 
                       isTalking ? "🎤 Listening..." : 
                       "✨ Ready to help"}
                    </Badge>
                  </motion.div>
                </div>
                
                {/* AI Overlay Badges */}
                {overlayItems.map((finding) => (
                  <motion.div
                    key={finding.label}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn("absolute z-20", finding.position)}
                  >
                    <Badge variant={finding.variant as "default" | "secondary" | "destructive" | "outline"}>
                      {finding.label}
                      {finding.label === "AI Monitoring Active" && aiAnalyzing && (
                        <Activity className="ml-2 h-3 w-3 animate-pulse" />
                      )}
                    </Badge>
                  </motion.div>
                ))}
                
                {/* User Camera - Small Picture-in-Picture (Bottom Right) */}
                <div className="absolute bottom-4 right-4 h-36 w-48 overflow-hidden rounded-xl border-2 border-white/20 bg-black shadow-2xl z-20">
                  <video
                    ref={localVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className={cn(
                      "h-full w-full object-cover",
                      blurBackground && "blur-sm",
                      cameraOff && "hidden"
                    )}
                  />
                  {cameraOff && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-800">
                      <VideoOff className="h-8 w-8 text-white/60 mb-1" />
                      <p className="text-white/60 text-xs">Camera off</p>
                    </div>
                  )}
                  <div className="absolute bottom-1 left-1">
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0.5">You</Badge>
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border bg-muted/40 p-3">
                <div className="flex gap-2">
                  <Badge variant="outline">HR 82</Badge>
                  <Badge variant="outline">SpO2 96%</Badge>
                </div>
                <div className="flex gap-2">
                  <Button 
                    variant={isTalking ? "default" : "secondary"} 
                    className={cn("min-w-[100px]", isTalking && "animate-pulse bg-red-500 hover:bg-red-600 text-white")}
                    onMouseDown={startTalking}
                    onMouseUp={stopTalking}
                    onMouseLeave={() => isTalking && stopTalking()} // Stop if they drag away
                  >
                    {isTalking ? (
                        <>
                            <Activity className="mr-2 h-4 w-4 animate-pulse" />
                            Listening
                        </>
                    ) : (
                        <>
                            <MessageSquare className="mr-2 h-4 w-4" />
                            Talk to AI
                        </>
                    )}
                  </Button>
                  <Button variant={muted ? "destructive" : "outline"} size="icon" onClick={toggleMute} aria-label={muted ? "Unmute" : "Mute"}>
                    {muted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                  </Button>
                  <Button variant={cameraOff ? "destructive" : "outline"} size="icon" onClick={toggleCamera} aria-label={cameraOff ? "Camera on" : "Camera off"}>
                    {cameraOff ? <VideoOff className="h-4 w-4" /> : <Video className="h-4 w-4" />}
                  </Button>
                  <Button variant="outline" onClick={() => setBlurBackground((b) => !b)}>Blur</Button>
                  <Link href="/dashboard" onClick={endCall}>
                    <Button variant="destructive">End call</Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>
          <div className="mx-2 mb-2 lg:mx-4 lg:mb-4">
            <Card className="p-3">
              <p className="text-xs text-muted-foreground mb-2">Signal exchange (testing)</p>
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(offerSignal); toast.success("Copied"); }} disabled={!offerSignal}>
                  <Copy className="mr-1 h-3 w-3" /> Copy offer
                </Button>
                <input
                  className="min-w-[200px] flex-1 rounded border border-input bg-background px-2 py-1 text-xs"
                  placeholder="Paste remote signal"
                  value={remoteSignal}
                  onChange={(e) => setRemoteSignal(e.target.value)}
                />
                <Button size="sm" onClick={applyRemoteSignal}>Apply</Button>
                <Badge variant="secondary">{connectionStatus}</Badge>
              </div>
            </Card>
          </div>
        </div>
        <ConsultationAIPanel mode="video" className="relative hidden lg:block" />
      </div>
    </>
  );
}
