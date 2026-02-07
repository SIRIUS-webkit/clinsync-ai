"use client";

import * as React from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConsultationAIPanel } from "@/components/consultation/ConsultationAIPanel";
import { IntakeForm } from "@/components/consultation/IntakeForm";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import { Mic, MicOff, PhoneOff, Video, VideoOff, Activity, Sparkles, Volume2, Wifi, WifiOff, FileText, User } from "lucide-react";
import { useAnalysisStore } from "@/stores/useAnalysisStore";
import { useConsultationContext, PatientIntake } from "@/stores/useConsultationContext";

// Connection states
type ConnectionState = "disconnected" | "connecting" | "connected" | "failed";
// AI states
type AIState = "idle" | "listening" | "processing" | "speaking";

export default function ConsultationVideoPage() {
  const [localStream, setLocalStream] = React.useState<MediaStream | null>(null);
  const [connectionState, setConnectionState] = React.useState<ConnectionState>("disconnected");
  const [aiState, setAiState] = React.useState<AIState>("idle");
  const [muted, setMuted] = React.useState(false);
  const [cameraOff, setCameraOff] = React.useState(false);
  const [blurBackground, setBlurBackground] = React.useState(false);
  const [transcript, setTranscript] = React.useState("");
  const [responseText, setResponseText] = React.useState("");
  const [isClient, setIsClient] = React.useState(false);
  const [showVideoCall, setShowVideoCall] = React.useState(false);
  
  // Generate session ID only on client to avoid hydration mismatch
  const sessionIdRef = React.useRef<string>("");
  
  const localVideoRef = React.useRef<HTMLVideoElement | null>(null);
  const wsRef = React.useRef<WebSocket | null>(null);
  const audioContextRef = React.useRef<AudioContext | null>(null);
  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null);
  const isRecordingRef = React.useRef(false);
  const silenceTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);
  
  const setAnalysisData = useAnalysisStore((s) => s.setFromResponse);
  
  // Get consultation context (patient intake data)
  const { intake, setIntake, startSession, getSystemPromptContext, addMessage } = useConsultationContext();
  const hasIntakeData = !!intake;
  
  // Set client flag after mount to avoid hydration issues
  React.useEffect(() => {
    setIsClient(true);
    sessionIdRef.current = `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }, []);
  
  // Handle intake form completion - start video call
  const handleIntakeComplete = React.useCallback((intakeData: PatientIntake) => {
    setIntake(intakeData);
    startSession();
    setShowVideoCall(true);
    toast.success("Starting video consultation...");
  }, [setIntake, startSession]);

  // Initialize connection using WebSocket (more reliable than full WebRTC for AI calls)
  const initializeConnection = React.useCallback(async (stream: MediaStream) => {
    try {
      setConnectionState("connecting");
      
      const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
      const wsUrl = apiUrl.replace(/^http/, "ws").replace(/\/$/, "");
      
      // Use the simpler real-time WebSocket endpoint
      const ws = new WebSocket(`${wsUrl}/chat/ws/realtime`);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionState("connected");
        setAiState("listening");
        toast.success("Connected to AI Assistant");
        
        // Send patient context if available
        if (intake) {
          const context = getSystemPromptContext();
          ws.send(JSON.stringify({
            type: "context",
            data: {
              patientId: intake.patientId,
              fullName: intake.fullName,
              age: intake.age,
              gender: intake.gender,
              chiefComplaint: intake.chiefComplaint,
              symptoms: intake.symptoms,
              symptomDuration: intake.symptomDuration,
              painLevel: intake.painLevel,
              allergies: intake.allergies,
              currentMedications: intake.currentMedications,
              medicalHistory: intake.medicalHistory,
              consultationType: intake.consultationType,
              contextPrompt: context,
            }
          }));
          console.log("Sent patient context to AI");
          
          // Send attached images if any
          if (intake.attachedImages && intake.attachedImages.length > 0) {
            for (const img of intake.attachedImages) {
              ws.send(JSON.stringify({
                type: "image",
                data: img.dataUrl,
                imageType: img.type,
              }));
            }
            console.log(`Sent ${intake.attachedImages.length} attached images`);
          }
        }
        
        // Start voice activity detection
        startVoiceActivityDetection(stream);
      };
      
      ws.onclose = () => {
        console.log("WebSocket closed");
        setConnectionState("disconnected");
        setAiState("idle");
      };
      
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionState("failed");
        toast.error("Connection failed. Please check if the backend is running.");
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleServerMessage(data);
        } catch (e) {
          console.error("Failed to parse message:", e);
        }
      };
      
    } catch (error) {
      console.error("Connection initialization failed:", error);
      setConnectionState("failed");
      toast.error("Failed to connect");
    }
  }, []);
  
  // Handle server messages
  const handleServerMessage = React.useCallback((data: any) => {
    switch (data.type) {
      case "listening":
        setAiState("listening");
        break;
      case "processing":
        setAiState("processing");
        break;
      case "speaking":
        setAiState("speaking");
        setResponseText(data.text || "");
        if (data.transcript) setTranscript(data.transcript);
        speakResponse(data.text);
        break;
      case "complete":
        if (data.data) setAnalysisData(data.data);
        break;
      case "error":
        toast.error(data.message || "Error occurred");
        setAiState("listening");
        break;
      case "pong":
        break;
    }
  }, [setAnalysisData]);
  
  // Voice activity detection
  const startVoiceActivityDetection = React.useCallback((stream: MediaStream) => {
    try {
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.85;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      audioContextRef.current = audioContext;
      
      // Voice detection settings
      const VOICE_THRESHOLD = 0.08;  // Increased from 0.02 - less sensitive
      const SILENCE_DELAY = 1800;    // Wait 1.8 seconds of silence before stopping
      const MIN_SPEECH_FRAMES = 10;  // Require ~10 frames of speech before triggering
      
      let speechFrameCount = 0;
      
      const checkAudio = () => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || muted) {
          requestAnimationFrame(checkAudio);
          return;
        }
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedVolume = avg / 255;
        
        // Voice detected
        if (normalizedVolume > VOICE_THRESHOLD) {
          speechFrameCount++;
          
          if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
          }
          
          // Start recording only after sustained speech (not just a brief noise)
          if (!isRecordingRef.current && 
              speechFrameCount >= MIN_SPEECH_FRAMES && 
              aiState !== "speaking" && 
              aiState !== "processing") {
            console.log("Starting recording - voice detected");
            startRecording(stream);
          }
        } else {
          // Reset speech counter on silence
          if (speechFrameCount > 0 && speechFrameCount < MIN_SPEECH_FRAMES) {
            // Brief noise, reset
            speechFrameCount = 0;
          }
          
          // Silence - stop recording after delay
          if (isRecordingRef.current && !silenceTimeoutRef.current) {
            silenceTimeoutRef.current = setTimeout(() => {
              console.log("Stopping recording - silence detected");
              stopRecording();
              silenceTimeoutRef.current = null;
              speechFrameCount = 0;
            }, SILENCE_DELAY);
          }
        }
        
        requestAnimationFrame(checkAudio);
      };
      
      requestAnimationFrame(checkAudio);
      
    } catch (e) {
      console.error("Voice activity detection setup failed:", e);
    }
  }, [muted, aiState]);
  
  // Start recording
  const startRecording = (stream: MediaStream) => {
    if (isRecordingRef.current) return;
    
    try {
      // Get audio tracks only
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        console.error("No audio tracks available");
        return;
      }
      
      // Create audio-only stream
      const audioStream = new MediaStream(audioTracks);
      
      // Find supported MIME type
      const mimeTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
        "audio/wav",
        "",  // Browser default
      ];
      
      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (mimeType === "" || MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }
      
      console.log("Using MIME type:", selectedMimeType || "(browser default)");
      
      isRecordingRef.current = true;
      
      const recorderOptions: MediaRecorderOptions = {};
      if (selectedMimeType) {
        recorderOptions.mimeType = selectedMimeType;
      }
      
      const recorder = new MediaRecorder(audioStream, recorderOptions);
      
      // Collect all chunks then send complete audio on stop
      const chunks: Blob[] = [];
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };
      
      recorder.onstop = async () => {
        // Combine all chunks into one complete blob
        if (chunks.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          const completeBlob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          console.log("Sending complete audio:", completeBlob.size, "bytes", completeBlob.type);
          
          const reader = new FileReader();
          reader.onloadend = () => {
            wsRef.current?.send(JSON.stringify({
              type: "audio",
              data: reader.result,
            }));
            
            // Capture frame for context
            if (localVideoRef.current && !cameraOff) {
              const canvas = document.createElement("canvas");
              canvas.width = localVideoRef.current.videoWidth || 640;
              canvas.height = localVideoRef.current.videoHeight || 480;
              canvas.getContext("2d")?.drawImage(localVideoRef.current, 0, 0);
              
              wsRef.current?.send(JSON.stringify({
                type: "image",
                data: canvas.toDataURL("image/jpeg", 0.7),
              }));
            }
            
            // Signal end of audio
            wsRef.current?.send(JSON.stringify({ type: "audio_end" }));
          };
          reader.readAsDataURL(completeBlob);
        } else {
          // No audio captured, just send end signal
          wsRef.current?.send(JSON.stringify({ type: "audio_end" }));
        }
      };
      
      recorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        isRecordingRef.current = false;
      };
      
      // Start recording without timeslice to get complete audio with headers
      recorder.start();
      mediaRecorderRef.current = recorder;
      
    } catch (e) {
      console.error("Recording start failed:", e);
      isRecordingRef.current = false;
    }
  };
  
  // Stop recording
  const stopRecording = () => {
    if (!isRecordingRef.current || !mediaRecorderRef.current) return;
    
    isRecordingRef.current = false;
    
    try {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    } catch (e) {
      console.error("Recording stop failed:", e);
    }
  };
  
  // Text-to-speech with improved reliability
  const speakResponse = React.useCallback((text: string) => {
    if (!text || typeof window === "undefined") {
      console.log("TTS: No text or not in browser");
      setAiState("listening");
      return;
    }
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    // Clean up the text for speaking
    const cleanText = text
      .replace(/\*\*/g, "")
      .replace(/#/g, "")
      .replace(/`/g, "")
      .replace(/\n+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    
    if (!cleanText) {
      console.log("TTS: Text is empty after cleanup");
      setAiState("listening");
      return;
    }
    
    console.log("TTS: Speaking text:", cleanText.substring(0, 50) + "...");
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    const speak = () => {
      const voices = window.speechSynthesis.getVoices();
      console.log("TTS: Available voices:", voices.length);
      
      // Find a good voice
      const voice = voices.find(v => v.name.includes("Google US English")) ||
                    voices.find(v => v.name.includes("Samantha")) ||
                    voices.find(v => v.lang.startsWith("en") && v.localService) ||
                    voices.find(v => v.lang === "en-US") ||
                    voices[0];
      
      if (voice) {
        utterance.voice = voice;
        console.log("TTS: Using voice:", voice.name);
      }
      
      utterance.onstart = () => {
        console.log("TTS: Started speaking");
      };
      
      utterance.onend = () => {
        console.log("TTS: Finished speaking");
        setAiState("listening");
      };
      
      utterance.onerror = (event) => {
        console.error("TTS: Error:", event.error);
        setAiState("listening");
      };
      
      // Chrome bug workaround: Speech synthesis stops after ~15 seconds
      // Solution: Pause/resume periodically
      let resumeTimer: NodeJS.Timeout | null = null;
      
      const keepAlive = () => {
        if (window.speechSynthesis.speaking) {
          window.speechSynthesis.pause();
          window.speechSynthesis.resume();
          resumeTimer = setTimeout(keepAlive, 10000);
        }
      };
      
      utterance.onstart = () => {
        console.log("TTS: Started speaking");
        resumeTimer = setTimeout(keepAlive, 10000);
      };
      
      utterance.onend = () => {
        console.log("TTS: Finished speaking");
        if (resumeTimer) clearTimeout(resumeTimer);
        setAiState("listening");
      };
      
      utterance.onerror = (event) => {
        console.error("TTS: Error:", event.error);
        if (resumeTimer) clearTimeout(resumeTimer);
        setAiState("listening");
      };
      
      window.speechSynthesis.speak(utterance);
    };
    
    // Voices might not be loaded yet
    if (window.speechSynthesis.getVoices().length > 0) {
      speak();
    } else {
      // Wait for voices to load
      window.speechSynthesis.onvoiceschanged = speak;
      // Fallback timeout - speak even without voices after 500ms
      setTimeout(() => {
        if (window.speechSynthesis.getVoices().length === 0) {
          console.log("TTS: No voices loaded, speaking with default");
          window.speechSynthesis.speak(utterance);
        }
      }, 500);
    }
  }, []);
  
  // Initialize camera/mic only after intake form is complete
  React.useEffect(() => {
    if (!isClient || !showVideoCall) return;
    
    let mounted = true;
    
    const setup = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
        
        if (!mounted) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }
        
        setLocalStream(stream);
        initializeConnection(stream);
        
      } catch (error) {
        console.error("Media access error:", error);
        toast.error("Camera/microphone access denied");
      }
    };
    
    setup();
    
    return () => {
      mounted = false;
      cleanup();
    };
  }, [isClient, showVideoCall, initializeConnection]);
  
  // Cleanup
  const cleanup = () => {
    localStream?.getTracks().forEach(t => t.stop());
    wsRef.current?.close();
    audioContextRef.current?.close();
    if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
    if (mediaRecorderRef.current) {
      try { mediaRecorderRef.current.stop(); } catch (e) {}
    }
    window.speechSynthesis.cancel();
  };
  
  // Toggle controls
  const toggleMute = () => {
    localStream?.getAudioTracks().forEach(t => (t.enabled = muted));
    setMuted(!muted);
    if (!muted) stopRecording();
  };
  
  const toggleCamera = () => {
    localStream?.getVideoTracks().forEach(t => (t.enabled = cameraOff));
    setCameraOff(!cameraOff);
  };
  
  const endCall = () => {
    cleanup();
    setLocalStream(null);
    setConnectionState("disconnected");
    toast.success("Call ended");
  };
  
  // Update video element
  React.useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);
  
  // Heartbeat
  React.useEffect(() => {
    if (!isClient) return;
    
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [isClient]);

  // Get avatar styles based on AI state
  const getAvatarStyles = () => {
    switch (aiState) {
      case "listening":
        return { gradient: "from-blue-500 to-indigo-600", ring: "ring-blue-400/50", bg: "bg-blue-500/20" };
      case "processing":
        return { gradient: "from-amber-500 to-orange-600", ring: "ring-amber-400/50", bg: "bg-amber-500/30" };
      case "speaking":
        return { gradient: "from-emerald-500 to-teal-600", ring: "ring-emerald-400/50", bg: "bg-emerald-500/20" };
      default:
        return { gradient: "from-slate-500 to-slate-600", ring: "ring-slate-400/30", bg: "bg-slate-500/10" };
    }
  };
  
  const styles = getAvatarStyles();

  // Don't render until client-side to avoid hydration mismatch
  if (!isClient) {
    return (
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div className="flex-1 bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950 min-h-[400px] flex items-center justify-center">
            <div className="text-white text-lg">Loading...</div>
          </div>
        </div>
      </div>
    );
  }

  // Show intake form first, then video call after completion
  if (!showVideoCall) {
    return <IntakeForm onComplete={handleIntakeComplete} />;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
      <div className="relative flex min-h-0 flex-1 basis-[70%] flex-col">
        <Card className="m-2 flex flex-1 flex-col overflow-hidden lg:m-4">
          <CardContent className="relative flex flex-1 flex-col p-0">
            <div className="relative flex-1 bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950 min-h-[400px]">
              {/* Patient Context Banner */}
              {hasIntakeData && intake && (
                <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-r from-blue-900/80 to-purple-900/80 backdrop-blur-sm border-b border-white/10 px-4 py-2">
                  <div className="flex items-center gap-3 text-sm">
                    <div className="flex items-center gap-2 text-blue-300">
                      <User className="w-4 h-4" />
                      <span className="font-medium">{intake.fullName}</span>
                      <span className="text-blue-400/70">({intake.age}y, {intake.gender})</span>
                    </div>
                    <div className="h-4 w-px bg-white/20" />
                    <div className="flex-1 text-white/70 truncate">
                      <FileText className="w-3 h-3 inline mr-1" />
                      {intake.chiefComplaint.substring(0, 80)}{intake.chiefComplaint.length > 80 ? "..." : ""}
                    </div>
                    {intake.symptoms.length > 0 && (
                      <>
                        <div className="h-4 w-px bg-white/20" />
                        <div className="flex gap-1">
                          {intake.symptoms.slice(0, 3).map((s) => (
                            <Badge key={s} variant="outline" className="text-[10px] border-blue-400/50 text-blue-300">
                              {s}
                            </Badge>
                          ))}
                          {intake.symptoms.length > 3 && (
                            <Badge variant="outline" className="text-[10px] border-blue-400/50 text-blue-300">
                              +{intake.symptoms.length - 3}
                            </Badge>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
              
              {/* AI Assistant Avatar */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                {/* Background effects */}
                <div className="absolute inset-0 overflow-hidden">
                  <motion.div
                    animate={{
                      scale: aiState === "speaking" ? [1, 1.1, 1] : 1,
                      opacity: aiState === "idle" ? 0.1 : 0.3,
                    }}
                    transition={{ duration: 2, repeat: aiState === "speaking" ? Infinity : 0 }}
                    className={cn(
                      "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full blur-3xl",
                      styles.bg
                    )}
                  />
                </div>
                
                {/* Avatar */}
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="relative z-10 flex flex-col items-center"
                >
                  <div className="relative">
                    {/* Pulse rings */}
                    <AnimatePresence>
                      {aiState !== "idle" && (
                        <>
                          <motion.div
                            initial={{ scale: 1, opacity: 0.5 }}
                            animate={{ scale: 1.5, opacity: 0 }}
                            transition={{ duration: 2, repeat: Infinity }}
                            className={cn("absolute inset-0 rounded-full bg-gradient-to-br", styles.gradient)}
                          />
                          <motion.div
                            initial={{ scale: 1, opacity: 0.3 }}
                            animate={{ scale: 1.8, opacity: 0 }}
                            transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                            className={cn("absolute inset-0 rounded-full bg-gradient-to-br", styles.gradient)}
                          />
                        </>
                      )}
                    </AnimatePresence>
                    
                    {/* Main avatar */}
                    <motion.div
                      animate={{ scale: aiState === "speaking" ? [1, 1.02, 1] : 1 }}
                      transition={{ duration: 0.3, repeat: aiState === "speaking" ? Infinity : 0 }}
                      className={cn(
                        "relative h-36 w-36 rounded-full flex items-center justify-center shadow-2xl ring-4",
                        `bg-gradient-to-br ${styles.gradient}`,
                        styles.ring
                      )}
                    >
                      {aiState === "processing" ? (
                        <div className="flex space-x-1.5">
                          {[0, 0.15, 0.3].map((delay, i) => (
                            <motion.div
                              key={i}
                              animate={{ y: [-4, 4, -4] }}
                              transition={{ duration: 0.6, repeat: Infinity, delay }}
                              className="w-3.5 h-3.5 bg-white rounded-full"
                            />
                          ))}
                        </div>
                      ) : aiState === "speaking" ? (
                        <div className="flex items-end space-x-1 h-12">
                          {[0.6, 1, 0.4, 0.8, 0.5].map((height, i) => (
                            <motion.div
                              key={i}
                              animate={{ scaleY: [1, 1.5, 1] }}
                              transition={{ duration: 0.4, repeat: Infinity, delay: i * 0.1 }}
                              className="w-2 bg-white rounded-full origin-bottom"
                              style={{ height: `${height * 3}rem` }}
                            />
                          ))}
                        </div>
                      ) : aiState === "listening" ? (
                        <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1, repeat: Infinity }}>
                          <Mic className="h-16 w-16 text-white" />
                        </motion.div>
                      ) : (
                        <Sparkles className="h-16 w-16 text-white" />
                      )}
                    </motion.div>
                  </div>
                  
                  <h2 className="text-white text-2xl font-bold mt-6 mb-2">ClinSync AI</h2>
                  <p className="text-blue-300 text-sm mb-3">Real-time Medical Assistant</p>
                  
                  <Badge className={cn(
                    "transition-all",
                    aiState === "listening" && "bg-blue-500",
                    aiState === "processing" && "bg-amber-500",
                    aiState === "speaking" && "bg-emerald-500",
                    aiState === "idle" && "bg-slate-500"
                  )}>
                    {aiState === "processing" ? "🔄 Thinking..." :
                     aiState === "speaking" ? "🔊 Speaking..." :
                     aiState === "listening" ? "🎤 Listening..." :
                     "⏸️ Ready"}
                  </Badge>
                  
                  {/* Response preview */}
                  <AnimatePresence>
                    {responseText && aiState === "speaking" && (
                      <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="mt-4 text-white/80 text-sm max-w-md text-center line-clamp-3"
                      >
                        {responseText.substring(0, 150)}...
                      </motion.p>
                    )}
                  </AnimatePresence>
                </motion.div>
              </div>
              
              {/* Connection status */}
              <div className="absolute top-6 left-6">
                <Badge variant={connectionState === "connected" ? "default" : "destructive"} className="backdrop-blur-sm">
                  {connectionState === "connected" ? <><Wifi className="h-3 w-3 mr-1" /> Connected</> :
                   connectionState === "connecting" ? "🟡 Connecting..." :
                   <><WifiOff className="h-3 w-3 mr-1" /> Disconnected</>}
                </Badge>
              </div>
              
              {/* User camera PIP */}
              <div className="absolute bottom-4 right-4 h-36 w-48 overflow-hidden rounded-xl border-2 border-white/20 bg-black shadow-2xl z-20">
                <video
                  ref={localVideoRef}
                  autoPlay
                  playsInline
                  muted
                  className={cn("h-full w-full object-cover", blurBackground && "blur-sm", cameraOff && "hidden")}
                />
                {cameraOff && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-800">
                    <VideoOff className="h-8 w-8 text-white/60" />
                  </div>
                )}
                <Badge variant="secondary" className="absolute bottom-1 left-1 text-[10px]">You</Badge>
                
                {aiState === "listening" && !muted && (
                  <motion.div
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="absolute top-1 right-1 h-2 w-2 rounded-full bg-green-500"
                  />
                )}
              </div>
            </div>
            
            {/* Controls */}
            <div className="flex items-center justify-between border-t border-border bg-muted/40 p-3">
              <div className="flex gap-2">
                <Badge variant="outline">
                  <Volume2 className="h-3 w-3 mr-1" />
                  {aiState === "speaking" ? "AI Speaking" : aiState === "listening" ? "Listening" : "Ready"}
                </Badge>
                {!muted && aiState === "listening" && (
                  <Badge className="bg-green-500/20 text-green-500 border-green-500/50">
                    <Activity className="h-3 w-3 mr-1 animate-pulse" /> Voice Active
                  </Badge>
                )}
              </div>
              <div className="flex gap-2">
                <Button variant={muted ? "destructive" : "outline"} size="icon" onClick={toggleMute}>
                  {muted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
                <Button variant={cameraOff ? "destructive" : "outline"} size="icon" onClick={toggleCamera}>
                  {cameraOff ? <VideoOff className="h-4 w-4" /> : <Video className="h-4 w-4" />}
                </Button>
                <Button variant="outline" onClick={() => setBlurBackground(!blurBackground)}>Blur</Button>
                <Link href="/dashboard" onClick={endCall}>
                  <Button variant="destructive">
                    <PhoneOff className="h-4 w-4 mr-2" /> End call
                  </Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Transcript */}
        {transcript && (
          <Card className="mx-2 mb-2 lg:mx-4 lg:mb-4 p-3">
            <p className="text-xs text-muted-foreground mb-1">You said:</p>
            <p className="text-sm">{transcript}</p>
          </Card>
        )}
      </div>
      <ConsultationAIPanel mode="video" className="relative hidden lg:block" />
    </div>
  );
}
