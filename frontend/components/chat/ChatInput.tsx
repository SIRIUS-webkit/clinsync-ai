"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Camera, Mic, Send } from "lucide-react";
import toast from "react-hot-toast";
import { cn } from "@/lib/utils";

const MAX_IMAGE_SIZE = 10 * 1024 * 1024;
const ACCEPT_IMAGE = "image/jpeg,image/png,image/gif";
const TEXTAREA_MAX_HEIGHT = 160; // ~6 lines

type ChatInputProps = {
  onSend: (payload: { text?: string; image?: string; audio?: string }) => void;
  disabled?: boolean;
  placeholder?: string;
};

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function ChatInput({
  onSend,
  disabled,
  placeholder = "Type a message...",
}: ChatInputProps) {
  const [text, setText] = React.useState("");
  const [imagePreview, setImagePreview] = React.useState<string | null>(null);
  const [isRecording, setIsRecording] = React.useState(false);
  const [waveform, setWaveform] = React.useState<number[]>(
    Array.from({ length: 20 }, () => 15),
  );
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null);
  const audioChunksRef = React.useRef<Blob[]>([]);

  /* ---- auto-resize textarea ---- */
  const resizeTextarea = React.useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, TEXTAREA_MAX_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > TEXTAREA_MAX_HEIGHT ? "auto" : "hidden";
  }, []);

  React.useEffect(() => {
    resizeTextarea();
  }, [text, resizeTextarea]);

  /* ---- recording waveform ---- */
  React.useEffect(() => {
    if (!isRecording) return;
    const interval = setInterval(() => {
      setWaveform((prev) => prev.map(() => Math.floor(10 + Math.random() * 40)));
    }, 100);
    return () => clearInterval(interval);
  }, [isRecording]);

  /* ---- handlers ---- */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitText();
    }
  };

  const submitText = () => {
    const trimmed = text.trim();
    if (trimmed || imagePreview) {
      onSend({ text: trimmed || undefined, image: imagePreview || undefined });
      setText("");
      setImagePreview(null);
      // Reset textarea height after send
      requestAnimationFrame(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto";
        }
      });
    }
  };

  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_IMAGE_SIZE) {
      toast.error("Image must be under 10MB.");
      return;
    }
    if (!file.type.match(/^image\/(jpeg|png|gif)$/)) {
      toast.error("Use JPG, PNG, or GIF.");
      return;
    }
    try {
      const dataUrl = await fileToDataUrl(file);
      setImagePreview(dataUrl);
    } catch {
      toast.error("Failed to read image.");
    }
  };

  const startRecording = async () => {
    if (typeof MediaRecorder === "undefined") {
      toast.error("Recording not supported.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (ev) =>
        ev.data.size > 0 && audioChunksRef.current.push(ev.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const dataUrl = await fileToDataUrl(
          new File([blob], "audio.webm", { type: "audio/webm" }),
        );
        onSend({ audio: dataUrl });
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      toast.error("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  };

  const canSend = text.trim() || imagePreview;

  return (
    <div className="shrink-0 border-t border-border bg-background px-3 pb-3 pt-2 transition-colors duration-200">
      {/* Image preview */}
      {imagePreview && (
        <div className="mb-2 flex items-center gap-2">
          <img
            src={imagePreview}
            alt="Preview"
            className="h-14 w-14 rounded-lg border border-border object-cover"
          />
          <button
            type="button"
            className="text-xs text-muted-foreground underline hover:text-foreground"
            onClick={() => setImagePreview(null)}
          >
            Remove
          </button>
        </div>
      )}

      {/* Recording indicator */}
      {isRecording && (
        <div className="mb-2 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-destructive opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-destructive" />
          </span>
          <span className="text-xs text-muted-foreground">Recording…</span>
          <div className="flex items-end gap-0.5">
            {waveform.map((h, i) => (
              <motion.span
                key={i}
                animate={{ height: h }}
                className="w-1 rounded-full bg-destructive/60"
              />
            ))}
          </div>
          <button
            type="button"
            className="ml-auto text-xs font-medium text-destructive hover:underline"
            onClick={stopRecording}
          >
            Stop
          </button>
        </div>
      )}

      {/* Input row */}
      <div className="flex items-end gap-1.5 rounded-xl border border-input bg-background px-2 py-1.5 transition-colors focus-within:ring-2 focus-within:ring-ring">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="mb-0.5 shrink-0 rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
          aria-label="Upload image"
        >
          <Camera className="h-4 w-4" />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPT_IMAGE}
          className="hidden"
          onChange={handleImageChange}
          aria-label="Upload image"
        />

        <button
          type="button"
          onMouseDown={(e) => e.button === 0 && startRecording()}
          onMouseUp={stopRecording}
          onMouseLeave={stopRecording}
          onTouchStart={(e) => {
            e.preventDefault();
            startRecording();
          }}
          onTouchEnd={(e) => {
            e.preventDefault();
            stopRecording();
          }}
          disabled={disabled}
          className={cn(
            "mb-0.5 shrink-0 rounded-lg p-1.5 transition-colors disabled:opacity-40",
            isRecording
              ? "text-destructive"
              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
          )}
          aria-label="Record voice"
        >
          <Mic className="h-4 w-4" />
        </button>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          className="max-h-[160px] min-h-[36px] flex-1 resize-none bg-transparent py-1.5 text-sm leading-snug outline-none placeholder:text-muted-foreground"
          aria-label="Message"
          disabled={disabled}
        />

        <button
          type="button"
          onClick={submitText}
          disabled={disabled || !canSend}
          className={cn(
            "mb-0.5 shrink-0 rounded-lg p-1.5 transition-all",
            canSend
              ? "bg-primary text-primary-foreground hover:bg-primary/90"
              : "text-muted-foreground/40",
          )}
          aria-label="Send"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>

      <p className="mt-1.5 text-center text-[10px] text-muted-foreground/60">
        ClinSync AI may make mistakes. Verify important medical information.
      </p>
    </div>
  );
}
