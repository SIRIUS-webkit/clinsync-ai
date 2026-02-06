"use client";

import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { format, isToday, isYesterday, isSameDay } from "date-fns";
import { ArrowDown } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useChatStore, type ChatMessage } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Markdown renderers                                                 */
/* ------------------------------------------------------------------ */

const markdownComponents = {
  p: ({ children, ...props }: React.ComponentPropsWithoutRef<"p">) => (
    <p className="mb-2 last:mb-0 text-sm leading-relaxed" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }: React.ComponentPropsWithoutRef<"ul">) => (
    <ul className="mb-2 list-disc space-y-1 pl-5 text-sm" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: React.ComponentPropsWithoutRef<"ol">) => (
    <ol className="mb-2 list-decimal space-y-1 pl-5 text-sm" {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }: React.ComponentPropsWithoutRef<"li">) => (
    <li className="text-sm" {...props}>
      {children}
    </li>
  ),
  strong: ({ children, ...props }: React.ComponentPropsWithoutRef<"strong">) => (
    <strong className="font-semibold" {...props}>
      {children}
    </strong>
  ),
  a: ({ children, ...props }: React.ComponentPropsWithoutRef<"a">) => (
    <a className="text-primary underline" target="_blank" rel="noreferrer" {...props}>
      {children}
    </a>
  ),
};

/* ------------------------------------------------------------------ */
/*  Grouping helpers                                                   */
/* ------------------------------------------------------------------ */

const GROUP_WINDOW_MS = 2 * 60 * 1000; // 2 minutes

type GroupedMessage = ChatMessage & {
  showAvatar: boolean;
  showTimestamp: boolean;
  isFirstOfDay: boolean;
};

function groupMessages(messages: ChatMessage[]): GroupedMessage[] {
  const result: GroupedMessage[] = [];
  let prevDate: Date | null = null;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const prev = i > 0 ? messages[i - 1] : null;
    const msgDate = new Date(msg.timestamp);

    const isFirstOfDay = !prevDate || !isSameDay(msgDate, prevDate);
    prevDate = msgDate;

    const sameSender = prev && prev.role === msg.role && msg.role !== "system";
    const withinWindow =
      prev &&
      Math.abs(msgDate.getTime() - new Date(prev.timestamp).getTime()) < GROUP_WINDOW_MS;

    const isGrouped = sameSender && withinWindow && !isFirstOfDay;

    result.push({
      ...msg,
      showAvatar: !isGrouped,
      showTimestamp: !isGrouped,
      isFirstOfDay,
    });
  }

  return result;
}

function formatDayLabel(dateStr: string): string {
  const d = new Date(dateStr);
  if (isToday(d)) return "Today";
  if (isYesterday(d)) return "Yesterday";
  return format(d, "EEEE, MMMM d, yyyy");
}

/* ------------------------------------------------------------------ */
/*  Single message bubble                                              */
/* ------------------------------------------------------------------ */

function MessageBubble({ message }: { message: GroupedMessage }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  if (isSystem) {
    return (
      <div className="flex justify-center py-1.5">
        <span className="rounded-full bg-muted px-3 py-1 text-xs text-muted-foreground">
          {message.content}
        </span>
      </div>
    );
  }

  const time = format(new Date(message.timestamp), "HH:mm");

  return (
    <div
      className={cn(
        "flex gap-2.5 animate-in fade-in slide-in-from-bottom-2 duration-300",
        isUser && "flex-row-reverse",
        !message.showAvatar && (isUser ? "pr-[42px]" : "pl-[42px]"),
      )}
    >
      {message.showAvatar ? (
        <Avatar className="mt-0.5 h-8 w-8 shrink-0">
          <AvatarFallback className="text-xs font-medium">
            {isUser ? "U" : "AI"}
          </AvatarFallback>
        </Avatar>
      ) : null}
      <div className={cn("flex max-w-[75%] flex-col", isUser && "items-end")}>
        <div
          className={cn(
            "rounded-2xl px-3.5 py-2 text-sm shadow-sm",
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md"
              : "bg-muted text-foreground rounded-bl-md",
          )}
        >
          {message.imageDataUrl && (
            <img
              src={message.imageDataUrl}
              alt="Attached"
              className="mb-2 max-h-40 rounded-lg object-cover"
            />
          )}
          {isUser ? (
            <span className="whitespace-pre-wrap">{message.content}</span>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>
        {message.showTimestamp && (
          <span className="mt-1 px-1 text-[11px] text-muted-foreground">{time}</span>
        )}
        {message.status === "error" && message.error && (
          <Button variant="outline" size="sm" className="mt-1">
            Retry
          </Button>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Typing indicator                                                   */
/* ------------------------------------------------------------------ */

function TypingIndicator() {
  return (
    <div className="flex gap-2.5 animate-in fade-in duration-300">
      <Avatar className="mt-0.5 h-8 w-8 shrink-0">
        <AvatarFallback className="text-xs font-medium">AI</AvatarFallback>
      </Avatar>
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-md bg-muted px-4 py-3">
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:0ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:150ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:300ms]" />
        </div>
        <span className="px-1 text-[11px] text-muted-foreground">Thinking…</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  MessageList                                                        */
/* ------------------------------------------------------------------ */

const BOTTOM_THRESHOLD = 60; // px – user counts as "at bottom" within this range

export function MessageList() {
  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);

  const scrollContainerRef = React.useRef<HTMLDivElement>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const isAtBottomRef = React.useRef(true);
  const [showScrollBtn, setShowScrollBtn] = React.useState(false);
  const [hasNewMessages, setHasNewMessages] = React.useState(false);
  const prevMsgCountRef = React.useRef(messages.length);

  /* ---- check if user is near bottom ---- */
  const checkAtBottom = React.useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_THRESHOLD;
    isAtBottomRef.current = atBottom;
    setShowScrollBtn(!atBottom);
    if (atBottom) setHasNewMessages(false);
  }, []);

  /* ---- scroll handler ---- */
  React.useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const handler = () => checkAtBottom();
    el.addEventListener("scroll", handler, { passive: true });
    return () => el.removeEventListener("scroll", handler);
  }, [checkAtBottom]);

  /* ---- auto-scroll on new messages ---- */
  React.useEffect(() => {
    const isNewMessage = messages.length > prevMsgCountRef.current;
    prevMsgCountRef.current = messages.length;

    if (!isNewMessage && !isLoading) return;

    if (isAtBottomRef.current) {
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      });
    } else if (isNewMessage) {
      setHasNewMessages(true);
    }
  }, [messages, isLoading]);

  /* ---- initial scroll to bottom ---- */
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "instant" });
    checkAtBottom();
  }, []);

  /* ---- scroll-to-bottom handler ---- */
  const scrollToBottom = React.useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    setHasNewMessages(false);
  }, []);

  const grouped = React.useMemo(() => groupMessages(messages), [messages]);

  return (
    <div className="relative flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      {/* Scrollable message area */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto overscroll-contain scroll-smooth px-4 py-4 [scrollbar-gutter:stable] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/30 [&::-webkit-scrollbar-track]:bg-transparent"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-1">
          {messages.length === 0 && !isLoading && (
            <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
              Send a message to start the conversation.
            </div>
          )}

          {grouped.map((msg) => (
            <React.Fragment key={msg.id}>
              {msg.isFirstOfDay && (
                <div className="flex items-center gap-3 py-4">
                  <div className="h-px flex-1 bg-border" />
                  <span className="shrink-0 text-[11px] font-medium text-muted-foreground">
                    {formatDayLabel(msg.timestamp)}
                  </span>
                  <div className="h-px flex-1 bg-border" />
                </div>
              )}
              <div className={cn(msg.showAvatar ? "mt-3" : "mt-0.5")}>
                <MessageBubble message={msg} />
              </div>
            </React.Fragment>
          ))}

          {isLoading && (
            <div className="mt-3">
              <TypingIndicator />
            </div>
          )}

          {/* anchor for scrolling */}
          <div ref={bottomRef} className="h-px" />
        </div>
      </div>

      {/* Scroll-to-bottom FAB */}
      {showScrollBtn && (
        <button
          type="button"
          onClick={scrollToBottom}
          className={cn(
            "absolute bottom-3 right-4 z-10 flex items-center gap-1.5 rounded-full border border-border bg-background/90 px-3 py-1.5 text-xs font-medium text-muted-foreground shadow-lg backdrop-blur transition-all hover:bg-accent hover:text-accent-foreground",
            "animate-in fade-in zoom-in-95 duration-200",
          )}
        >
          <ArrowDown className="h-3.5 w-3.5" />
          {hasNewMessages ? "New messages" : "Scroll to bottom"}
        </button>
      )}
    </div>
  );
}
