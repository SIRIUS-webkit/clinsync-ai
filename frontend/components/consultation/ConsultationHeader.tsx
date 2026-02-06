"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowLeft, MessageSquare, Mic, Stethoscope, Wifi } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MobileNav } from "@/components/layout/MobileNav";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { usePatientStore } from "@/stores/usePatientStore";
import { cn } from "@/lib/utils";

const modeTabs = [
  { value: "video", label: "Video", href: "/consultation/video", icon: Stethoscope },
  { value: "chat", label: "Chat", href: "/consultation/chat", icon: MessageSquare },
  { value: "voice", label: "Voice", href: "/consultation/voice", icon: Mic },
] as const;

export function ConsultationHeader() {
  const pathname = usePathname();
  const selectedPatient = usePatientStore((s) => s.selectedPatient);
  const patientName = selectedPatient?.name ?? "No patient selected";

  const currentMode = modeTabs.find((t) => pathname.startsWith(t.href))?.value ?? "video";

  return (
    <header className="flex shrink-0 flex-wrap items-center justify-between gap-4 border-b border-border bg-card/80 px-4 py-3 backdrop-blur">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild aria-label="Back to dashboard">
          <Link href="/dashboard">
            <ArrowLeft className="h-4 w-4" aria-hidden />
          </Link>
        </Button>
        <div className="flex items-center gap-3">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="text-xs">
              {patientName.slice(0, 2).toUpperCase()}
            </AvatarFallback>
          </Avatar>
          <div>
            <p className="text-sm font-medium">{patientName}</p>
            <p className="text-xs text-muted-foreground">Consultation in progress</p>
          </div>
        </div>
        <Badge variant="outline" className="hidden sm:flex">
          <Wifi className="mr-1 h-3 w-3" aria-hidden />
          Connected
        </Badge>
        <div className="flex items-center gap-1 md:hidden">
          <MobileNav />
          <ThemeToggle />
        </div>
      </div>

      <Tabs value={currentMode} className="w-full sm:w-auto">
        <TabsList className="grid w-full grid-cols-3 sm:inline-flex">
          {modeTabs.map((tab) => {
            const isActive = pathname.startsWith(tab.href);
            const Icon = tab.icon;
            return (
              <TabsTrigger key={tab.value} value={tab.value} asChild>
                <Link
                  href={tab.href}
                  className={cn(
                    "flex items-center gap-2",
                    isActive && "bg-accent text-accent-foreground"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" aria-hidden />
                  {tab.label}
                </Link>
              </TabsTrigger>
            );
          })}
        </TabsList>
      </Tabs>
    </header>
  );
}
