"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Activity, LayoutDashboard, MessageSquare, Mic, Settings, Stethoscope, Users } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const navItems: { href: string; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/consultation/video", label: "Video Consult", icon: Stethoscope },
  { href: "/consultation/chat", label: "Chat", icon: MessageSquare },
  { href: "/consultation/voice", label: "Voice", icon: Mic },
  { href: "/patients", label: "Patients", icon: Users },
  { href: "/insights", label: "AI Insights", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/dashboard") return pathname === "/dashboard";
  if (href === "/consultation/video") {
    return pathname === "/consultation/video" || (pathname.startsWith("/consultation/") && !pathname.startsWith("/consultation/chat") && !pathname.startsWith("/consultation/voice"));
  }
  if (href === "/consultation/chat") return pathname === "/consultation/chat";
  if (href === "/consultation/voice") return pathname === "/consultation/voice";
  return pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
}

export function SidebarNav() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-card/80 p-4 md:flex">
      <div className="flex shrink-0 items-center gap-3 pb-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Stethoscope className="h-5 w-5" aria-hidden />
        </div>
        <div>
          <p className="text-sm font-semibold">ClinSync AI</p>
          <p className="text-xs text-muted-foreground">Virtual Assistant</p>
        </div>
      </div>
      <nav className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
        {navItems.map((item) => {
          const active = isActive(pathname, item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent",
                active ? "bg-accent text-foreground" : "text-muted-foreground"
              )}
            >
              {active && (
                <motion.span
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1/2 h-6 w-1 -translate-y-1/2 rounded-r-full bg-primary"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <Icon className="relative h-4 w-4 shrink-0" aria-hidden />
              <span className="relative">{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-6 shrink-0 rounded-lg border border-border bg-background p-4">
        <div className="flex items-center gap-3">
          <Avatar>
            <AvatarFallback>DR</AvatarFallback>
          </Avatar>
          <div>
            <p className="text-sm font-medium">Dr. Rivera</p>
            <p className="text-xs text-muted-foreground">Cardiology</p>
          </div>
        </div>
        <Badge variant="secondary" className="mt-3 w-fit">
          On Call
        </Badge>
      </div>
    </aside>
  );
}
