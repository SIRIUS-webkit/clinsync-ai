"use client";

import Link from "next/link";
import { Menu } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const navItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/consultation/video", label: "Video Consult" },
  { href: "/consultation/chat", label: "Chat" },
  { href: "/consultation/voice", label: "Voice" },
  { href: "/patients", label: "Patients" },
  { href: "/insights", label: "AI Insights" },
  { href: "/settings", label: "Settings" },
];

export function MobileNav() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Open navigation menu">
          <Menu className="h-5 w-5" aria-hidden />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Navigation</DialogTitle>
        </DialogHeader>
        <nav className="flex flex-col gap-2">
          {navItems.map((item) => (
            <Button key={item.label} asChild variant="ghost" className="justify-start">
              <Link href={item.href}>{item.label}</Link>
            </Button>
          ))}
        </nav>
      </DialogContent>
    </Dialog>
  );
}
