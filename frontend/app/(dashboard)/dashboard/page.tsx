"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { MessageSquare, Mic, Stethoscope } from "lucide-react";

import { OfflineSyncIndicator } from "@/components/offline/OfflineSyncIndicator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useConnectionStore } from "@/stores/useConnectionStore";

const recentConsultations = [
  { id: "1", patient: "Jane Doe", mode: "Video", time: "10:30 AM", date: "Today" },
  { id: "2", patient: "John Smith", mode: "Chat", time: "9:15 AM", date: "Today" },
  { id: "3", patient: "Maria Garcia", mode: "Voice", time: "4:45 PM", date: "Yesterday" },
  { id: "4", patient: "Robert Lee", mode: "Video", time: "2:00 PM", date: "Yesterday" },
  { id: "5", patient: "Sarah Kim", mode: "Chat", time: "11:20 AM", date: "Feb 3" },
];

export default function DashboardPage() {
  const apiOnline = useConnectionStore((s) => s.apiOnline);

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
            <p className="text-muted-foreground mt-1">
              {apiOnline ? (
                <Badge variant="success" className="font-normal">System online</Badge>
              ) : (
                <Badge variant="secondary" className="font-normal">Checking…</Badge>
              )}
            </p>
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Start Consultation</CardTitle>
            <p className="text-sm text-muted-foreground">
              Choose a mode to begin a new consultation session.
            </p>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-3">
            <Button asChild size="lg" className="h-auto flex-col gap-2 py-6">
              <Link href="/consultation/video">
                <Stethoscope className="h-8 w-8" aria-hidden />
                <span>Video</span>
                <span className="text-xs font-normal opacity-90">WebRTC with AI overlay</span>
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="h-auto flex-col gap-2 py-6">
              <Link href="/consultation/chat">
                <MessageSquare className="h-8 w-8" aria-hidden />
                <span>Chat</span>
                <span className="text-xs font-normal opacity-90">Text + image analysis</span>
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="h-auto flex-col gap-2 py-6">
              <Link href="/consultation/voice">
                <Mic className="h-8 w-8" aria-hidden />
                <span>Voice</span>
                <span className="text-xs font-normal opacity-90">Audio triage & biomarkers</span>
              </Link>
            </Button>
          </CardContent>
        </Card>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="grid gap-6 lg:grid-cols-2"
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Consultations</CardTitle>
            <p className="text-sm text-muted-foreground">Last 5 sessions</p>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {recentConsultations.map((c) => (
                <li
                  key={c.id}
                  className="flex items-center justify-between rounded-md border border-border bg-muted/20 px-3 py-2 text-sm"
                >
                  <div>
                    <p className="font-medium">{c.patient}</p>
                    <p className="text-xs text-muted-foreground">
                      {c.mode} · {c.date} {c.time}
                    </p>
                  </div>
                  <Badge variant="outline">{c.mode}</Badge>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Quick Stats</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold">12</p>
                <p className="text-xs text-muted-foreground">Patients today</p>
              </div>
              <div>
                <p className="text-2xl font-bold">8</p>
                <p className="text-xs text-muted-foreground">AI analyses</p>
              </div>
              <div>
                <p className="text-2xl font-bold">2m</p>
                <p className="text-xs text-muted-foreground">Avg wait</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Notifications</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>No new alerts.</p>
              <p>All systems operational.</p>
            </CardContent>
          </Card>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.15 }}
      >
        <OfflineSyncIndicator />
      </motion.section>
    </div>
  );
}
