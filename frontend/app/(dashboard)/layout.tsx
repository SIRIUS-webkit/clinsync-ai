"use client";

import { usePathname } from "next/navigation";
import { ConnectionStatusBar } from "@/components/layout/ConnectionStatusBar";
import { MobileNav } from "@/components/layout/MobileNav";
import { SidebarNav } from "@/components/layout/SidebarNav";
import { ThemeToggle } from "@/components/layout/ThemeToggle";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isConsultation = pathname?.startsWith("/consultation") ?? false;

  return (
    <>
      <ConnectionStatusBar />
      <div className="flex h-screen overflow-hidden bg-muted/30">
        <SidebarNav />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {!isConsultation && (
            <header className="flex shrink-0 items-center justify-between border-b border-border bg-background px-4 py-3 lg:px-6">
              <div className="flex items-center gap-3">
                <div className="md:hidden">
                  <MobileNav />
                </div>
                <div>
                  <p className="text-sm font-semibold">ClinSync AI Console</p>
                  <p className="text-xs text-muted-foreground">Medical virtual assistant workspace</p>
                </div>
              </div>
              <ThemeToggle />
            </header>
          )}
          {isConsultation ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">{children}</div>
          ) : (
            <main className="flex-1 p-4 lg:p-6">{children}</main>
          )}
        </div>
      </div>
    </>
  );
}
