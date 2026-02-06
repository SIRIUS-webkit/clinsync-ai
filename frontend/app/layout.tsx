import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import { AppProviders } from "@/components/providers/app-providers";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ClinSync AI",
  description: "AI-powered medical virtual assistant dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <AppProviders>{children}</AppProviders>
      </body>
    </html>
  );
}
