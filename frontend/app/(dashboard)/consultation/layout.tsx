import { ConsultationHeader } from "@/components/consultation/ConsultationHeader";

export default function ConsultationLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden border-t border-border bg-card">
      <ConsultationHeader />
      <div className="flex min-h-0 flex-1">{children}</div>
    </div>
  );
}
