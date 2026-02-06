"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePatientStore } from "@/stores/usePatientStore";
import { Button } from "@/components/ui/button";

export default function PatientsPage() {
  const patientList = usePatientStore((s) => s.patientList);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Patients</h1>
        <p className="text-muted-foreground mt-1">View and manage patient list.</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent patients</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {patientList.map((p) => (
              <li key={p.id} className="flex items-center justify-between rounded-md border border-border bg-muted/20 px-4 py-3">
                <div>
                  <p className="font-medium">{p.name}</p>
                  <p className="text-sm text-muted-foreground">Age {p.age} · {p.lastConsult && `Last consult ${p.lastConsult}`}</p>
                </div>
                <Button asChild variant="outline" size="sm">
                  <Link href={`/patients/${p.id}`}>View</Link>
                </Button>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
