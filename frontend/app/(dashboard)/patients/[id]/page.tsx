"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { usePatientStore } from "@/stores/usePatientStore";

export default function PatientDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const patientList = usePatientStore((s) => s.patientList);
  const setSelectedPatient = usePatientStore((s) => s.setSelectedPatient);
  const patient = React.useMemo(() => patientList.find((p) => p.id === id), [patientList, id]);

  React.useEffect(() => {
    if (patient) setSelectedPatient(patient);
    return () => setSelectedPatient(null);
  }, [patient, setSelectedPatient]);

  if (!patient) {
    return (
      <div className="space-y-4">
        <Button asChild variant="ghost">
          <Link href="/patients">← Back to patients</Link>
        </Button>
        <p className="text-muted-foreground">Patient not found.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <Button asChild variant="ghost">
          <Link href="/patients">← Back to patients</Link>
        </Button>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{patient.name}</CardTitle>
          <p className="text-sm text-muted-foreground">ID: {patient.id}</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-muted-foreground">Age</p>
            <p className="text-sm">{patient.age}</p>
          </div>
          {patient.symptoms && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground">Symptoms</p>
              <p className="text-sm">{patient.symptoms}</p>
            </div>
          )}
          {patient.history && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground">History</p>
              <p className="text-sm">{patient.history}</p>
            </div>
          )}
          {patient.lastConsult && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground">Last consultation</p>
              <p className="text-sm">{patient.lastConsult}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
