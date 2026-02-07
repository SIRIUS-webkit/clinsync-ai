"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import toast from "react-hot-toast";
import {
  User,
  HeartPulse,
  Stethoscope,
  Camera,
  Upload,
  X,
  ChevronRight,
  ChevronLeft,
  Video,
  AlertCircle,
  Pill,
  Activity,
  Brain,
} from "lucide-react";
import { generatePatientId, PatientIntake } from "@/stores/useConsultationContext";

const SYMPTOM_OPTIONS = [
  "Fever", "Cough", "Headache", "Fatigue", "Nausea",
  "Rash", "Itching", "Pain", "Swelling", "Breathing difficulty",
  "Dizziness", "Chest pain", "Abdominal pain", "Joint pain", "Back pain"
];

const CONSULTATION_TYPES = [
  { id: "general", label: "General Checkup", icon: Stethoscope, color: "bg-blue-500" },
  { id: "skin", label: "Skin Condition", icon: Camera, color: "bg-orange-500" },
  { id: "respiratory", label: "Respiratory", icon: Activity, color: "bg-teal-500" },
  { id: "cardiology", label: "Heart/Cardiology", icon: HeartPulse, color: "bg-red-500" },
  { id: "other", label: "Other", icon: Brain, color: "bg-purple-500" },
] as const;

interface IntakeFormProps {
  onComplete: (intake: PatientIntake) => void;
}

export function IntakeForm({ onComplete }: IntakeFormProps) {
  const [step, setStep] = React.useState(1);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  
  // Form state
  const [patientId] = React.useState(() => generatePatientId());
  const [fullName, setFullName] = React.useState("");
  const [age, setAge] = React.useState("");
  const [gender, setGender] = React.useState("");
  const [chiefComplaint, setChiefComplaint] = React.useState("");
  const [selectedSymptoms, setSelectedSymptoms] = React.useState<string[]>([]);
  const [symptomDuration, setSymptomDuration] = React.useState("");
  const [painLevel, setPainLevel] = React.useState(0);
  const [allergies, setAllergies] = React.useState("");
  const [currentMedications, setCurrentMedications] = React.useState("");
  const [medicalHistory, setMedicalHistory] = React.useState("");
  const [consultationType, setConsultationType] = React.useState<PatientIntake["consultationType"]>("general");
  const [attachedImages, setAttachedImages] = React.useState<PatientIntake["attachedImages"]>([]);
  
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const toggleSymptom = (symptom: string) => {
    setSelectedSymptoms((prev) =>
      prev.includes(symptom)
        ? prev.filter((s) => s !== symptom)
        : [...prev, symptom]
    );
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      if (file.size > 10 * 1024 * 1024) {
        toast.error("Image too large. Max 10MB.");
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        setAttachedImages((prev) => [
          ...prev,
          {
            id: Math.random().toString(36).slice(2),
            name: file.name,
            type: consultationType === "skin" ? "skin" : "other",
            dataUrl: reader.result as string,
          },
        ]);
      };
      reader.readAsDataURL(file);
    });

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeImage = (id: string) => {
    setAttachedImages((prev) => prev.filter((img) => img.id !== id));
  };

  const validateStep = (stepNum: number): boolean => {
    switch (stepNum) {
      case 1:
        if (!fullName.trim()) {
          toast.error("Please enter your name");
          return false;
        }
        if (!age || parseInt(age) < 1 || parseInt(age) > 120) {
          toast.error("Please enter a valid age");
          return false;
        }
        if (!gender) {
          toast.error("Please select your gender");
          return false;
        }
        return true;
      case 2:
        if (!chiefComplaint.trim()) {
          toast.error("Please describe your main concern");
          return false;
        }
        return true;
      case 3:
        return true;
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (validateStep(step)) {
      setStep((prev) => Math.min(prev + 1, 4));
    }
  };

  const handleBack = () => {
    setStep((prev) => Math.max(prev - 1, 1));
  };

  const handleStartConsultation = async () => {
    if (!validateStep(step)) return;

    setIsSubmitting(true);

    try {
      const intake: PatientIntake = {
        patientId,
        fullName,
        age,
        gender,
        chiefComplaint,
        symptoms: selectedSymptoms,
        symptomDuration,
        painLevel,
        allergies,
        currentMedications,
        medicalHistory,
        attachedImages,
        consultationType,
        createdAt: new Date().toISOString(),
      };

      onComplete(intake);
    } catch (error) {
      toast.error("Failed to start consultation");
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-1 flex-col bg-gradient-to-br from-slate-900 via-blue-950 to-purple-950 p-4 md:p-8 overflow-auto">
      <div className="w-full max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center"
          >
            <Video className="w-7 h-7 text-white" />
          </motion.div>
          <h1 className="text-2xl font-bold text-white mb-1">Video Consultation</h1>
          <p className="text-blue-300 text-sm">Let us know about your health concern before we begin</p>
        </div>

        {/* Progress */}
        <div className="flex items-center justify-center gap-2 mb-6">
          {[1, 2, 3, 4].map((s) => (
            <React.Fragment key={s}>
              <motion.div
                animate={{
                  scale: step === s ? 1.2 : 1,
                  backgroundColor: step >= s ? "#3b82f6" : "#475569",
                }}
                className={cn(
                  "w-9 h-9 rounded-full flex items-center justify-center text-white font-semibold text-sm",
                  step >= s ? "bg-blue-500" : "bg-slate-600"
                )}
              >
                {s}
              </motion.div>
              {s < 4 && (
                <div
                  className={cn(
                    "w-10 h-1 rounded-full transition-colors",
                    step > s ? "bg-blue-500" : "bg-slate-600"
                  )}
                />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Form Card */}
        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
          <CardContent className="p-5">
            <AnimatePresence mode="wait">
              {/* Step 1: Basic Info */}
              {step === 1 && (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-5"
                >
                  <div className="flex items-center gap-3 mb-5">
                    <div className="p-2 rounded-lg bg-blue-500/20">
                      <User className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">Basic Information</h2>
                      <p className="text-sm text-slate-400">Tell us about yourself</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="md:col-span-2">
                      <Label className="text-slate-300">Patient ID</Label>
                      <Input
                        value={patientId}
                        disabled
                        className="bg-slate-700/50 border-slate-600 text-slate-400 mt-1"
                      />
                    </div>

                    <div className="md:col-span-2">
                      <Label className="text-slate-300">Full Name *</Label>
                      <Input
                        value={fullName}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setFullName(e.target.value)}
                        placeholder="Enter your full name"
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-slate-300">Age *</Label>
                      <Input
                        type="number"
                        value={age}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAge(e.target.value)}
                        placeholder="Years"
                        min="1"
                        max="120"
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-slate-300">Gender *</Label>
                      <div className="flex gap-2 mt-1">
                        {["Male", "Female", "Other"].map((g) => (
                          <Button
                            key={g}
                            type="button"
                            variant={gender === g ? "default" : "outline"}
                            size="sm"
                            onClick={() => setGender(g)}
                            className={cn(
                              "flex-1",
                              gender === g
                                ? "bg-blue-600"
                                : "border-slate-600 text-slate-300 hover:bg-slate-700"
                            )}
                          >
                            {g}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 2: Chief Complaint */}
              {step === 2 && (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-5"
                >
                  <div className="flex items-center gap-3 mb-5">
                    <div className="p-2 rounded-lg bg-orange-500/20">
                      <AlertCircle className="w-5 h-5 text-orange-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">Your Concern</h2>
                      <p className="text-sm text-slate-400">What brings you here today?</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label className="text-slate-300">Consultation Type</Label>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-2">
                        {CONSULTATION_TYPES.map(({ id, label, icon: Icon, color }) => (
                          <Button
                            key={id}
                            type="button"
                            variant={consultationType === id ? "default" : "outline"}
                            onClick={() => setConsultationType(id as PatientIntake["consultationType"])}
                            className={cn(
                              "h-auto py-2 flex flex-col items-center gap-1",
                              consultationType === id
                                ? color
                                : "border-slate-600 text-slate-300 hover:bg-slate-700"
                            )}
                          >
                            <Icon className="w-4 h-4" />
                            <span className="text-[10px]">{label}</span>
                          </Button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <Label className="text-slate-300">Main Concern *</Label>
                      <Textarea
                        value={chiefComplaint}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setChiefComplaint(e.target.value)}
                        placeholder="Describe your main health concern or symptoms..."
                        rows={3}
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 resize-none mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-slate-300">Common Symptoms</Label>
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {SYMPTOM_OPTIONS.map((symptom) => (
                          <Badge
                            key={symptom}
                            variant={selectedSymptoms.includes(symptom) ? "default" : "outline"}
                            className={cn(
                              "cursor-pointer transition-colors text-xs",
                              selectedSymptoms.includes(symptom)
                                ? "bg-blue-600 hover:bg-blue-700"
                                : "border-slate-600 text-slate-300 hover:bg-slate-700"
                            )}
                            onClick={() => toggleSymptom(symptom)}
                          >
                            {symptom}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 3: Additional Details */}
              {step === 3 && (
                <motion.div
                  key="step3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-5"
                >
                  <div className="flex items-center gap-3 mb-5">
                    <div className="p-2 rounded-lg bg-teal-500/20">
                      <Pill className="w-5 h-5 text-teal-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">Medical Details</h2>
                      <p className="text-sm text-slate-400">Help us understand your health better</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-slate-300">Symptom Duration</Label>
                      <Input
                        value={symptomDuration}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSymptomDuration(e.target.value)}
                        placeholder="e.g., 3 days"
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-slate-300">Pain Level (1-10)</Label>
                      <div className="flex gap-1 mt-1">
                        {[...Array(10)].map((_, i) => (
                          <Button
                            key={i}
                            type="button"
                            size="sm"
                            variant={painLevel === i + 1 ? "default" : "outline"}
                            onClick={() => setPainLevel(i + 1)}
                            className={cn(
                              "w-7 h-7 p-0 text-xs",
                              painLevel === i + 1
                                ? i < 3
                                  ? "bg-green-600"
                                  : i < 6
                                  ? "bg-yellow-600"
                                  : "bg-red-600"
                                : "border-slate-600 text-slate-400"
                            )}
                          >
                            {i + 1}
                          </Button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <Label className="text-slate-300">Allergies</Label>
                      <Input
                        value={allergies}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAllergies(e.target.value)}
                        placeholder="Any known allergies..."
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-slate-300">Current Medications</Label>
                      <Input
                        value={currentMedications}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCurrentMedications(e.target.value)}
                        placeholder="Medications you're currently taking..."
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 mt-1"
                      />
                    </div>

                    <div className="md:col-span-2">
                      <Label className="text-slate-300">Medical History</Label>
                      <Textarea
                        value={medicalHistory}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setMedicalHistory(e.target.value)}
                        placeholder="Any past surgeries, chronic conditions..."
                        rows={2}
                        className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 resize-none mt-1"
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 4: Images & Review */}
              {step === 4 && (
                <motion.div
                  key="step4"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-5"
                >
                  <div className="flex items-center gap-3 mb-5">
                    <div className="p-2 rounded-lg bg-purple-500/20">
                      <Camera className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">Upload Images (Optional)</h2>
                      <p className="text-sm text-slate-400">Share any relevant photos</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-slate-600 rounded-xl p-6 text-center cursor-pointer hover:border-blue-500/50 hover:bg-slate-700/30 transition-colors"
                    >
                      <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                      <p className="text-slate-300 text-sm mb-1">Click to upload images</p>
                      <p className="text-xs text-slate-500">Skin photos, X-rays, lab results, etc.</p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handleImageUpload}
                        className="hidden"
                      />
                    </div>

                    {attachedImages.length > 0 && (
                      <div className="grid grid-cols-4 gap-2">
                        {attachedImages.map((img) => (
                          <div
                            key={img.id}
                            className="relative rounded-lg overflow-hidden border border-slate-600"
                          >
                            <img
                              src={img.dataUrl}
                              alt={img.name}
                              className="w-full h-20 object-cover"
                            />
                            <button
                              onClick={() => removeImage(img.id)}
                              className="absolute top-1 right-1 w-5 h-5 rounded-full bg-red-500 flex items-center justify-center hover:bg-red-600"
                            >
                              <X className="w-3 h-3 text-white" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}

                    <Card className="bg-slate-700/50 border-slate-600">
                      <CardHeader className="pb-2 pt-3 px-4">
                        <CardTitle className="text-base text-white">Summary</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm text-slate-300 space-y-1 px-4 pb-3">
                        <p><strong>Patient:</strong> {fullName}, {age}y, {gender}</p>
                        <p><strong>Type:</strong> {CONSULTATION_TYPES.find(t => t.id === consultationType)?.label}</p>
                        <p><strong>Concern:</strong> {chiefComplaint.substring(0, 80)}{chiefComplaint.length > 80 ? "..." : ""}</p>
                        {selectedSymptoms.length > 0 && (
                          <p><strong>Symptoms:</strong> {selectedSymptoms.join(", ")}</p>
                        )}
                        {attachedImages.length > 0 && (
                          <p><strong>Images:</strong> {attachedImages.length} attached</p>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Navigation */}
            <div className="flex justify-between mt-6 pt-5 border-t border-slate-700">
              <Button
                variant="outline"
                onClick={handleBack}
                disabled={step === 1}
                className="border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back
              </Button>

              {step < 4 ? (
                <Button onClick={handleNext} className="bg-blue-600 hover:bg-blue-700">
                  Next
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              ) : (
                <Button
                  onClick={handleStartConsultation}
                  disabled={isSubmitting}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                >
                  {isSubmitting ? (
                    "Starting..."
                  ) : (
                    <>
                      <Video className="w-4 h-4 mr-2" />
                      Start Video Consultation
                    </>
                  )}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
