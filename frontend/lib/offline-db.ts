import { openDB } from "idb";

const DB_NAME = "clinsync-offline";
const DB_VERSION = 1;

type ConsultationRecord = {
  id: string;
  patientName: string;
  summary: string;
  createdAt: string;
};

type SyncRecord = {
  id: string;
  type: "consultation";
  payload: ConsultationRecord;
};

async function getDb() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains("consultations")) {
        db.createObjectStore("consultations", { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains("syncQueue")) {
        db.createObjectStore("syncQueue", { keyPath: "id" });
      }
    },
  });
}

export async function saveConsultation(record: ConsultationRecord) {
  const db = await getDb();
  await db.put("consultations", record);
}

export async function listConsultations() {
  const db = await getDb();
  return db.getAll("consultations");
}

export async function enqueueSync(record: SyncRecord) {
  const db = await getDb();
  await db.put("syncQueue", record);
}

export async function getQueueCount() {
  const db = await getDb();
  const keys = await db.getAllKeys("syncQueue");
  return keys.length;
}

export async function clearQueue() {
  const db = await getDb();
  const tx = db.transaction("syncQueue", "readwrite");
  await tx.store.clear();
  await tx.done;
}
