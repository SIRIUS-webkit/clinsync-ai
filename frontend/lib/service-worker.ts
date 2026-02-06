export async function registerServiceWorker() {
  if (typeof window === "undefined") return;
  if (!("serviceWorker" in navigator)) return;

  try {
    await navigator.serviceWorker.register("/sw.js");
  } catch (error) {
    console.warn("Service worker registration failed", error);
  }
}
