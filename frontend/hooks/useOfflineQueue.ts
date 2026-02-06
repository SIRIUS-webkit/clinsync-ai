import * as React from "react";

import { clearQueue, getQueueCount } from "@/lib/offline-db";
import { useConnectionStore } from "@/stores/useConnectionStore";

export function useOfflineQueue() {
  const online = useConnectionStore((state) => state.online);
  const setQueueCount = useConnectionStore((state) => state.setQueueCount);
  const setLastSync = useConnectionStore((state) => state.setLastSync);

  React.useEffect(() => {
    let active = true;

    const updateQueue = async () => {
      const count = await getQueueCount();
      if (active) {
        setQueueCount(count);
      }
    };

    updateQueue();
    const interval = window.setInterval(updateQueue, 5000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [setQueueCount]);

  React.useEffect(() => {
    const sync = async () => {
      if (!online) return;
      await clearQueue();
      setQueueCount(0);
      setLastSync(new Date().toLocaleTimeString());
    };

    sync();
  }, [online, setLastSync, setQueueCount]);
}
