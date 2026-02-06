import * as React from "react";

import { useConnectionStore } from "@/stores/useConnectionStore";

export function useOnlineStatus() {
  const setOnline = useConnectionStore((state) => state.setOnline);

  React.useEffect(() => {
    const update = () => {
      setOnline(navigator.onLine);
    };

    update();
    window.addEventListener("online", update);
    window.addEventListener("offline", update);

    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, [setOnline]);
}
