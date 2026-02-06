import * as React from "react";

import { healthCheck } from "@/lib/api";
import { useConnectionStore } from "@/stores/useConnectionStore";

export function useApiHealth() {
  const setApiOnline = useConnectionStore((state) => state.setApiOnline);

  React.useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        await healthCheck();
        if (active) setApiOnline(true);
      } catch (error) {
        if (active) setApiOnline(false);
      }
    };

    check();
    const interval = window.setInterval(check, 15000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [setApiOnline]);
}
