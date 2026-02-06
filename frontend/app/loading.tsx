import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {[...Array(3)].map((_, index) => (
        <div key={index} className="space-y-4">
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ))}
    </div>
  );
}
