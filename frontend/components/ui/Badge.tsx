import * as React from "react"
import { cn } from "@/lib/utils"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info" | "neutral";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variants = {
    default: "border-transparent bg-slate-800 text-slate-100",
    success: "border-transparent bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
    warning: "border-transparent bg-amber-500/20 text-amber-400 border border-amber-500/30",
    danger: "border-transparent bg-red-500/20 text-red-400 border border-red-500/30",
    info: "border-transparent bg-blue-500/20 text-blue-400 border border-blue-500/30",
    neutral: "border-transparent bg-slate-700 text-slate-300",
  }

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2",
        variants[variant],
        className
      )}
      {...props}
    />
  )
}

export { Badge }
