'use client';

import React, { useState } from 'react';
import { cn } from '@/lib/utils';
import { Info } from 'lucide-react';

interface GradCAMOverlayProps {
  originalSrc: string;
  heatmapSrc: string;
  overlaySrc: string;
  explanationNote?: string;
  className?: string;
}

export function GradCAMOverlay({ 
  originalSrc, 
  heatmapSrc, 
  overlaySrc, 
  explanationNote = "Highlighted regions indicate areas that influenced the prediction. They do not prove physical causation.",
  className 
}: GradCAMOverlayProps) {
  const [activeTab, setActiveTab] = useState<'original' | 'heatmap' | 'overlay'>('overlay');

  const tabs = [
    { id: 'original', label: 'Original' },
    { id: 'heatmap', label: 'Heatmap' },
    { id: 'overlay', label: 'Overlay' },
  ] as const;

  const currentSrc = 
    activeTab === 'original' ? originalSrc : 
    activeTab === 'heatmap' ? heatmapSrc : 
    overlaySrc;

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex p-1 bg-slate-900 rounded-lg w-fit border border-slate-800">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
              activeTab === tab.id 
                ? "bg-slate-700 text-white shadow" 
                : "text-slate-400 hover:text-slate-200"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="relative w-full aspect-square rounded-xl overflow-hidden bg-slate-900 border border-slate-700">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img 
          src={currentSrc} 
          alt={`${activeTab} view`}
          className="object-contain w-full h-full"
        />
      </div>

      <div className="flex items-start gap-3 bg-blue-950/30 border border-blue-900/50 p-4 rounded-lg text-sm text-blue-200">
        <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <p>{explanationNote}</p>
      </div>
    </div>
  );
}
