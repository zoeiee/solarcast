'use client';

import React from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';

interface MagnetogramViewerProps {
  src?: string;
  alt?: string;
  className?: string;
}

export function MagnetogramViewer({ src, alt = "Magnetogram", className }: MagnetogramViewerProps) {
  return (
    <div className={cn("relative w-full aspect-square rounded-xl overflow-hidden bg-slate-900 border border-slate-700 flex items-center justify-center group", className)}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img 
          src={src} 
          alt={alt}
          className="object-contain w-full h-full transition-transform duration-500 group-hover:scale-105"
        />
      ) : (
        <div className="flex flex-col items-center justify-center text-slate-500">
          <div className="w-16 h-16 rounded-full border-2 border-dashed border-slate-700 mb-2"></div>
          <span>No image available</span>
        </div>
      )}
      
      {/* Optional scale overlay */}
      <div className="absolute bottom-4 right-4 bg-slate-950/80 backdrop-blur border border-slate-700 rounded-md p-2 text-xs flex flex-col gap-1 z-10">
        <div className="flex justify-between w-24">
          <span>-B</span>
          <span>0</span>
          <span>+B</span>
        </div>
        <div className="h-2 w-full rounded-sm bg-gradient-to-r from-blue-600 via-white to-red-600"></div>
      </div>
    </div>
  );
}
