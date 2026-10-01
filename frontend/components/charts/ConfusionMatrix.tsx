'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface ConfusionMatrixProps {
  matrix: [[number, number], [number, number]];
  className?: string;
}

export function ConfusionMatrix({ matrix, className }: ConfusionMatrixProps) {
  // matrix[0][0] = TN, matrix[0][1] = FP
  // matrix[1][0] = FN, matrix[1][1] = TP
  
  const tn = matrix[0][0];
  const fp = matrix[0][1];
  const fn = matrix[1][0];
  const tp = matrix[1][1];
  
  const total = tn + fp + fn + tp;
  
  const getOpacity = (val: number) => Math.max(0.2, val / total);

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <div className="grid grid-cols-[auto_1fr] gap-2">
        <div />
        <div className="grid grid-cols-2 text-center text-sm font-medium text-slate-400 mb-2">
          <div>Predicted Negative</div>
          <div>Predicted Positive</div>
        </div>
        
        <div className="flex flex-col justify-around text-sm font-medium text-slate-400 pr-2">
          <div className="-rotate-90 origin-center whitespace-nowrap mb-8">Actual Negative</div>
          <div className="-rotate-90 origin-center whitespace-nowrap mt-8">Actual Positive</div>
        </div>
        
        <div className="grid grid-cols-2 gap-1 bg-slate-800 p-1 rounded-lg">
          <div 
            className="flex flex-col items-center justify-center p-6 rounded-md bg-blue-500/20 border border-blue-500/30 min-h-[100px]"
            style={{ backgroundColor: `rgba(59, 130, 246, ${getOpacity(tn)})` }}
          >
            <span className="text-2xl font-bold text-white">{tn}</span>
            <span className="text-xs text-white/80">True Negative</span>
          </div>
          
          <div 
            className="flex flex-col items-center justify-center p-6 rounded-md bg-red-500/20 border border-red-500/30 min-h-[100px]"
            style={{ backgroundColor: `rgba(239, 68, 68, ${getOpacity(fp)})` }}
          >
            <span className="text-2xl font-bold text-white">{fp}</span>
            <span className="text-xs text-white/80">False Positive</span>
          </div>
          
          <div 
            className="flex flex-col items-center justify-center p-6 rounded-md bg-amber-500/20 border border-amber-500/30 min-h-[100px]"
            style={{ backgroundColor: `rgba(245, 158, 11, ${getOpacity(fn)})` }}
          >
            <span className="text-2xl font-bold text-white">{fn}</span>
            <span className="text-xs text-white/80">False Negative</span>
          </div>
          
          <div 
            className="flex flex-col items-center justify-center p-6 rounded-md bg-emerald-500/20 border border-emerald-500/30 min-h-[100px]"
            style={{ backgroundColor: `rgba(16, 185, 129, ${getOpacity(tp)})` }}
          >
            <span className="text-2xl font-bold text-white">{tp}</span>
            <span className="text-xs text-white/80">True Positive</span>
          </div>
        </div>
      </div>
    </div>
  );
}
