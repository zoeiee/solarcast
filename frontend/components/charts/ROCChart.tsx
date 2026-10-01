'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { cn } from '@/lib/utils';

interface ROCChartProps {
  data: { fpr: number; tpr: number }[];
  auc: number;
  className?: string;
}

export function ROCChart({ data, auc, className }: ROCChartProps) {
  return (
    <div className={cn("w-full h-[300px]", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
          <XAxis 
            dataKey="fpr" 
            type="number" 
            domain={[0, 1]} 
            stroke="#94a3b8"
            label={{ value: 'False Positive Rate', position: 'bottom', fill: '#94a3b8' }}
          />
          <YAxis 
            type="number" 
            domain={[0, 1]} 
            stroke="#94a3b8"
            label={{ value: 'True Positive Rate', angle: -90, position: 'left', fill: '#94a3b8' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem' }}
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={(label: number) => `FPR: ${label.toFixed(3)}`}
          />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#64748b" strokeDasharray="3 3" />
          <Line 
            type="monotone" 
            dataKey="tpr" 
            stroke="#f59e0b" 
            strokeWidth={3}
            dot={false}
            activeDot={{ r: 6, fill: '#f59e0b', stroke: '#020617' }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="text-center text-sm text-slate-300 mt-2">
        ROC AUC: <span className="font-bold text-amber-500">{auc.toFixed(3)}</span>
      </div>
    </div>
  );
}
